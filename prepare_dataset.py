"""
Build the CNN-ready image manifest from a (manually reviewed) quality report.

    python prepare_dataset.py --report outputs/full_run_v1/quality_report.csv

Decision rule for each image (every decision is written with its reason):
  1. manual_decision == EXCLUDE                 -> excluded ("manual: <notes>")
  2. manual_decision == INCLUDE                 -> included (overrides flags)
  3. flag in --exclude-statuses                 -> excluded (automatic)
     (default: CORRUPTED only; e.g. add BLURRY OVEREXPOSED after you have
     reviewed the contact sheets and trust the calibrated thresholds)
  4. EXACT_DUPLICATE                            -> the first file of each
     duplicate group is kept, the others excluded
  5. everything else (OK, REVIEW, POSSIBLE_NEAR_DUPLICATE) -> included
     Use --strict to also exclude un-reviewed REVIEW / near-duplicate images.

Outputs (in --out, default <report folder>/dataset):
  manifest_all.csv         every image with include/exclude + reason
  manifest_included.csv    included images + tree metadata + split
  paired_top_side.csv      (tree, date) pairs where BOTH views are included,
                           for multimodal fusion with SPAD, temperature, etc.
  exclusion_log.csv        excluded images only, with reasons

Splits are made BY TREE (all 18 dates of a tree in the same split), stratified
by variety x irrigation regime, so that the CNN is never evaluated on a tree
it has seen during training (avoids optimistic leakage).

Original images are not copied, moved or altered. Use --copy-to to copy the
included originals byte-for-byte (shutil.copy2) into a new folder.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Only unreadable files are excluded without a human decision. Blur and
# exposure flags are CANDIDATES (see README): confirm them with
# manual_decision=EXCLUDE, or opt in explicitly with --exclude-statuses.
DEFAULT_EXCLUDE = ["CORRUPTED"]


def decide(df: pd.DataFrame, exclude_statuses, strict: bool) -> pd.DataFrame:
    df = df.copy()
    md = df.get("manual_decision", pd.Series("", index=df.index)).fillna("").astype(str).str.upper().str.strip()
    notes = df.get("manual_notes", pd.Series("", index=df.index)).fillna("").astype(str)
    include, reason = [], []
    seen_groups = set()
    for i, r in df.iterrows():
        flags = set(str(r.get("quality_flags", "") or "").split(";")) - {""}
        if md[i] == "EXCLUDE":
            include.append(False); reason.append(f"manual: {notes[i]}".strip())
            continue
        if md[i] == "INCLUDE":
            include.append(True); reason.append("manual include")
            continue
        hit = [s for s in exclude_statuses if s in flags]
        if hit:
            include.append(False); reason.append("auto: " + ",".join(hit))
            continue
        g = r.get("exact_dup_group")
        if isinstance(g, str) and g:
            if g in seen_groups:
                include.append(False); reason.append(f"auto: exact duplicate of earlier file ({g})")
                continue
            seen_groups.add(g)
        if strict and r["final_status"] in ("REVIEW", "POSSIBLE_NEAR_DUPLICATE"):
            include.append(False); reason.append(f"strict: unreviewed {r['final_status']}")
            continue
        include.append(True); reason.append("passed QC" + (f" (flags: {';'.join(sorted(flags))})" if flags else ""))
    df["include"] = include
    df["decision_reason"] = reason
    return df


def tree_split(df: pd.DataFrame, fractions=(0.7, 0.15, 0.15), seed: int = 42) -> dict:
    """Assign each tree to train/val/test, stratified by variety x irrigation."""
    trees = df.dropna(subset=["tree_id"]).groupby("tree_id")[["variety", "irrigation"]].first().reset_index()
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in trees.groupby(["variety", "irrigation"], dropna=False):
        ids = g["tree_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(round(fractions[1] * n))) if n >= 3 else 0
        n_test = max(1, int(round(fractions[2] * n))) if n >= 3 else 0
        for k, t in enumerate(ids):
            out[t] = "test" if k < n_test else ("val" if k < n_test + n_val else "train")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", required=True, help="quality_report.csv (optionally with manual_decision filled)")
    p.add_argument("--out", default=None)
    p.add_argument("--exclude-statuses", nargs="*", default=DEFAULT_EXCLUDE)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-to", default=None, help="copy included originals (unchanged) here")
    a = p.parse_args()

    rep = Path(a.report)
    df = pd.read_csv(rep, parse_dates=["date"], encoding="utf-8-sig")
    out = Path(a.out) if a.out else rep.parent / "dataset"
    out.mkdir(parents=True, exist_ok=True)

    df = decide(df, a.exclude_statuses, a.strict)
    split = tree_split(df, seed=a.seed)
    df["split"] = df["tree_id"].map(split)

    keep_cols = ["filename", "path", "view", "tree_id", "date", "variety", "irrigation", "group",
                 "final_status", "quality_flags", "include", "decision_reason", "split",
                 "roi_x", "roi_y", "roi_w", "roi_h", "sha256"]
    keep_cols = [c for c in keep_cols if c in df]
    df[keep_cols].to_csv(out / "manifest_all.csv", index=False, encoding="utf-8-sig")
    inc = df[df["include"]]
    inc[keep_cols].to_csv(out / "manifest_included.csv", index=False, encoding="utf-8-sig")
    df[~df["include"]][keep_cols].to_csv(out / "exclusion_log.csv", index=False, encoding="utf-8-sig")

    if inc["tree_id"].notna().any():
        inc = inc.assign(day=inc["date"].dt.date)
        top = inc[inc["view"] == "top"][["tree_id", "day", "path", "date", "split"]]
        side = inc[inc["view"] == "side"][["tree_id", "day", "path"]]
        paired = top.merge(side, on=["tree_id", "day"], suffixes=("_top", "_side"))
        paired.to_csv(out / "paired_top_side.csv", index=False, encoding="utf-8-sig")
        print(f"Paired (tree, date) with both views included: {len(paired)}")

    unreviewed = df[df["include"] & (df["final_status"] != "OK")
                    & (df.get("manual_decision", pd.Series("", index=df.index)).fillna("") == "")]
    if len(unreviewed):
        print(f"WARNING: {len(unreviewed)} included images carry quality flags but have no "
              f"manual_decision yet (see manifest_all.csv, decision_reason).")
    print(pd.crosstab(df["view"], df["include"].map({True: "included", False: "excluded"}), margins=True))
    print("\nIncluded images per split:")
    print(inc.groupby(["split", "view"]).size().unstack(fill_value=0))
    print(f"\nManifests written to {out}")

    if a.copy_to:
        dst = Path(a.copy_to)
        for _, r in inc.iterrows():
            d = dst / r["view"]
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r["path"], d / r["filename"])  # byte-identical copy
        print(f"Copied {len(inc)} original files to {dst}")


if __name__ == "__main__":
    main()
