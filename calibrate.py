"""
Threshold calibration against manually inspected images.

Step 1 - draw a labelling sample (stratified over the score range, per view):
    python calibrate.py sample --report outputs/full_run_v1/quality_report.csv --metric blur_score --n 40

  -> writes calibration/labels_blur_score.csv (fill the `label` column with
     1 = defective (e.g. blurry) or 0 = acceptable) and contact sheets
     calibration/labels_blur_score_<view>.jpg showing the same images in order.
     Always open the ORIGINAL file at 100 % zoom before labelling.

Step 2 - evaluate thresholds on your labels:
    python calibrate.py fit --labels calibration/labels_blur_score.csv --metric blur_score --direction below

  -> for each view prints sensitivity / specificity for candidate thresholds
     and the threshold maximising Youden's J and the one reaching the target
     sensitivity (default 0.95: miss at most 5 % of truly defective images).
     Copy the chosen values into config.py (thresholds[view]).

`--direction below`: low values are bad (blur_score, roi_mean for under-exposure,
roi_dynamic_range). `--direction above`: high values are bad (roi_clip_any_pct,
roi_mean for over-exposure).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import default_config


def cmd_sample(a):
    df = pd.read_csv(a.report, parse_dates=["date"])
    df = df[df["decode_ok"].astype(bool) & df[a.metric].notna()]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(a.seed)
    for view, sub in df.groupby("view"):
        sub = sub.sort_values(a.metric)
        # Over-sample the low (or high) tail where the decision boundary lies:
        # half of the images from the worst 15 %, half spread over the rest.
        k = len(sub)
        tail_n = a.n // 2
        if a.direction == "below":
            tail, rest = sub.iloc[: int(0.15 * k)], sub.iloc[int(0.15 * k):]
        else:
            tail, rest = sub.iloc[int(0.85 * k):], sub.iloc[: int(0.85 * k)]
        pick_tail = tail.iloc[np.linspace(0, len(tail) - 1, min(tail_n, len(tail))).astype(int)]
        q = rest.iloc[np.sort(rng.choice(len(rest), min(a.n - len(pick_tail), len(rest)), replace=False))]
        s = pd.concat([pick_tail, q]).drop_duplicates("path").sort_values(a.metric)
        rows.append(s)
    lab = pd.concat(rows)[["filename", "path", "view", "tree_id", "date", a.metric, "quality_flags"]]
    lab["label"] = ""
    lab["labeller_notes"] = ""
    p = out_dir / f"labels_{a.metric}.csv"
    lab.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"Label template: {p}  ({len(lab)} images). Fill `label` with 1 (defective) / 0 (acceptable).")

    try:
        import visualization as viz
        cfg = default_config()
        full = pd.read_csv(a.report, parse_dates=["date"]).set_index("path")
        for view, sub in lab.groupby("view"):
            rows = full.loc[sub["path"]].reset_index()
            rows["quality_flags"] = [f"#{i + 1}" for i in range(len(rows))]
            cfg.contact_sheet_max_images = len(rows)
            viz.contact_sheet(rows, cfg, out_dir / f"labels_{a.metric}_{view}.jpg",
                              f"Calibration sample - {view} - sorted by {a.metric}")
    except Exception as e:  # noqa: BLE001
        print("Contact sheet not written:", e)


def _metrics(y, bad_pred):
    tp = int(((y == 1) & bad_pred).sum())
    fn = int(((y == 1) & ~bad_pred).sum())
    tn = int(((y == 0) & ~bad_pred).sum())
    fp = int(((y == 0) & bad_pred).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return tp, fn, tn, fp, sens, spec


def cmd_fit(a):
    lab = pd.read_csv(a.labels)
    lab = lab[lab["label"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    lab["label"] = lab["label"].astype(float).astype(int)
    if lab.empty:
        raise SystemExit("No labelled rows (label must be 0 or 1).")
    for view, sub in lab.groupby("view"):
        x = sub[a.metric].to_numpy(float)
        y = sub["label"].to_numpy(int)
        print(f"\n=== {view}: {len(sub)} labelled images, {y.sum()} defective ===")
        if y.sum() == 0 or y.sum() == len(y):
            print("Need both defective and acceptable examples to calibrate.")
            continue
        cands = np.unique(np.concatenate([x, (x[:-1] + x[1:]) / 2 if len(x) > 1 else x]))
        res = []
        for t in cands:
            bad = x < t if a.direction == "below" else x > t
            tp, fn, tn, fp, se, sp = _metrics(y, bad)
            res.append((t, tp, fn, tn, fp, se, sp, se + sp - 1))
        r = pd.DataFrame(res, columns=["threshold", "TP", "FN", "TN", "FP", "sensitivity", "specificity", "youden_J"])
        best = r.loc[r["youden_J"].idxmax()]
        ok = r[r["sensitivity"] >= a.target_sensitivity]
        tgt = (ok.loc[ok["specificity"].idxmax()] if len(ok) else None)
        print(r.iloc[np.linspace(0, len(r) - 1, min(15, len(r))).astype(int)].round(3).to_string(index=False))
        print(f"\nMax Youden J : threshold {best.threshold:.4g}  (sens {best.sensitivity:.2f}, spec {best.specificity:.2f})")
        if tgt is not None:
            print(f"Sens >= {a.target_sensitivity:.2f}: threshold {tgt.threshold:.4g}  "
                  f"(sens {tgt.sensitivity:.2f}, spec {tgt.specificity:.2f})")
        print("Suggestion: hard flag at the Youden threshold, REVIEW band up to the high-sensitivity threshold.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sample")
    s.add_argument("--report", required=True)
    s.add_argument("--metric", default="blur_score")
    s.add_argument("--direction", choices=["below", "above"], default="below")
    s.add_argument("--n", type=int, default=40, help="images per view")
    s.add_argument("--out", default="calibration")
    s.add_argument("--seed", type=int, default=42)
    f = sp.add_parser("fit")
    f.add_argument("--labels", required=True)
    f.add_argument("--metric", default="blur_score")
    f.add_argument("--direction", choices=["below", "above"], default="below")
    f.add_argument("--target-sensitivity", type=float, default=0.95)
    a = p.parse_args()
    {"sample": cmd_sample, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
