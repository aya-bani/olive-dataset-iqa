"""
Plots and contact sheets. Everything is written to the output folder;
original images are only read (as small thumbnails).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config  # noqa: E402
from image_loader import load_thumbnail  # noqa: E402
from quality_flags import ALL_FLAGS, STATUS_ORDER  # noqa: E402

log = logging.getLogger(__name__)
VIEW_COLORS = {"top": "#2a6f97", "side": "#c8553d"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- plots
def plot_blur_distributions(df: pd.DataFrame, cfg: Config, out: Path):
    ok = df[df["decode_ok"]]
    cols = [("blur_lapvar_whole", "Whole image"), ("blur_lapvar_roi", "Tree ROI box"),
            ("blur_lapvar_leaf", "Leaf edges (primary)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")
    for r, view in enumerate(sorted(ok["view"].unique())):
        sub = ok[ok["view"] == view]
        t = cfg.for_view(view)
        for c, (col, title) in enumerate(cols):
            ax = axes[r, c]
            v = sub[col].dropna()
            v = v[v > 0]
            if len(v):
                bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 50)
                ax.hist(v, bins=bins, color=VIEW_COLORS.get(view, "grey"), alpha=0.85)
                ax.set_xscale("log")
            if col == "blur_lapvar_" + cfg.blur_score_source or (c == 2 and cfg.blur_score_source == "leaf"):
                ax.axvline(t.blur_threshold, color="red", ls="--", label=f"BLURRY < {t.blur_threshold:g}")
                ax.axvline(t.blur_review_threshold, color="orange", ls=":",
                           label=f"REVIEW < {t.blur_review_threshold:g}")
                ax.legend(fontsize=8)
            ax.set_title(f"{view} view - {title} (n={len(v)})")
            ax.set_xlabel("Variance of Laplacian (log scale)")
            ax.set_ylabel("Images")
    _save(fig, out / "blur_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, view in zip(axes, sorted(ok["view"].unique())):
        sub = ok[ok["view"] == view]
        ax.scatter(sub["blur_lapvar_whole"], sub["blur_score"], s=8, alpha=0.5,
                   c=np.where(sub["flag_BLURRY"], "red", VIEW_COLORS.get(view, "grey")))
        ax.set_xscale("log")
        ax.set_yscale("log")
        lim = [max(1, min(sub["blur_lapvar_whole"].min(), sub["blur_score"].min())),
               max(sub["blur_lapvar_whole"].max(), sub["blur_score"].max())]
        ax.plot(lim, lim, "k:", lw=1)
        ax.set_xlabel("Whole-image Laplacian variance")
        ax.set_ylabel("Primary (leaf/ROI) Laplacian variance")
        ax.set_title(f"{view}: whole image vs tree region (red = BLURRY)")
    _save(fig, out / "blur_whole_vs_tree.png")


def plot_brightness(df: pd.DataFrame, cfg: Config, out: Path):
    ok = df[df["decode_ok"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for r, view in enumerate(sorted(ok["view"].unique())):
        sub = ok[ok["view"] == view]
        t = cfg.for_view(view)
        col = VIEW_COLORS.get(view, "grey")
        ax = axes[r, 0]
        ax.hist(sub["whole_mean"], bins=50, alpha=0.6, color="grey", label="whole image")
        ax.hist(sub["roi_mean"], bins=50, alpha=0.7, color=col, label="tree ROI")
        ax.axvline(t.roi_mean_over, color="red", ls="--", label="ROI over")
        ax.axvline(t.roi_mean_under, color="purple", ls="--", label="ROI under")
        ax.set_title(f"{view}: mean grayscale brightness")
        ax.legend(fontsize=8)
        ax = axes[r, 1]
        ax.scatter(sub["whole_white_pct"], sub["roi_clip_any_pct"], s=8, alpha=0.5, color=col)
        ax.axhline(t.roi_clip_pct_over, color="red", ls="--")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlabel("Whole-image nearly-white %")
        ax.set_ylabel("ROI clipped-channel %")
        ax.set_title(f"{view}: bright background vs clipped tree")
        ax = axes[r, 2]
        ax.hist(sub["roi_dynamic_range"], bins=50, color=col)
        ax.axvline(t.roi_min_dynamic_range, color="red", ls="--", label="LOW_CONTRAST")
        ax.set_title(f"{view}: ROI dynamic range (p95-p5)")
        ax.legend(fontsize=8)
    _save(fig, out / "brightness_distributions.png")


def plot_flag_counts(df: pd.DataFrame, out: Path):
    views = sorted(df["view"].unique())
    flag_cols = [f for f in ALL_FLAGS if df.get(f"flag_{f}", pd.Series(dtype=bool)).any()]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    x = np.arange(len(flag_cols))
    wbar = 0.8 / max(1, len(views))
    for k, v in enumerate(views):
        sub = df[df["view"] == v]
        counts = [int(sub[f"flag_{f}"].sum()) for f in flag_cols]
        bars = axes[0].bar(x + k * wbar, counts, wbar, label=v, color=VIEW_COLORS.get(v, "grey"))
        axes[0].bar_label(bars, fontsize=7)
    axes[0].set_xticks(x + wbar * (len(views) - 1) / 2)
    axes[0].set_xticklabels(flag_cols, rotation=35, ha="right", fontsize=8)
    axes[0].set_title("Images per flag (an image can carry several flags)")
    axes[0].legend()

    st = [s for s in STATUS_ORDER if (df["final_status"] == s).any()]
    x = np.arange(len(st))
    for k, v in enumerate(views):
        sub = df[df["view"] == v]
        counts = [int((sub["final_status"] == s).sum()) for s in st]
        bars = axes[1].bar(x + k * wbar, counts, wbar, label=v, color=VIEW_COLORS.get(v, "grey"))
        axes[1].bar_label(bars, fontsize=7)
    axes[1].set_xticks(x + wbar * (len(views) - 1) / 2)
    axes[1].set_xticklabels(st, rotation=35, ha="right", fontsize=8)
    axes[1].set_title("Final status (most severe flag)")
    axes[1].legend()
    _save(fig, out / "flag_counts.png")


# ------------------------------------------------------------ contact sheets
def _tile(row: pd.Series, cfg: Config, extra: str = "") -> np.ndarray:
    size = cfg.thumb_size
    th = load_thumbnail(Path(row["path"]), max_side=size)
    canvas = np.full((size + 58, size, 3), 255, np.uint8)
    if th is not None:
        # draw ROI box (stored in full-resolution coordinates)
        if not pd.isna(row.get("roi_x")):
            full_h, full_w = row["height"], row["width"]
            sx, sy = th.shape[1] / full_w, th.shape[0] / full_h
            x, y = int(row["roi_x"] * sx), int(row["roi_y"] * sy)
            w, h = int(row["roi_w"] * sx), int(row["roi_h"] * sy)
            cv2.rectangle(th, (x, y), (x + w, y + h), (0, 255, 255), 2)
        oy, ox = (size - th.shape[0]) // 2, (size - th.shape[1]) // 2
        canvas[oy:oy + th.shape[0], ox:ox + th.shape[1]] = th
    else:
        cv2.putText(canvas, "UNREADABLE", (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def txt(s, y, color=(0, 0, 0)):
        cv2.putText(canvas, s[:42], (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    tree = row.get("tree_id") or "?"
    date = row["date"].strftime("%Y-%m-%d") if not pd.isna(row.get("date")) else "?"
    txt(f"{row['filename']}  {tree} {date}", size + 12)
    txt(f"blur {row.get('blur_score', np.nan):.0f} ({row.get('blur_score_source', '')})"
        f"  whole {row.get('blur_lapvar_whole', np.nan):.0f}", size + 26)
    txt(f"ROI mean {row.get('roi_mean', np.nan):.0f} clip {row.get('roi_clip_any_pct', np.nan):.1f}%"
        f" DR {row.get('roi_dynamic_range', np.nan):.0f}", size + 40)
    txt(extra or str(row.get("quality_flags", "")), size + 54, (0, 0, 200))
    return canvas


def contact_sheet(rows: pd.DataFrame, cfg: Config, path: Path, title: str):
    if rows.empty:
        return
    rows = rows.head(cfg.contact_sheet_max_images)
    tiles = [_tile(r, cfg) for _, r in rows.iterrows()]
    cols = cfg.contact_sheet_cols
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    header = np.full((34, grid.shape[1], 3), 235, np.uint8)
    cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([header, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def flagged_contact_sheets(df: pd.DataFrame, cfg: Config, out: Path) -> List[Path]:
    written = []
    sort_keys = {
        "BLURRY": ("blur_score", True), "BLUR_BORDERLINE": ("blur_score", True),
        "BLUR_OUTLIER": ("blur_score", True), "OVEREXPOSED": ("roi_clip_any_pct", False),
        "UNDEREXPOSED": ("roi_mean", True), "LOW_CONTRAST": ("roi_dynamic_range", True),
    }
    for view in sorted(df["view"].unique()):
        sub = df[df["view"] == view]
        for f in ALL_FLAGS:
            col = f"flag_{f}"
            if col not in sub or not sub[col].any():
                continue
            rows = sub[sub[col]]
            key, asc = sort_keys.get(f, ("filename", True))
            rows = rows.sort_values(key, ascending=asc)
            p = out / view / f"{f}.jpg"
            contact_sheet(rows, cfg, p, f"{view} view - {f} ({len(rows)} images"
                          f"{', first ' + str(cfg.contact_sheet_max_images) if len(rows) > cfg.contact_sheet_max_images else ''})")
            written.append(p)
        # reference sheets: sharpest / median images for calibration
        ok = sub[sub["final_status"] == "OK"].sort_values("blur_score")
        if len(ok):
            mid = len(ok) // 2
            contact_sheet(ok.iloc[max(0, mid - 6):mid + 6], cfg, out / view / "_reference_OK_median_blur.jpg",
                          f"{view} view - OK images around the median blur score (reference)")
            contact_sheet(ok.head(18), cfg, out / view / "_reference_OK_lowest_blur.jpg",
                          f"{view} view - lowest-scoring images still OK (check threshold)")
    return written


def roi_preview_sheet(df: pd.DataFrame, cfg: Config, out: Path, n: int = 24):
    for view in sorted(df["view"].unique()):
        sub = df[(df["view"] == view) & df["decode_ok"]]
        if sub.empty:
            continue
        sample = sub.sample(min(n, len(sub)), random_state=cfg.random_seed).sort_values("filename")
        contact_sheet(sample, cfg, out / f"roi_preview_{view}.jpg",
                      f"{view} view - random sample with detected tree ROI (yellow box)")


def duplicate_pair_sheet(pairs: pd.DataFrame, df: pd.DataFrame, cfg: Config, path: Path,
                         max_pairs: int = 40):
    if pairs is None or pairs.empty:
        return
    by_path = df.set_index("path", drop=False)
    rows = []
    for _, p in pairs.head(max_pairs).iterrows():
        ta = _tile(by_path.loc[p["path_a"]], cfg, extra=f"{p['pair_id']} d={p['phash_distance']}")
        tb = _tile(by_path.loc[p["path_b"]], cfg, extra=f"{p['relation']}")
        sep = np.full((ta.shape[0], 12, 3), 80, np.uint8)
        rows.append(np.hstack([ta, sep, tb]))
    pair_rows = []
    for i in range(0, len(rows), 2):
        chunk = rows[i:i + 2]
        if len(chunk) == 1:
            chunk.append(np.full_like(chunk[0], 255))
        pair_rows.append(np.hstack([chunk[0], np.full((chunk[0].shape[0], 30, 3), 255, np.uint8), chunk[1]]))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(pair_rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
