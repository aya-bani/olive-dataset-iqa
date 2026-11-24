"""
Turn measurements into quality flags and a final status.

An image keeps EVERY detected issue in `quality_flags` (';'-separated) and
in one boolean column per flag. `final_status` is the single most severe
category, for convenient filtering. `info_notes` holds observations that are
NOT defects (e.g. bright soil background) but help interpretation.

Severity order for final_status:
  CORRUPTED > EXACT_DUPLICATE > BLURRY > OVEREXPOSED > UNDEREXPOSED
  > POSSIBLE_NEAR_DUPLICATE > REVIEW > OK

Flags that map to REVIEW (need a human look, not a defect by themselves):
  BLUR_BORDERLINE, BLUR_OUTLIER, LOW_CONTRAST, ABNORMAL_DIMENSIONS,
  ORIENTATION_MISMATCH, METADATA_MISSING, VIEW_NAME_MISMATCH

Info notes (never change the status): BRIGHT_BACKGROUND, ROI_LOW_CONFIDENCE,
  ROI_FALLBACK_CENTRAL_BOX, BLUR_NOT_ON_LEAVES
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import Config

STATUS_ORDER = [
    "CORRUPTED", "EXACT_DUPLICATE", "BLURRY", "OVEREXPOSED", "UNDEREXPOSED",
    "POSSIBLE_NEAR_DUPLICATE", "REVIEW", "OK",
]
DIRECT_STATUS_FLAGS = set(STATUS_ORDER[:-2])
REVIEW_FLAGS = [
    "BLUR_BORDERLINE", "BLUR_OUTLIER", "LOW_CONTRAST", "ABNORMAL_DIMENSIONS",
    "ORIENTATION_MISMATCH", "METADATA_MISSING", "VIEW_NAME_MISMATCH",
]
ALL_FLAGS = STATUS_ORDER[:-2] + REVIEW_FLAGS


def _robust_z(x: pd.Series) -> pd.Series:
    lx = np.log(x.clip(lower=1e-6))
    med = lx.median()
    mad = (lx - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=x.index)
    return (lx - med) / mad


def assign_flags(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    flags: Dict[int, List[str]] = {i: [] for i in df.index}
    notes: Dict[int, List[str]] = {i: [] for i in df.index}

    def add(mask, flag, target=flags):
        if not isinstance(mask, pd.Series):
            mask = pd.Series(np.asarray(mask), index=df.index)
        for i in df.index[mask.reindex(df.index).fillna(False).to_numpy(dtype=bool)]:
            if flag not in target[i]:
                target[i].append(flag)

    readable = df["decode_ok"].fillna(False).astype(bool)

    # ---------------- Corruption
    add(df["load_errors"].fillna("") != "", "CORRUPTED")

    # ---------------- Exact duplicates
    add(df["exact_dup_group"].fillna("") != "", "EXACT_DUPLICATE")

    # ---------------- Near duplicates
    add(df["near_dup_pairs"].fillna("") != "", "POSSIBLE_NEAR_DUPLICATE")

    # ---------------- Per-view metrics
    df["blur_robust_z"] = np.nan
    for view, sub in df[readable].groupby("view"):
        t = cfg.for_view(view)
        idx = sub.index
        s = sub["blur_score"]
        df.loc[idx, "blur_robust_z"] = _robust_z(s)
        add(df.index.isin(idx) & (df["blur_score"] < t.blur_threshold), "BLURRY")
        add(df.index.isin(idx) & (df["blur_score"] >= t.blur_threshold)
            & (df["blur_score"] < t.blur_review_threshold), "BLUR_BORDERLINE")
        add(df.index.isin(idx) & (df["blur_robust_z"] < -t.blur_robust_z)
            & (df["blur_score"] >= t.blur_threshold), "BLUR_OUTLIER")

        over = (sub["roi_clip_any_pct"] > t.roi_clip_pct_over) | (sub["roi_mean"] > t.roi_mean_over)
        under = (sub["roi_mean"] < t.roi_mean_under) | (sub["roi_dark_pct"] > t.roi_dark_pct_under)
        add(df.index.isin(over[over].index), "OVEREXPOSED")
        add(df.index.isin(under[under].index), "UNDEREXPOSED")
        add(df.index.isin(sub.index[sub["roi_dynamic_range"] < t.roi_min_dynamic_range]), "LOW_CONTRAST")

        bright_bg = (sub["whole_white_pct"] > t.whole_white_pct_note) & ~over
        add(df.index.isin(bright_bg[bright_bg].index), "BRIGHT_BACKGROUND", notes)

        # Geometry relative to the view's typical (mode) image
        if len(sub):
            mode_w = sub["width"].mode().iloc[0]
            mode_h = sub["height"].mode().iloc[0]
            dims_bad = ((sub["width"] != mode_w) | (sub["height"] != mode_h)) & \
                       ~((sub["width"] == mode_h) & (sub["height"] == mode_w))
            small = sub[["width", "height"]].min(axis=1) < cfg.min_side_px
            ar = sub[["width", "height"]].max(axis=1) / sub[["width", "height"]].min(axis=1)
            ar_mode = max(mode_w, mode_h) / min(mode_w, mode_h)
            ar_bad = (ar / ar_mode - 1).abs() > cfg.aspect_tolerance
            add(df.index.isin(sub.index[dims_bad | small | ar_bad]), "ABNORMAL_DIMENSIONS")
            portrait = sub["height"] > sub["width"]
            majority_portrait = portrait.mean() >= 0.5
            add(df.index.isin(sub.index[portrait != majority_portrait]), "ORIENTATION_MISMATCH")

    # ---------------- Metadata / naming consistency
    add(df["tree_id"].isna() | (df["tree_id"].astype(str) == ""), "METADATA_MISSING")
    add(df["view_name_mismatch"].fillna(False).astype(bool), "VIEW_NAME_MISMATCH")

    # ---------------- ROI reliability (note, not a defect of the photo)
    add(df["roi_confidence"] == "low", "ROI_LOW_CONFIDENCE", notes)
    add(df["roi_method"] == "fallback", "ROI_FALLBACK_CENTRAL_BOX", notes)
    add(df["blur_score_source"].isin(["roi", "whole"]), "BLUR_NOT_ON_LEAVES", notes)

    # ---------------- Compose outputs
    df["quality_flags"] = [";".join(flags[i]) for i in df.index]
    df["info_notes"] = [";".join(notes[i]) for i in df.index]
    df["n_flags"] = [len(flags[i]) for i in df.index]
    for f in ALL_FLAGS:
        df[f"flag_{f}"] = [f in flags[i] for i in df.index]

    def status(fl: List[str]) -> str:
        for s in STATUS_ORDER[:-2]:
            if s in fl:
                return s
        return "REVIEW" if fl else "OK"

    df["final_status"] = [status(flags[i]) for i in df.index]
    return df
