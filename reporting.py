"""
CSV / Excel reports and summary statistics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from quality_flags import ALL_FLAGS, STATUS_ORDER

log = logging.getLogger(__name__)

# Column order of the main report (remaining columns are appended after).
MAIN_COLUMNS = [
    "filename", "path", "view", "tree_id", "date", "variety", "irrigation", "group",
    "width", "height", "file_size_bytes", "format", "exif_orientation", "exif_datetime",
    "final_status", "quality_flags", "info_notes", "n_flags",
    "blur_score", "blur_score_source", "blur_robust_z",
    "blur_lapvar_whole", "blur_lapvar_roi", "blur_lapvar_leaf",
    "blur_tenengrad_whole", "blur_tenengrad_roi", "leaf_pixel_frac_roi",
    "whole_mean", "whole_std", "whole_white_pct", "whole_dark_pct", "whole_clip_any_pct",
    "roi_mean", "roi_std", "roi_white_pct", "roi_dark_pct", "roi_clip_any_pct", "roi_dynamic_range",
    "roi_method", "roi_confidence", "roi_x", "roi_y", "roi_w", "roi_h", "roi_area_pct", "veg_pct_image",
    "sha256", "exact_dup_group", "phash", "dhash", "near_dup_pairs", "load_errors",
    "manual_decision", "manual_notes",
]

SUMMARY_METRICS = [
    "blur_score", "blur_lapvar_whole", "blur_lapvar_roi", "blur_lapvar_leaf",
    "whole_mean", "roi_mean", "whole_white_pct", "roi_white_pct", "whole_dark_pct",
    "roi_dark_pct", "roi_clip_any_pct", "roi_dynamic_range", "roi_area_pct",
]


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("manual_decision", "manual_notes"):
        if c not in df:
            df[c] = ""
    first = [c for c in MAIN_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def summary_tables(df: pd.DataFrame) -> dict:
    tables = {}
    ok = df[df["decode_ok"]]
    stats = ok.groupby("view")[SUMMARY_METRICS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95])
    stats = stats.stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "metric"})
    tables["metric_stats_by_view"] = stats

    rows = []
    for view, sub in list(df.groupby("view")) + [("ALL", df)]:
        r = {"view": view, "n_images": len(sub)}
        for s in STATUS_ORDER:
            r[f"status_{s}"] = int((sub["final_status"] == s).sum())
        for f in ALL_FLAGS:
            r[f"flag_{f}"] = int(sub.get(f"flag_{f}", pd.Series(False, index=sub.index)).sum())
        r["note_BRIGHT_BACKGROUND"] = int(sub["info_notes"].str.contains("BRIGHT_BACKGROUND").sum())
        r["note_ROI_LOW_CONFIDENCE"] = int(sub["info_notes"].str.contains("ROI_LOW_CONFIDENCE").sum())
        r["roi_method_vegetation"] = int((sub["roi_method"] == "vegetation").sum())
        r["roi_method_manual"] = int((sub["roi_method"] == "manual").sum())
        r["roi_method_fallback"] = int((sub["roi_method"] == "fallback").sum())
        rows.append(r)
    counts = pd.DataFrame(rows).set_index("view").T.reset_index().rename(columns={"index": "count"})
    tables["counts_by_view"] = counts

    if "tree_id" in df and df["tree_id"].notna().any():
        per_tree = (df.assign(flagged=df["final_status"] != "OK")
                    .groupby(["view", "tree_id"])["flagged"].agg(["sum", "count"]).reset_index()
                    .rename(columns={"sum": "n_not_ok", "count": "n_images"}))
        tables["not_ok_per_tree"] = per_tree.sort_values("n_not_ok", ascending=False)
        if df["date"].notna().any():
            per_date = (df.assign(flagged=df["final_status"] != "OK", day=pd.to_datetime(df["date"]).dt.date)
                        .groupby(["view", "day"])["flagged"].agg(["sum", "count"]).reset_index()
                        .rename(columns={"sum": "n_not_ok", "count": "n_images"}))
            tables["not_ok_per_day"] = per_date
    return tables


def write_reports(df: pd.DataFrame, pairs: pd.DataFrame, cfg: Config, out: Path,
                  run_info: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = order_columns(df)
    paths = {}

    csv_path = out / "quality_report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    paths["csv"] = csv_path

    pairs_path = out / "duplicate_pairs.csv"
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    paths["pairs"] = pairs_path

    tables = summary_tables(df)
    for name, t in tables.items():
        t.to_csv(out / f"summary_{name}.csv", index=False, encoding="utf-8-sig")

    review = df[df["final_status"] != "OK"].sort_values(["view", "final_status", "filename"])
    review_path = out / "manual_review_list.csv"
    review[["filename", "path", "view", "tree_id", "date", "final_status", "quality_flags",
            "info_notes", "blur_score", "roi_mean", "roi_clip_any_pct", "roi_dynamic_range",
            "near_dup_pairs", "manual_decision", "manual_notes"]] \
        .to_csv(review_path, index=False, encoding="utf-8-sig")
    paths["review"] = review_path

    xlsx_path = out / "quality_report.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            _excel_safe(df).to_excel(xw, sheet_name="report", index=False)
            _excel_safe(review).to_excel(xw, sheet_name="needs_review", index=False)
            _excel_safe(pairs).to_excel(xw, sheet_name="duplicate_pairs", index=False)
            for name, t in tables.items():
                _excel_safe(t).to_excel(xw, sheet_name=name[:31], index=False)
            pd.DataFrame(_flatten(cfg.to_dict()).items(), columns=["setting", "value"]) \
              .to_excel(xw, sheet_name="config", index=False)
            pd.DataFrame(run_info.items(), columns=["item", "value"]).to_excel(xw, sheet_name="run_info", index=False)
            for ws in xw.book.worksheets:
                ws.freeze_panes = "B2"
        paths["xlsx"] = xlsx_path
    except Exception as e:  # noqa: BLE001 - CSV outputs remain available
        log.warning("Excel report could not be written (%s). CSV files are complete.", e)

    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "run_info": run_info}, f, indent=2, default=str)
    return paths


def _excel_safe(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: v if not isinstance(v, (list, dict, tuple)) else str(v))
    return d


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, default=str) if isinstance(v, (list, tuple)) else v
    return out


def print_summary(df: pd.DataFrame):
    print("\n=== Final status by view ===")
    print(pd.crosstab(df["final_status"], df["view"], margins=True).to_string())
    print("\n=== Images per flag (non-exclusive) ===")
    rows = {f: df.groupby("view")[f"flag_{f}"].sum() for f in ALL_FLAGS if df[f"flag_{f}"].any()}
    if rows:
        print(pd.DataFrame(rows).T.astype(int).to_string())
    ok = df[df["decode_ok"]]
    print("\n=== Median metrics by view ===")
    print(ok.groupby("view")[["blur_score", "blur_lapvar_whole", "roi_mean", "whole_mean",
                              "roi_clip_any_pct", "whole_white_pct"]].median().round(2).to_string())
    print()
    np.set_printoptions()
