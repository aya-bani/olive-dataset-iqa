"""
Olive-tree image quality assessment pipeline - entry point.

Examples (from this folder, with the virtual environment active):
    python main.py --sample 40                 # quick test on 40 images/view
    python main.py                             # full dataset
    python main.py --views side                # one perspective only
    python main.py --no-roi                    # whole-image metrics only
    python main.py --output outputs/run_x      # custom output folder
    python main.py --from-report outputs/full_run_v1/quality_report.csv --output outputs/full_run_v2
                                               # re-apply edited thresholds in seconds

The pipeline NEVER modifies, moves or deletes original images. It only reads
them and writes reports into the output folder.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

import config as config_module
from blur_detection import blur_scores, primary_blur_score
from config import Config
from duplicate_detection import (exact_duplicate_groups, find_near_duplicates,
                                 near_duplicate_ids, perceptual_hashes)
from exposure_analysis import exposure_stats
from image_loader import discover_images, load_image
from metadata import load_metadata
from quality_flags import assign_flags
from reporting import print_summary, write_reports
from roi_extraction import box_to_full_res, crop, extract_roi, load_manual_boxes

log = logging.getLogger("iqa")


# --------------------------------------------------------------------------- #
def analyze_one(args) -> dict:
    """Measure one image. Runs in a worker process; must not raise."""
    path_str, view, cfg, manual_box = args
    # One OpenCV thread per worker process: parallelism comes from the
    # process pool; nested threading would oversubscribe the CPU.
    cv2.setNumThreads(1)
    path = Path(path_str)
    row = {"filename": path.name, "path": str(path), "view": view}
    try:
        res = load_image(path, cfg)
        row.update({
            "sha256": res.sha256, "file_size_bytes": res.file_size_bytes, "format": res.format,
            "width": res.width, "height": res.height, "exif_orientation": res.exif_orientation,
            "exif_datetime": res.exif_datetime, "camera_model": res.camera_model,
            "load_errors": ";".join(res.errors), "decode_ok": res.image_bgr is not None,
        })
        img = res.image_bgr
        if img is None:
            return row

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # ---- ROI
        if cfg.use_roi:
            roi = extract_roi(img, cfg, manual_box)
            box, mask = roi.box, roi.veg_mask
            row.update({"roi_method": roi.method, "roi_confidence": roi.confidence,
                        "veg_pct_image": roi.veg_pct_image, "roi_note": roi.note})
        else:
            box, mask = (0, 0, W, H), None
            row.update({"roi_method": "whole_image", "roi_confidence": "n/a", "veg_pct_image": np.nan})
        fx, fy, fw, fh = box_to_full_res(box, cfg.reduce_factor)
        row.update({"roi_x": fx, "roi_y": fy, "roi_w": fw, "roi_h": fh,
                    "roi_area_pct": 100.0 * box[2] * box[3] / float(W * H)})

        # ---- Blur
        bs = blur_scores(gray, box if cfg.use_roi else None, mask,
                         cfg.veg_edge_dilate_px, cfg.min_veg_frac_for_leaf_blur)
        row.update(bs)
        score, src = primary_blur_score(bs, cfg.blur_score_source if cfg.use_roi else "whole")
        row["blur_score"], row["blur_score_source"] = score, src

        # ---- Exposure
        kw = dict(white_level=cfg.white_level, dark_level=cfg.dark_level, clip_level=cfg.clip_level)
        row.update(exposure_stats(img, "whole", full=True, **kw))
        row.update(exposure_stats(crop(img, box), "roi", full=False, **kw))

        # ---- Perceptual hashes
        row.update(perceptual_hashes(img, cfg.phash_size))

        # ---- Optional ROI crop (never touches the original file)
        if cfg.save_roi_crops and cfg.use_roi:
            out = Path(cfg.output_dir) / "roi_crops" / view
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{path.stem}_roi.jpg"), crop(img, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
    except Exception as e:  # noqa: BLE001 - record, never crash the whole run
        row["load_errors"] = (row.get("load_errors", "") + f";ANALYSIS_ERROR:{type(e).__name__}:{e}").strip(";")
        row["decode_ok"] = row.get("decode_ok", False)
    return row


# --------------------------------------------------------------------------- #
def run(cfg: Config, views=None, sample: Optional[int] = None, make_plots: bool = True) -> pd.DataFrame:
    t0 = time.time()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    records = discover_images(cfg, views)
    if sample:
        rng = random.Random(cfg.random_seed)
        by_view = {}
        for r in records:
            by_view.setdefault(r.view, []).append(r)
        records = []
        for v, rs in by_view.items():
            records.extend(sorted(rng.sample(rs, min(sample, len(rs))), key=lambda r: r.path.name))
    if not records:
        raise SystemExit("No images found - check config.image_dirs.")

    meta = load_metadata(cfg)
    manual = load_manual_boxes(cfg.manual_roi_csv)

    workers = cfg.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    jobs = [(str(r.path), r.view, cfg, manual.get(r.path.name)) for r in records]
    log.info("Analysing %d images with %d worker(s)...", len(jobs), workers)
    rows = []
    if workers == 1:
        for k, j in enumerate(jobs, 1):
            rows.append(analyze_one(j))
            _progress(k, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(analyze_one, jobs, chunksize=4), 1):
                rows.append(r)
                _progress(k, len(jobs))
    df = pd.DataFrame(rows)
    return finalize(df, cfg, meta, out, t0, sample, len(manual), make_plots)


FLAG_DERIVED_PREFIXES = ("flag_",)
FLAG_DERIVED_COLS = {"quality_flags", "info_notes", "n_flags", "final_status", "blur_robust_z",
                     "exact_dup_group", "near_dup_pairs"}


def reflag_from_report(cfg: Config, report: str, make_plots: bool = True) -> pd.DataFrame:
    """Re-apply (new) thresholds to stored measurements - no image re-processing.

    Manual review columns (manual_decision, manual_notes) are preserved.
    Note: settings that change the MEASUREMENTS (reduce_factor, ROI
    parameters, white/dark/clip levels) require a full run instead.
    """
    t0 = time.time()
    df = pd.read_csv(report, encoding="utf-8-sig", keep_default_na=True)
    drop = [c for c in df.columns if c.startswith(FLAG_DERIVED_PREFIXES) or c in FLAG_DERIVED_COLS]
    df = df.drop(columns=drop)
    for c in ("manual_decision", "manual_notes", "load_errors", "sha256", "phash", "dhash",
              "roi_method", "roi_confidence", "blur_score_source"):
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return finalize(df, cfg, None, out, t0, "from_report", 0, make_plots, source_report=report)


def finalize(df: pd.DataFrame, cfg: Config, meta, out: Path, t0: float, sample, n_manual: int,
             make_plots: bool, source_report: Optional[str] = None) -> pd.DataFrame:
    if meta is not None:
        _join_metadata(df, cfg, meta)
    df["date"] = pd.to_datetime(df["date"])
    if "decode_ok" not in df:
        df["decode_ok"] = False
    df["decode_ok"] = df["decode_ok"].fillna(False).astype(bool)
    for c in ("load_errors", "sha256", "phash", "dhash", "roi_method", "roi_confidence", "blur_score_source"):
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("")
    df["tree_id"] = df["tree_id"].where(df["tree_id"].notna(), None)

    # ---- Duplicates
    df["exact_dup_group"] = exact_duplicate_groups(df)
    pairs = find_near_duplicates(df, cfg.near_dup_strict, cfg.near_dup_loose,
                                 cfg.compare_across_views, cfg.same_session_max_days)
    nd = near_duplicate_ids(df, pairs)
    df["near_dup_pairs"] = df["path"].map(lambda p: nd.get(p, ""))

    # ---- Flags
    df = assign_flags(df, cfg)

    run_info = {
        "run_started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0, 1),
        "n_images": len(df),
        "sample_per_view": sample or "all",
        "source_report": source_report or "",
        "views": ",".join(sorted(df["view"].unique())),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_manual_roi_boxes": n_manual,
    }
    try:
        import imagehash
        run_info["imagehash"] = getattr(imagehash, "__version__", "?")
    except ImportError:
        pass

    paths = write_reports(df, pairs, cfg, out, run_info)

    if make_plots:
        import visualization as viz
        viz.plot_blur_distributions(df, cfg, out / "plots")
        viz.plot_brightness(df, cfg, out / "plots")
        viz.plot_flag_counts(df, out / "plots")
        viz.flagged_contact_sheets(df, cfg, out / "contact_sheets")
        viz.roi_preview_sheet(df, cfg, out / "contact_sheets")
        viz.duplicate_pair_sheet(pairs[pairs["flagged"]] if len(pairs) else pairs, df, cfg,
                                 out / "contact_sheets" / "duplicate_pairs_flagged.jpg")
        viz.duplicate_pair_sheet(pairs[~pairs["flagged"]] if len(pairs) else pairs, df, cfg,
                                 out / "contact_sheets" / "duplicate_pairs_expected_similarity.jpg")

    print_summary(df)
    log.info("Done in %.1f s. Reports: %s", time.time() - t0, ", ".join(str(p) for p in paths.values()))
    return df


def _join_metadata(df: pd.DataFrame, cfg: Config, meta) -> None:
    """Join workbook metadata by view + file stem, as recorded in the workbook."""
    df["stem"] = df["filename"].map(lambda s: Path(s).stem)
    for key in ("tree_id", "date", "variety", "irrigation", "group"):
        df[key] = [(meta.get(v, {}).get(s) or {}).get(key) for v, s in zip(df["view"], df["stem"])]
    df["view_name_mismatch"] = [
        not s.startswith(cfg.view_filename_prefix.get(v, "")) or
        any(s.startswith(p) and len(p) > len(cfg.view_filename_prefix.get(v, ""))
            for ov, p in cfg.view_filename_prefix.items() if ov != v)
        for v, s in zip(df["view"], df["stem"])
    ]


def _progress(k: int, n: int):
    if k == n or k % max(1, n // 20) == 0:
        log.info("  %d / %d", k, n)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Olive-tree image quality assessment")
    p.add_argument("--sample", type=int, default=None, help="random images per view (reproducible)")
    p.add_argument("--views", nargs="+", choices=["top", "side"], default=None)
    p.add_argument("--output", default=None, help="output folder (default: outputs/)")
    p.add_argument("--no-roi", action="store_true", help="disable tree ROI extraction")
    p.add_argument("--save-roi-crops", action="store_true", help="write ROI crops as JPEG files")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--from-report", default=None,
                   help="re-apply thresholds from config.py to an existing quality_report.csv "
                        "(fast; no image re-processing; keeps manual_decision/manual_notes)")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    a = parse_args(argv)
    cfg = config_module.default_config()
    if a.output:
        cfg.output_dir = str(Path(a.output).resolve())
    if a.no_roi:
        cfg.use_roi = False
    if a.save_roi_crops:
        cfg.save_roi_crops = True
    if a.workers is not None:
        cfg.workers = a.workers
    if a.from_report:
        reflag_from_report(cfg, a.from_report, make_plots=not a.no_plots)
    else:
        run(cfg, views=a.views, sample=a.sample, make_plots=not a.no_plots)


if __name__ == "__main__":  # required for multiprocessing on Windows
    main()
