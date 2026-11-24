"""
Central configuration for the olive-tree image quality assessment (IQA) pipeline.

Every threshold used to raise a flag lives here, so that a run can be
reproduced exactly. `main.py` writes a JSON snapshot of the effective
configuration into each output folder.

Thresholds are deliberately PER VIEW ("top" / "side"): the two perspectives
have different compositions, so their score distributions differ. The default
values below are STARTING POINTS chosen from the first full run on this
dataset; calibrate them with `calibrate.py` against manually labelled samples
(see README, section "Calibrating thresholds").
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = PROJECT_DIR.parent  # .../olive-research


@dataclass
class ViewThresholds:
    """Thresholds for one perspective. All values are configurable."""

    # ---- Blur (variance of Laplacian, computed at the analysis resolution) --
    # Defaults come from the first full run (reduce_factor=2, leaf score):
    # median ~28 000-32 000, 1st percentile ~1 400. Visual checks at 100 %
    # zoom showed that SHARP extreme close-ups of large leaves can score
    # < 300 too (little fine texture), so BLURRY is a candidate list that must
    # be confirmed by eye, and the REVIEW band is deliberately wide.
    # Image is BLURRY if its primary sharpness score is below this value.
    blur_threshold: float = 300.0
    # Image gets BLUR_BORDERLINE (-> REVIEW) between blur_threshold and this.
    blur_review_threshold: float = 1500.0
    # Additionally flag statistical outliers within the view: robust z-score of
    # log(blur score) below -blur_robust_z -> BLUR_OUTLIER (-> REVIEW).
    blur_robust_z: float = 3.0

    # ---- Exposure, assessed on the tree ROI (not the whole photo) ----------
    # OVEREXPOSED if > this % of ROI pixels have at least one channel clipped
    roi_clip_pct_over: float = 8.0
    # OVEREXPOSED if mean ROI grayscale brightness is above this
    roi_mean_over: float = 210.0
    # UNDEREXPOSED if mean ROI grayscale brightness is below this
    roi_mean_under: float = 45.0
    # UNDEREXPOSED if > this % of ROI pixels are very dark
    roi_dark_pct_under: float = 35.0
    # LOW_CONTRAST (haze, lens fog, flare) if ROI p95 - p5 grey range < this
    roi_min_dynamic_range: float = 85.0

    # ---- Whole-image informational thresholds (never a failure alone) ------
    # BRIGHT_BACKGROUND note if whole-image white % is above this while the
    # tree ROI itself is not overexposed (= bright soil / sky, not a defect).
    whole_white_pct_note: float = 5.0


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    # view name -> folder containing that view's images
    image_dirs: Dict[str, str] = field(
        default_factory=lambda: {
            "top": str(RESEARCH_DIR / "Top View" / "Drought Picture 2"),
            "side": str(RESEARCH_DIR / "Side View" / "Drought Picture 1"),
        }
    )
    # Expected filename prefix per view, used as a consistency check only.
    view_filename_prefix: Dict[str, str] = field(
        default_factory=lambda: {"top": "drought2_", "side": "drought_"}
    )
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    # Field-measurement workbook that links image IDs to tree ID and date.
    metadata_excel: Optional[str] = str(RESEARCH_DIR / "Growth.xlsx")
    metadata_columns: Dict[str, str] = field(
        default_factory=lambda: {
            "top_image_id": "Top view image ID",
            "side_image_id": "side view image ID",
            "tree_id": "Olive_ID",
            "date": "Date",
            "variety": "Variety",
            "irrigation": "Irrigation Regime",
            "group": "Group",
        }
    )

    # Optional CSV with manual ROI boxes: filename,x,y,w,h (pixels in the
    # EXIF-rotated, full-resolution image). Overrides automatic ROI.
    manual_roi_csv: Optional[str] = str(PROJECT_DIR / "manual_roi_boxes.csv")

    output_dir: str = str(PROJECT_DIR / "outputs")

    # ------------------------------------------------------------- analysis
    # OpenCV decodes JPEGs at 1/reduce_factor resolution using DCT scaling
    # (1, 2, 4 or 8). 2 -> 2040x1536 for this dataset. All blur scores depend
    # on this value: thresholds calibrated at one factor are NOT valid at
    # another.
    reduce_factor: int = 2
    # Respect EXIF orientation (phone held in portrait). Keep True.
    apply_exif_orientation: bool = True

    # Grayscale level >= white_level counts as "nearly white"
    white_level: int = 250
    # Grayscale level <= dark_level counts as "very dark"
    dark_level: int = 10
    # A channel value >= clip_level counts as clipped (saturated sensor)
    clip_level: int = 254

    # ---------------------------------------------------------------- ROI
    use_roi: bool = True
    # Lab a* (OpenCV scale, 128 = neutral) below 128 - veg_a_star_offset is
    # treated as green vegetation. Olive leaves are grey-green, so the margin
    # is small on purpose.
    veg_a_star_offset: float = 4.0
    # Morphological closing kernel (fraction of image width) that joins
    # individual leaves into one tree blob.
    roi_close_kernel_frac: float = 0.04
    # Penalise components far from the image centre (trees are framed
    # roughly centrally; weeds tend to sit at the edges).
    roi_center_weight: float = 1.5
    # Fragments of the same tree are merged into the box when they overlap it
    # horizontally, are within this vertical gap (fraction of image size) and
    # hold at least this fraction of the main blob's vegetation pixels.
    roi_merge_gap_frac: float = 0.06
    roi_merge_min_rel_size: float = 0.05
    # Padding added around the detected tree box (fraction of box size)
    roi_pad_frac: float = 0.06
    # If the vegetation mask covers less than this % of the image, fall back
    # to the central box.
    roi_min_veg_pct: float = 0.3
    # Central fallback box (fractions of width / height)
    roi_fallback_box: tuple = (0.15, 0.10, 0.70, 0.85)  # x, y, w, h
    # If the chosen tree blob covers more than this % of the image the scene
    # is weed-dominated -> ROI is marked low-confidence.
    roi_max_box_pct: float = 85.0
    # Sharpness on leaves: vegetation mask is dilated by this many px (at
    # analysis resolution) so that leaf EDGES are included.
    veg_edge_dilate_px: int = 7
    # Minimum fraction of ROI that must be (dilated) vegetation to use the
    # leaf-edge sharpness score as primary score.
    min_veg_frac_for_leaf_blur: float = 0.03

    # Which blur score drives the BLURRY flag: "leaf", "roi" or "whole".
    # "leaf" falls back to "roi", then "whole", when not available.
    blur_score_source: str = "leaf"

    # Save ROI crops as files (JPEG q=95, in outputs/roi_crops). Originals
    # are never modified. Off by default (disk space); ROI boxes are always
    # stored as coordinates in the report.
    save_roi_crops: bool = False

    # ------------------------------------------------------------ geometry
    # Smallest acceptable side (pixels, full resolution)
    min_side_px: int = 1000
    # Abnormal if aspect ratio (long/short) deviates from view's mode by more
    # than this relative amount
    aspect_tolerance: float = 0.02

    # ---------------------------------------------------------- duplicates
    phash_size: int = 8  # 8 -> 64-bit pHash
    # Hamming distance <= strict -> always POSSIBLE_NEAR_DUPLICATE, whatever
    # the metadata relation (practically the same photo).
    near_dup_strict: int = 4
    # Hamming distance <= loose -> POSSIBLE_NEAR_DUPLICATE only for pairs
    # whose metadata relation makes similarity suspicious (same view,
    # different trees or same tree + same session). Same tree on different
    # dates and top-vs-side pairs are expected to look alike and are only
    # listed for information.
    # 6 chosen after review: pairs of DIFFERENT trees at distance 8-10 were
    # clearly different photographs with similar soil backgrounds.
    near_dup_loose: int = 6
    compare_across_views: bool = True
    # Two dates closer than this many days are treated as one sampling session
    same_session_max_days: float = 3.0

    # ---------------------------------------------------------- execution
    # 0 -> min(8, cpu_count - 1). More workers did not help on a laptop
    # (memory bandwidth / hybrid cores); measure before raising it.
    workers: int = 0
    random_seed: int = 42

    # ------------------------------------------------------------- outputs
    contact_sheet_max_images: int = 60
    contact_sheet_cols: int = 6
    thumb_size: int = 260

    # Per-view thresholds
    thresholds: Dict[str, ViewThresholds] = field(
        default_factory=lambda: {
            "top": ViewThresholds(),
            "side": ViewThresholds(),
        }
    )

    def for_view(self, view: str) -> ViewThresholds:
        return self.thresholds.get(view, ViewThresholds())

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Config":
        return copy.deepcopy(self)


def default_config() -> Config:
    return Config()
