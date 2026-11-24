"""
Tree region-of-interest (ROI) extraction.

Why a ROI? In these field photographs the tree typically occupies 15-40 % of
the frame. Whole-image statistics are dominated by soil, gravel, weeds and
irrigation pipes: gravel makes a motion-blurred tree look "sharp", bright
soil makes a correctly exposed tree look "overexposed". Quality decisions
are therefore taken on the tree region.

Methods, in order of priority:
  1. "manual"      - box from manual_roi_boxes.csv (filename,x,y,w,h in
                     full-resolution, EXIF-rotated pixels). Baseline / gold
                     standard for calibration and for difficult images.
  2. "vegetation"  - classical colour segmentation: pixels with Lab a* below
                     neutral are green vegetation; a morphological closing
                     joins leaves into blobs; the blob with the best
                     (area x centrality) score is the tree.
  3. "fallback"    - fixed central box, used when too little vegetation is
                     found (e.g. very washed-out image).

Why not a deep segmentation model (SAM, Mask R-CNN, DeepLab)? For QUALITY
CONTROL we only need an approximate box around one tree that is framed
roughly centrally against soil. The colour method is transparent, has no
learned weights, is deterministic and runs in milliseconds. A learned model
would add a large dependency and its own failure modes without changing the
QC decision. A segmentation model becomes justified later if the CNN should
see background-masked trees; it is not needed to judge blur/exposure.

Known limitations (documented in README):
  * Weeds of similar colour next to the tree (early weeks) can be merged into
    the box -> box larger than the tree ("low confidence").
  * Strongly yellowed / desiccated or clipped-white leaves are not green and
    are missed by the mask; the closing usually still covers them, but the
    box can be slightly tight.
  * Trees framed at the image edge are penalised by the centrality prior.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import Config

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class ROIResult:
    box: Box                       # analysis-resolution pixels
    method: str                    # manual | vegetation | fallback
    confidence: str                # high | low
    veg_pct_image: float           # % of whole image classified as vegetation
    veg_mask: np.ndarray           # uint8 mask (analysis resolution), 1 = vegetation
    note: str = ""


def load_manual_boxes(path: Optional[str]) -> Dict[str, Box]:
    """Read manual boxes. Keys are file names (e.g. 'drought_80.jpg')."""
    boxes: Dict[str, Box] = {}
    if not path or not Path(path).exists():
        return boxes
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                boxes[row["filename"].strip()] = (
                    int(float(row["x"])), int(float(row["y"])),
                    int(float(row["w"])), int(float(row["h"])),
                )
            except (KeyError, ValueError):
                log.warning("Skipping malformed manual ROI row: %s", row)
    log.info("Loaded %d manual ROI boxes from %s", len(boxes), path)
    return boxes


def vegetation_mask(img_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Binary mask of green vegetation using the Lab a* (green-red) axis.

    a* is largely independent of brightness, so the mask is robust to sun /
    shade. Olive foliage is grey-green, so the offset from neutral is small.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[..., 1].astype(np.int16)
    L = lab[..., 0]
    mask = (a < 128 - cfg.veg_a_star_offset) & (L > 15)  # ignore deep shadow noise
    return mask.astype(np.uint8)


def _fallback_box(w: int, h: int, cfg: Config) -> Box:
    fx, fy, fw, fh = cfg.roi_fallback_box
    return int(fx * w), int(fy * h), int(fw * w), int(fh * h)


def extract_roi(img_bgr: np.ndarray, cfg: Config, manual_box_full: Optional[Box] = None) -> ROIResult:
    h, w = img_bgr.shape[:2]

    # Work at reduced size for speed; boxes are scaled back.
    scale = 512.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    mask_small = vegetation_mask(small, cfg)
    veg_pct = 100.0 * float(mask_small.mean())
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    if manual_box_full is not None:
        f = cfg.reduce_factor
        x, y, bw, bh = (int(round(v / f)) for v in manual_box_full)
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w - x), min(bh, h - y)
        if bw > 8 and bh > 8:
            return ROIResult((x, y, bw, bh), "manual", "high", veg_pct, mask_full)
        log.warning("Manual ROI box outside image; ignoring it.")

    if veg_pct < cfg.roi_min_veg_pct:
        return ROIResult(_fallback_box(w, h, cfg), "fallback", "low", veg_pct, mask_full,
                         note="too little vegetation detected")

    sh, sw = mask_small.shape
    k = max(3, int(cfg.roi_close_kernel_frac * sw) | 1)
    # Vertically elongated kernel: young olive trees are upright, and thin
    # stems between leaf clusters must not split the tree into pieces.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, 3 * k))
    closed = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return ROIResult(_fallback_box(w, h, cfg), "fallback", "low", veg_pct, mask_full,
                         note="no vegetation component")

    cx0, cy0 = sw / 2.0, sh / 2.0
    diag = np.hypot(sw, sh) / 2.0
    # actual vegetation pixels inside each component (not the closed area)
    veg_counts = np.bincount(labels.ravel(), weights=mask_small.ravel().astype(np.float64), minlength=n)
    best, best_score = -1, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        veg_in = float(veg_counts[i])
        dist = np.hypot(cents[i][0] - cx0, cents[i][1] - cy0) / diag
        score = veg_in * (1.0 - min(dist, 1.0)) ** cfg.roi_center_weight
        if area > 0 and score > best_score:
            best, best_score = i, score

    x, y, bw, bh = (int(v) for v in stats[best, :4])
    best_veg = float(veg_counts[best])

    # Merge nearby fragments of the same tree (e.g. crown separated from the
    # lower branches): sizeable components whose box lies within a small gap
    # of the current box and overlaps it horizontally.
    gap = int(cfg.roi_merge_gap_frac * max(sw, sh))
    merged = {best}
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            if i in merged:
                continue
            veg_i = float(veg_counts[i])
            if veg_i < cfg.roi_merge_min_rel_size * best_veg:
                continue
            cx, cy, cw, ch = (int(v) for v in stats[i, :4])
            h_overlap = min(x + bw, cx + cw) - max(x, cx)
            v_gap = max(cy - (y + bh), y - (cy + ch), 0)
            if h_overlap > 0 and v_gap <= gap:
                nx, ny = min(x, cx), min(y, cy)
                bw, bh = max(x + bw, cx + cw) - nx, max(y + bh, cy + ch) - ny
                x, y = nx, ny
                merged.add(i)
                changed = True

    px, py = int(cfg.roi_pad_frac * bw), int(cfg.roi_pad_frac * bh)
    x, y = max(0, x - px), max(0, y - py)
    bw, bh = min(sw - x, bw + 2 * px), min(sh - y, bh + 2 * py)

    inv = 1.0 / scale
    box = (int(x * inv), int(y * inv), int(bw * inv), int(bh * inv))
    box_pct = 100.0 * (bw * bh) / float(sw * sh)

    confidence, note = "high", ""
    if box_pct > cfg.roi_max_box_pct:
        confidence, note = "low", "vegetation covers most of frame (weeds?)"
    elif bw * bh < 0.02 * sw * sh:
        confidence, note = "low", "very small tree box"
    return ROIResult(box, "vegetation", confidence, veg_pct, mask_full, note)


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def box_to_full_res(box: Box, factor: int) -> Box:
    return tuple(int(v * factor) for v in box)  # type: ignore[return-value]
