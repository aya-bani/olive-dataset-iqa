"""
Sharpness / blur measures.

Primary metric: variance of the Laplacian (Pech-Pacheco et al., 2000).
The Laplacian responds to edges; a sharp image has many strong edges and a
high variance, a defocused or motion-blurred image a low one.

Three regions are scored:
  * whole  - entire frame
  * roi    - tree bounding box
  * leaf   - pixels on and around detected leaves (dilated vegetation mask
             inside the ROI). This isolates foliage edges from soil texture.

Why "leaf"? Gravel and cracked soil carry strong high-frequency texture.
A tree blurred by wind or by focusing on the ground can still produce a high
whole-image score. Scoring the foliage itself is the relevant test for leaf
shape, density and discoloration.

Limitations (see README):
  * The score is NOT absolute: it depends on scene content (a small sparse
    tree has fewer edges than a dense one), resolution (config.reduce_factor),
    JPEG compression and sensor noise (noise inflates the score).
  * It cannot tell defocus from motion blur, nor which part is in focus.
  * Thresholds must be calibrated per view on manually labelled images.

Secondary metric: Tenengrad (mean squared Sobel gradient magnitude), reported
for cross-checking; it is less sensitive to isolated noise pixels.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from roi_extraction import Box


def laplacian_map(gray: np.ndarray) -> np.ndarray:
    return cv2.Laplacian(gray, cv2.CV_64F, ksize=3)


def tenengrad_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(laplacian_map(gray).var())


def blur_scores(gray: np.ndarray, roi_box: Optional[Box], veg_mask: Optional[np.ndarray],
                dilate_px: int = 7, min_veg_frac: float = 0.03) -> Dict[str, float]:
    """Compute blur scores. Laplacian is computed once on the full frame and
    then summarised per region (avoids artificial edges at crop borders)."""
    lap = laplacian_map(gray)
    ten = tenengrad_map(gray)
    out: Dict[str, float] = {
        "blur_lapvar_whole": float(lap.var()),
        "blur_tenengrad_whole": float(ten.mean()),
        "blur_lapvar_roi": np.nan,
        "blur_tenengrad_roi": np.nan,
        "blur_lapvar_leaf": np.nan,
        "leaf_pixel_frac_roi": np.nan,
    }
    if roi_box is None:
        return out
    x, y, w, h = roi_box
    lap_roi = lap[y:y + h, x:x + w]
    out["blur_lapvar_roi"] = float(lap_roi.var())
    out["blur_tenengrad_roi"] = float(ten[y:y + h, x:x + w].mean())

    if veg_mask is not None:
        m = veg_mask[y:y + h, x:x + w]
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            m = cv2.dilate(m, k)
        frac = float(m.mean()) if m.size else 0.0
        out["leaf_pixel_frac_roi"] = frac
        if frac >= min_veg_frac:
            out["blur_lapvar_leaf"] = float(lap_roi[m > 0].var())
    return out


def primary_blur_score(scores: Dict[str, float], source: str = "leaf") -> tuple:
    """Return (score, source_used) following the configured fallback chain."""
    chain = {"leaf": ["leaf", "roi", "whole"], "roi": ["roi", "whole"], "whole": ["whole"]}[source]
    for s in chain:
        v = scores.get(f"blur_lapvar_{s}", np.nan)
        if v is not None and not np.isnan(v):
            return float(v), s
    return float("nan"), "none"
