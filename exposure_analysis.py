"""
Exposure statistics.

Computed for the whole image and for the tree ROI:
  * mean / std grayscale brightness
  * percentiles p1, p5, p50, p95, p99 (brightness distribution)
  * dynamic range p95 - p5 (low values = haze, lens fog, flare, low contrast)
  * % nearly-white pixels (gray >= white_level)
  * % very dark pixels   (gray <= dark_level)
  * per-channel clipping % (R, G, B >= clip_level), % pixels with ANY channel
    clipped and with ALL channels clipped
  * per-channel means and mean HSV saturation

Bright soil vs real overexposure
--------------------------------
Dry, sunlit soil is legitimately bright (often gray 180-230) and can contain
specular white stones or salt crust. A whole-image white % is therefore NOT
evidence of a bad photo. The decision is taken on the tree ROI:
  * OVEREXPOSED requires channel clipping or a very high mean INSIDE the ROI,
    i.e. foliage colour information has actually been lost.
  * A bright whole image with a normal ROI only receives the informational
    note BRIGHT_BACKGROUND.
Channel clipping is used rather than "white" because sunlit foliage can clip
in one channel (usually green/red) while still looking non-white, which
already distorts leaf colour.

Limitations: the ROI box still contains soil between branches; the measures
use gray levels after the camera's tone curve (not radiometric values); JPEG
compression slightly smears clipped regions.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def exposure_stats(img_bgr: np.ndarray, prefix: str, white_level: int = 250,
                   dark_level: int = 10, clip_level: int = 254, full: bool = True) -> Dict[str, float]:
    if img_bgr is None or img_bgr.size == 0:
        return {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.ravel()
    p1, p5, p50, p95, p99 = np.percentile(g, [1, 5, 50, 95, 99])
    b, gch, r = img_bgr[..., 0], img_bgr[..., 1], img_bgr[..., 2]
    clip_r, clip_g, clip_b = r >= clip_level, gch >= clip_level, b >= clip_level
    n = float(g.size)
    out = {
        f"{prefix}_mean": float(g.mean()),
        f"{prefix}_std": float(g.std()),
        f"{prefix}_p05": float(p5),
        f"{prefix}_p50": float(p50),
        f"{prefix}_p95": float(p95),
        f"{prefix}_dynamic_range": float(p95 - p5),
        f"{prefix}_white_pct": 100.0 * float((g >= white_level).sum()) / n,
        f"{prefix}_dark_pct": 100.0 * float((g <= dark_level).sum()) / n,
        f"{prefix}_clip_any_pct": 100.0 * float((clip_r | clip_g | clip_b).sum()) / n,
    }
    if full:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        out.update({
            f"{prefix}_p01": float(p1),
            f"{prefix}_p99": float(p99),
            f"{prefix}_clip_all_pct": 100.0 * float((clip_r & clip_g & clip_b).sum()) / n,
            f"{prefix}_clip_r_pct": 100.0 * float(clip_r.sum()) / n,
            f"{prefix}_clip_g_pct": 100.0 * float(clip_g.sum()) / n,
            f"{prefix}_clip_b_pct": 100.0 * float(clip_b.sum()) / n,
            f"{prefix}_mean_r": float(r.mean()),
            f"{prefix}_mean_g": float(gch.mean()),
            f"{prefix}_mean_b": float(b.mean()),
            f"{prefix}_mean_saturation": float(hsv[..., 1].mean()),
        })
    return out
