"""
Image discovery, integrity checks and read-only loading.

Originals are opened read-only and never written. Each file is read ONCE
into memory; the same bytes are used for the SHA-256 hash, the format /
truncation checks and the OpenCV decode.

Integrity checks (-> CORRUPTED when any fails):
  * file is non-empty
  * format (magic bytes) is a supported raster format
  * JPEG files end with the EOI marker (FFD9) -> otherwise TRUNCATED
  * Pillow can parse the header and verify the stream
  * OpenCV can decode the pixels
  * decoded array is non-empty, 3-channel, and not a constant image
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from config import Config

log = logging.getLogger(__name__)

MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"II*\x00": "TIFF",
    b"MM\x00*": "TIFF",
    b"BM": "BMP",
}

EXIF_ORIENTATION_TAG = 274
EXIF_DATETIME_TAG = 306


@dataclass
class ImageRecord:
    path: Path
    view: str


@dataclass
class LoadResult:
    ok: bool
    sha256: str = ""
    file_size_bytes: int = 0
    format: str = ""
    width: int = 0              # full-resolution, after EXIF rotation
    height: int = 0
    exif_orientation: Optional[int] = None
    exif_datetime: Optional[str] = None
    camera_model: Optional[str] = None
    image_bgr: Optional[np.ndarray] = None  # analysis-resolution, EXIF-rotated
    errors: List[str] = field(default_factory=list)


def discover_images(cfg: Config, views: Optional[List[str]] = None) -> List[ImageRecord]:
    recs: List[ImageRecord] = []
    for view, folder in cfg.image_dirs.items():
        if views and view not in views:
            continue
        p = Path(folder)
        if not p.exists():
            log.error("Image folder for view '%s' not found: %s", view, p)
            continue
        files = sorted(
            (f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in cfg.image_extensions),
            key=lambda f: _natural_key(f.name),
        )
        recs.extend(ImageRecord(f, view) for f in files)
        log.info("View '%s': %d image files in %s", view, len(files), p)
    return recs


def _natural_key(name: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _clean(s) -> Optional[str]:
    if s is None:
        return None
    return str(s).replace("\x00", "").strip() or None


def _reduce_flag(factor: int, exif: bool) -> int:
    flags = {
        1: cv2.IMREAD_COLOR,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
        8: cv2.IMREAD_REDUCED_COLOR_8,
    }
    if factor not in flags:
        raise ValueError("reduce_factor must be 1, 2, 4 or 8")
    f = flags[factor]
    if not exif:
        f |= cv2.IMREAD_IGNORE_ORIENTATION
    return f


def load_image(path: Path, cfg: Config) -> LoadResult:
    """Read and validate one image. Never raises; problems go to .errors."""
    res = LoadResult(ok=False)
    try:
        data = path.read_bytes()
    except OSError as e:
        res.errors.append(f"UNREADABLE_FILE:{e}")
        return res

    res.file_size_bytes = len(data)
    res.sha256 = hashlib.sha256(data).hexdigest()
    if not data:
        res.errors.append("EMPTY_FILE")
        return res

    res.format = next((fmt for magic, fmt in MAGIC.items() if data.startswith(magic)), "")
    if not res.format:
        res.errors.append("UNSUPPORTED_OR_UNKNOWN_FORMAT")
        return res
    if res.format == "JPEG" and not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
        res.errors.append("TRUNCATED_JPEG_NO_EOI")

    # --- Pillow: header, EXIF, stream verification
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            exif = im.getexif()
            res.exif_orientation = exif.get(EXIF_ORIENTATION_TAG)
            res.exif_datetime = _clean(exif.get(EXIF_DATETIME_TAG))
            res.camera_model = _clean(exif.get(272))
            im.verify()
        if cfg.apply_exif_orientation and res.exif_orientation in (5, 6, 7, 8):
            w, h = h, w
        res.width, res.height = int(w), int(h)
    except Exception as e:  # noqa: BLE001 - any parse failure is a finding
        res.errors.append(f"PIL_CANNOT_PARSE:{type(e).__name__}")

    # --- OpenCV decode (the library used for all measurements)
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, _reduce_flag(cfg.reduce_factor, cfg.apply_exif_orientation))
    except cv2.error as e:
        img = None
        res.errors.append(f"OPENCV_ERROR:{e.msg if hasattr(e, 'msg') else e}")
    if img is None:
        res.errors.append("OPENCV_CANNOT_DECODE")
        return res
    if img.size == 0 or img.ndim != 3 or img.shape[2] != 3:
        res.errors.append(f"INVALID_ARRAY_SHAPE:{img.shape}")
        return res
    if float(img.std()) < 1.0:
        res.errors.append("CONSTANT_IMAGE_NO_CONTENT")
    if not res.width:
        res.height, res.width = img.shape[0] * cfg.reduce_factor, img.shape[1] * cfg.reduce_factor

    res.image_bgr = img
    res.ok = not res.errors
    return res


def load_thumbnail(path: Path, max_side: int = 400) -> Optional[np.ndarray]:
    """Small EXIF-rotated BGR preview for contact sheets (read-only)."""
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_REDUCED_COLOR_8)
        if img is None:
            return None
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return img
    except Exception:  # noqa: BLE001
        return None


def full_res_to_analysis(box: Tuple[int, int, int, int], factor: int) -> Tuple[int, int, int, int]:
    return tuple(int(round(v / factor)) for v in box)  # type: ignore[return-value]
