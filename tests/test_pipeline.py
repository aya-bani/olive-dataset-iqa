"""
Unit tests on synthetic images (no dataset needed).

Run from the project folder:
    python -m unittest discover -s tests -v
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blur_detection import blur_scores, variance_of_laplacian  # noqa: E402
from config import Config  # noqa: E402
from duplicate_detection import (exact_duplicate_groups, find_near_duplicates,  # noqa: E402
                                 perceptual_hashes)
from exposure_analysis import exposure_stats  # noqa: E402
from image_loader import load_image  # noqa: E402
from quality_flags import assign_flags  # noqa: E402
from roi_extraction import extract_roi  # noqa: E402


def synthetic_tree(seed=0, h=800, w=600):
    """Brown textured soil with a grey-green 'tree' in the centre."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (90, 130, 170)  # BGR soil
    img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    cv2.line(img, (w // 2, h - 50), (w // 2, 150), (60, 80, 90), 6)
    for _ in range(120):
        cx = int(rng.normal(w / 2, 60))
        cy = int(rng.uniform(150, h - 120))
        ang = rng.uniform(0, 180)
        cv2.ellipse(img, (cx, cy), (28, 8), ang, 0, 360, (110, 160, 120), -1)
    return img


class TestBlur(unittest.TestCase):
    def test_blur_lowers_laplacian_variance(self):
        img = synthetic_tree()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = variance_of_laplacian(g)
        blurred = variance_of_laplacian(cv2.GaussianBlur(g, (0, 0), 3))
        self.assertGreater(sharp, 5 * blurred)

    def test_leaf_score_uses_vegetation(self):
        cfg = Config()
        img = synthetic_tree()
        roi = extract_roi(img, cfg)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = blur_scores(g, roi.box, roi.veg_mask)
        self.assertFalse(np.isnan(s["blur_lapvar_leaf"]))


class TestROI(unittest.TestCase):
    def test_roi_finds_central_tree(self):
        cfg = Config()
        roi = extract_roi(synthetic_tree(), cfg)
        self.assertEqual(roi.method, "vegetation")
        x, y, w, h = roi.box
        self.assertLess(x, 300)
        self.assertGreater(x + w, 300)
        self.assertLess(w, 500)

    def test_fallback_without_vegetation(self):
        cfg = Config()
        soil = np.full((800, 600, 3), (90, 130, 170), np.uint8)
        self.assertEqual(extract_roi(soil, cfg).method, "fallback")

    def test_manual_box_has_priority(self):
        cfg = Config()
        roi = extract_roi(synthetic_tree(), cfg, manual_box_full=(100, 100, 400, 600))
        self.assertEqual(roi.method, "manual")


class TestExposure(unittest.TestCase):
    def test_white_and_clipping(self):
        img = np.full((100, 100, 3), 255, np.uint8)
        s = exposure_stats(img, "x")
        self.assertAlmostEqual(s["x_white_pct"], 100.0)
        self.assertAlmostEqual(s["x_clip_any_pct"], 100.0)
        dark = np.zeros((100, 100, 3), np.uint8)
        self.assertAlmostEqual(exposure_stats(dark, "x")["x_dark_pct"], 100.0)


class TestLoaderAndDuplicates(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = Config()
        self.cfg.reduce_factor = 1
        img = synthetic_tree()
        cv2.imwrite(str(self.tmp / "a.jpg"), img)
        shutil.copy(self.tmp / "a.jpg", self.tmp / "a_copy.jpg")
        # near duplicate: re-encoded at lower quality and slightly brighter
        cv2.imwrite(str(self.tmp / "a_near.jpg"), cv2.convertScaleAbs(img, alpha=1.0, beta=6),
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
        cv2.imwrite(str(self.tmp / "other.jpg"), synthetic_tree(seed=5)[::-1])
        data = (self.tmp / "a.jpg").read_bytes()
        (self.tmp / "truncated.jpg").write_bytes(data[: len(data) // 2])
        (self.tmp / "empty.jpg").write_bytes(b"")
        (self.tmp / "garbage.jpg").write_bytes(b"this is not an image" * 50)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_corruption_checks(self):
        self.assertTrue(load_image(self.tmp / "a.jpg", self.cfg).ok)
        self.assertIn("EMPTY_FILE", load_image(self.tmp / "empty.jpg", self.cfg).errors)
        self.assertIn("UNSUPPORTED_OR_UNKNOWN_FORMAT", load_image(self.tmp / "garbage.jpg", self.cfg).errors)
        r = load_image(self.tmp / "truncated.jpg", self.cfg)
        self.assertFalse(r.ok)
        self.assertIn("TRUNCATED_JPEG_NO_EOI", r.errors)

    def test_exact_and_near_duplicates(self):
        rows = []
        for name in ("a.jpg", "a_copy.jpg", "a_near.jpg", "other.jpg"):
            r = load_image(self.tmp / name, self.cfg)
            h = perceptual_hashes(r.image_bgr)
            rows.append({"filename": name, "path": str(self.tmp / name), "view": "side",
                         "sha256": r.sha256, "tree_id": None, "date": pd.NaT, **h})
        df = pd.DataFrame(rows)
        groups = exact_duplicate_groups(df)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], "")
        self.assertEqual(groups[2], "")
        pairs = find_near_duplicates(df, strict=4, loose=10, across_views=True, same_session_days=3)
        names = {frozenset((a, b)) for a, b in zip(pairs["image_a"], pairs["image_b"])}
        self.assertIn(frozenset(("a.jpg", "a_near.jpg")), names)
        self.assertNotIn(frozenset(("a.jpg", "other.jpg")), names)

    def test_same_tree_different_date_not_flagged_unless_strict(self):
        df = pd.DataFrame([
            {"filename": "w1.jpg", "path": "w1", "view": "side", "sha256": "1", "tree_id": "A1L1",
             "date": pd.Timestamp("2024-04-01"), "phash": "ffff0000ffff0000", "dhash": "0" * 16},
            {"filename": "w2.jpg", "path": "w2", "view": "side", "sha256": "2", "tree_id": "A1L1",
             "date": pd.Timestamp("2024-04-08"), "phash": "ffff0000ffff00ff", "dhash": "0" * 16},
        ])
        pairs = find_near_duplicates(df, strict=4, loose=10, across_views=True, same_session_days=3)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.loc[0, "relation"], "same_view_same_tree_diff_date")
        self.assertFalse(pairs.loc[0, "flagged"])  # distance 8 > strict 4


class TestFlags(unittest.TestCase):
    def test_bright_soil_is_not_overexposure(self):
        cfg = Config()
        base = dict(view="side", decode_ok=True, load_errors="", exact_dup_group="", near_dup_pairs="",
                    width=3072, height=4080, tree_id="T", view_name_mismatch=False,
                    roi_confidence="high", roi_method="vegetation", blur_score_source="leaf",
                    blur_score=20000.0, roi_dark_pct=0.0, roi_dynamic_range=150.0)
        df = pd.DataFrame([
            {**base, "roi_mean": 150, "roi_clip_any_pct": 1.0, "whole_white_pct": 20.0},  # bright soil
            {**base, "roi_mean": 230, "roi_clip_any_pct": 30.0, "whole_white_pct": 20.0},  # burnt tree
        ])
        out = assign_flags(df, cfg)
        self.assertEqual(out.loc[0, "final_status"], "OK")
        self.assertIn("BRIGHT_BACKGROUND", out.loc[0, "info_notes"])
        self.assertEqual(out.loc[1, "final_status"], "OVEREXPOSED")

    def test_multiple_flags_preserved(self):
        cfg = Config()
        df = pd.DataFrame([dict(view="top", decode_ok=True, load_errors="", exact_dup_group="EXD0001",
                                near_dup_pairs="", width=3072, height=4080, tree_id="T",
                                view_name_mismatch=False, roi_confidence="high", roi_method="vegetation",
                                blur_score_source="leaf", blur_score=1.0, roi_mean=20, roi_dark_pct=80.0,
                                roi_clip_any_pct=0.0, roi_dynamic_range=10.0, whole_white_pct=0.0)])
        out = assign_flags(df, cfg)
        fl = out.loc[0, "quality_flags"].split(";")
        for f in ("EXACT_DUPLICATE", "BLURRY", "UNDEREXPOSED", "LOW_CONTRAST"):
            self.assertIn(f, fl)
        self.assertEqual(out.loc[0, "final_status"], "EXACT_DUPLICATE")


if __name__ == "__main__":
    unittest.main()
