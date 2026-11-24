"""
Exact and near-duplicate detection.

Exact duplicates: identical SHA-256 of the file bytes (byte-for-byte copies,
whatever their names or folders).

Near duplicates: perceptual hash (pHash, DCT-based; ImageHash library) on
the decoded image, compared by Hamming distance. A difference hash (dHash)
distance is reported alongside as a second opinion.

Why metadata matters here
-------------------------
In a longitudinal dataset the same young tree photographed one week later
from the same spot legitimately looks almost the same, and the "top" and
"side" photographs of one tree on one date are taken seconds apart. Treating
visual similarity alone as duplication would remove real observations. Each
pair is therefore classified by its metadata relation:

  same_view_same_tree_same_session   suspicious (only 1 photo expected)
  same_view_diff_tree                suspicious (mislabelled / copied file?)
  same_view_same_tree_diff_date      EXPECTED similarity
  cross_view_same_tree_same_session  EXPECTED similarity
  cross_view_other                   suspicious only if very close
  unknown_metadata                   treated as suspicious

Flagging rule:
  distance <= near_dup_strict                     -> POSSIBLE_NEAR_DUPLICATE
                                                     (any relation; essentially
                                                     the same picture)
  distance <= near_dup_loose and relation is
  suspicious                                       -> POSSIBLE_NEAR_DUPLICATE
  otherwise the pair is listed for information only (flagged = False).

Nothing is ever deleted; every candidate pair goes to duplicate_pairs.csv
with empty review columns, and a side-by-side contact sheet is produced.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:  # pragma: no cover
    imagehash = None

SUSPICIOUS_RELATIONS = {
    "same_view_same_tree_same_session",
    "same_view_diff_tree",
    "unknown_metadata",
}


def perceptual_hashes(img_bgr: np.ndarray, hash_size: int = 8) -> Dict[str, str]:
    if imagehash is None:
        raise ImportError("ImageHash is required: pip install ImageHash")
    rgb = img_bgr[..., ::-1]
    pil = Image.fromarray(np.ascontiguousarray(rgb))
    pil.thumbnail((512, 512))
    return {
        "phash": str(imagehash.phash(pil, hash_size=hash_size)),
        "dhash": str(imagehash.dhash(pil, hash_size=hash_size)),
    }


def _hex_to_bits(hexes: List[str]) -> np.ndarray:
    """Array of shape (n, nbits) of bools."""
    rows = []
    for h in hexes:
        b = bytes.fromhex(h)
        rows.append(np.unpackbits(np.frombuffer(b, dtype=np.uint8)))
    return np.vstack(rows).astype(bool)


def exact_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Map each row to an exact-duplicate group id (or '' if unique)."""
    ok = df["sha256"].fillna("") != ""
    counts = df.loc[ok, "sha256"].value_counts()
    dup_hashes = counts[counts > 1].index
    gid = {h: f"EXD{i + 1:04d}" for i, h in enumerate(sorted(dup_hashes))}
    return df["sha256"].map(lambda h: gid.get(h, "")).fillna("")


def _relation(a: dict, b: dict, same_session_days: float) -> str:
    def missing(v):
        return v is None or (not isinstance(v, str) and pd.isna(v)) or v == ""
    if missing(a.get("tree_id")) or missing(b.get("tree_id")) or missing(a.get("date")) or missing(b.get("date")):
        return "unknown_metadata"
    same_view = a["view"] == b["view"]
    same_tree = a["tree_id"] == b["tree_id"]
    days = abs((pd.Timestamp(a["date"]) - pd.Timestamp(b["date"])).total_seconds()) / 86400.0
    same_session = days <= same_session_days
    if same_view:
        if not same_tree:
            return "same_view_diff_tree"
        return "same_view_same_tree_same_session" if same_session else "same_view_same_tree_diff_date"
    if same_tree and same_session:
        return "cross_view_same_tree_same_session"
    return "cross_view_other"


def find_near_duplicates(df: pd.DataFrame, strict: int, loose: int, across_views: bool,
                         same_session_days: float) -> pd.DataFrame:
    """Return candidate pairs with Hamming distance <= loose."""
    valid = df[df["phash"].fillna("") != ""].reset_index()
    cols = ["image_a", "image_b", "view_a", "view_b", "tree_a", "tree_b", "date_a", "date_b",
            "phash_distance", "dhash_distance", "relation", "flagged", "pair_id",
            "review_decision", "review_notes"]
    if len(valid) < 2:
        return pd.DataFrame(columns=cols)

    # Hamming distances for all pairs at once: d(a,b) = a.(1-b) + (1-a).b
    P = _hex_to_bits(valid["phash"].tolist()).astype(np.int32)
    D = _hex_to_bits(valid["dhash"].tolist()).astype(np.int32)
    dist = P @ (1 - P).T + (1 - P) @ P.T
    iu, ju = np.nonzero(np.triu(dist <= loose, k=1))
    if not across_views:
        v = valid["view"].to_numpy()
        keep = v[iu] == v[ju]
        iu, ju = iu[keep], ju[keep]
    if len(iu) == 0:
        return pd.DataFrame(columns=cols)

    def col(name):
        return valid[name].to_numpy() if name in valid else np.full(len(valid), None)

    fn, pa, vw, tr = col("filename"), col("path"), col("view"), col("tree_id")
    dt = pd.to_datetime(valid["date"]) if "date" in valid else pd.Series(pd.NaT, index=valid.index)
    dt = dt.to_numpy()
    out = pd.DataFrame({
        "image_a": fn[iu], "image_b": fn[ju], "path_a": pa[iu], "path_b": pa[ju],
        "view_a": vw[iu], "view_b": vw[ju], "tree_a": tr[iu], "tree_b": tr[ju],
        "date_a": dt[iu], "date_b": dt[ju],
        "phash_distance": dist[iu, ju],
        "dhash_distance": (D[iu] != D[ju]).sum(axis=1),
    })
    out["relation"] = [
        _relation({"view": r.view_a, "tree_id": r.tree_a, "date": r.date_a},
                  {"view": r.view_b, "tree_id": r.tree_b, "date": r.date_b}, same_session_days)
        for r in out.itertuples(index=False)
    ]
    out["flagged"] = (out["phash_distance"] <= strict) | out["relation"].isin(SUSPICIOUS_RELATIONS)
    out = out.sort_values(["flagged", "phash_distance"], ascending=[False, True]).reset_index(drop=True)
    out["pair_id"] = [f"NDP{i + 1:05d}" for i in range(len(out))]
    out["review_decision"] = ""  # fill in: DUPLICATE / NOT_DUPLICATE
    out["review_notes"] = ""
    return out


def near_duplicate_ids(df: pd.DataFrame, pairs: pd.DataFrame) -> Dict[str, str]:
    """filename -> ';'-joined pair ids of FLAGGED pairs it belongs to."""
    m: Dict[str, List[str]] = {}
    if pairs is None or pairs.empty:
        return {}
    for _, p in pairs[pairs["flagged"]].iterrows():
        for k in ("path_a", "path_b"):
            m.setdefault(p[k], []).append(p["pair_id"])
    return {k: ";".join(v) for k, v in m.items()}


def hamming_hex(a: Optional[str], b: Optional[str]) -> int:
    if not a or not b:
        return -1
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))
