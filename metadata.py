"""
Link image files to field metadata (tree ID, date, variety, irrigation).

Filenames in this dataset carry only a running number (e.g. drought2_80.jpg),
NOT a tree ID or date. The link is provided by the field workbook
(Growth.xlsx), whose columns "Top view image ID" / "side view image ID" hold
the file stem of each photograph. This module reads that table; nothing is
inferred from the number itself.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import Config

log = logging.getLogger(__name__)

META_FIELDS = ["tree_id", "date", "variety", "irrigation", "group"]


def load_metadata(cfg: Config) -> Dict[str, Dict[str, dict]]:
    """Return {view: {file_stem: {tree_id, date, ...}}}. Empty if unavailable."""
    out: Dict[str, Dict[str, dict]] = {"top": {}, "side": {}}
    if not cfg.metadata_excel or not Path(cfg.metadata_excel).exists():
        log.warning("Metadata workbook not found (%s); tree/date columns will be empty.",
                    cfg.metadata_excel)
        return out

    c = cfg.metadata_columns
    df = pd.read_excel(cfg.metadata_excel)
    missing = [v for v in c.values() if v not in df.columns]
    if missing:
        log.warning("Metadata workbook lacks columns %s; they will be empty.", missing)

    for view, id_col in (("top", c["top_image_id"]), ("side", c["side_image_id"])):
        if id_col not in df.columns:
            continue
        sub = df.dropna(subset=[id_col])
        for _, row in sub.iterrows():
            stem = str(row[id_col]).strip()
            stem = Path(stem).stem  # tolerate "xxx.jpg" in the sheet
            rec = {}
            for key in META_FIELDS:
                col = c.get(key)
                val = row[col] if col in df.columns else None
                if key == "date" and val is not None and not pd.isna(val):
                    val = pd.Timestamp(val)
                rec[key] = None if (val is None or (not isinstance(val, pd.Timestamp) and pd.isna(val))) else val
            if stem in out[view]:
                log.warning("Image ID %s appears more than once in the %s column.", stem, id_col)
                rec["metadata_note"] = "ID_REPEATED_IN_SHEET"
            out[view][stem] = rec
    log.info("Metadata loaded: %d top-view and %d side-view image IDs.",
             len(out["top"]), len(out["side"]))
    return out


def lookup(meta: Dict[str, Dict[str, dict]], view: str, stem: str) -> Optional[dict]:
    return meta.get(view, {}).get(stem)
