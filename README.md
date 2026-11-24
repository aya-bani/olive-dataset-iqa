# Olive-tree image quality assessment (IQA)

Quality control for the olive drought-stress photographs (2 880 RGB images:
1 440 top-view + 1 440 side-view, 80 trees, 18 sampling sessions, 27 Mar – 25 Jul 2024).
The pipeline **flags** images for manual inspection. It never modifies, moves or deletes
an original file.

---

## 1. What was found in the dataset (inspection before coding)

| Item | Finding |
|---|---|
| Folders | `Top View/Drought Picture 2/` (top), `Side View/Drought Picture 1/` (side) → perspective comes from the **folder** |
| File names | `drought2_<N>.jpg` (top) and `drought_<N>.jpg` (side), N = 80…1519. The name holds **no tree ID or date** |
| Tree ID / date | Taken from `Growth.xlsx` (`Top view image ID`, `side view image ID` → `Olive_ID`, `Date`, `Variety`, `Irrigation Regime`, `Group`). Every file is listed, and each tree has exactly 18 images per view |
| ⚠ Numbering ≠ sessions | Do not derive dates from N. Rows with N = 320–479 mix two dates (17 and 24 Apr). Always use the workbook date |
| Images | All 4080×3072 JPEG from one phone (Xiaomi 23053RN02A). EXIF orientation 6 (portrait) for 98 %; 35 photos were shot in landscape |
| Composition | Both "views" are oblique hand-held shots of a centrally framed tree against soil, weeds (Mar–Apr), dry grass (Jun–Jul) and black drip pipes |

## 2. Project structure

```
image quality assessment/
├── config.py              all paths + thresholds (per view), documented
├── main.py                entry point / orchestration (parallel)
├── metadata.py            Growth.xlsx → tree, date, variety, irrigation
├── image_loader.py        discovery, SHA-256, corruption checks, read-only decode
├── roi_extraction.py      tree ROI: manual box > vegetation mask > central box
├── blur_detection.py      variance of Laplacian (whole / ROI / leaf edges), Tenengrad
├── exposure_analysis.py   brightness distribution, white/dark %, channel clipping
├── duplicate_detection.py SHA-256 exact, pHash/dHash near-duplicates + metadata relation
├── quality_flags.py       measurements → flags → final status
├── reporting.py           CSV / Excel / summary tables
├── visualization.py       histograms, scatter plots, contact sheets
├── calibrate.py           threshold calibration from manual labels
├── prepare_dataset.py     reviewed report → CNN manifest + tree-wise splits
├── manual_roi_boxes.example.csv
├── tests/test_pipeline.py unit tests on synthetic images
├── requirements.txt
└── outputs/               all results (git-ignored)
```

## 3. Running it on Windows (VS Code + Python)

1. Open the folder `olive-research\image quality assessment` in VS Code (*File → Open Folder*).
2. Open a terminal (*Terminal → New Terminal*, PowerShell). Create the environment once:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install -r requirements.txt
```

   If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
   In VS Code, pick the interpreter with *Ctrl+Shift+P → Python: Select Interpreter → .venv*.
3. Run the tests, then a small sample (20 random images per view, reproducible seed):

```powershell
python -m unittest discover -s tests -v
```

```powershell
python main.py --sample 20 --output outputs/test_sample
```

4. Run the full dataset (about 5 minutes with 8 worker processes):

```powershell
python main.py --output outputs/full_run_v1
```

5. After you edit thresholds in `config.py`, re-apply them in about 30 s without re-reading the
   images. Manual review columns are kept:

```powershell
python main.py --from-report outputs/full_run_v1/quality_report.csv --output outputs/full_run_v2
```

Other options: `--views side`, `--no-roi`, `--save-roi-crops`, `--workers 4`, `--no-plots`.
Each output folder contains `run_config.json` (all settings plus library versions) for reproducibility.

## 4. Methods

### 4.1 Tree ROI (why, and how)
In these photos the tree fills about 15–40 % of the frame. Gravel makes a wind-blurred tree look
"sharp", and bright soil makes a well-exposed tree look "overexposed". **Quality decisions are
therefore taken on the tree region.** Whole-image values are still reported, for comparison.

1. **Manual box** (`manual_roi_boxes.csv`: `filename,x,y,w,h` in full-resolution, EXIF-rotated
   pixels). This is the baseline / gold standard. Use it for images whose automatic box is wrong.
2. **Vegetation mask (default).** Pixels with Lab a\* below neutral are green. A vertically
   elongated closing joins leaves and thin stems. The blob with the best *green area ×
   centrality* score is chosen, and fragments above or below it are merged.
3. **Central box** fallback when almost no vegetation is found.

*Why no deep segmentation model?* QC only needs an approximate box around one centrally framed
tree. The colour rule is deterministic, transparent and fast, and needs no training data.
A learned model (SAM, Mask R-CNN) would add a large dependency and new failure modes without
changing the QC decision. It becomes useful later if the CNN should see background-masked trees.

### 4.2 Blur
Variance of the Laplacian is reported at three levels: whole image, ROI box, and **leaf edges**
(dilated vegetation pixels in the ROI). The leaf-edge score is the primary `blur_score`. Tenengrad
is reported as a cross-check. All scores are computed at half resolution (2040×1536,
`reduce_factor=2`). Thresholds are only valid at that resolution.

### 4.3 Exposure
For the whole image and the ROI: mean, std, p1/p5/p50/p95/p99, dynamic range (p95−p5),
% nearly white (≥250), % very dark (≤10), per-channel clipping (≥254), any/all-channel clipping,
channel means and HSV saturation.
**OVEREXPOSED requires clipping or a very high mean inside the tree ROI** (= leaf colour is lost).
Bright soil with a normal ROI only gets the info note `BRIGHT_BACKGROUND`, and the status stays OK.

### 4.4 Corruption
Each file is checked for: empty file; magic bytes of a supported format; JPEG EOI marker (catches
truncated files); Pillow header parse and `verify()`; OpenCV decode; array shape; constant image.

### 4.5 Duplicates
* **Exact:** identical SHA-256 → `EXACT_DUPLICATE`, with a shared `exact_dup_group`.
* **Near:** 64-bit pHash, Hamming distance (dHash is reported as a second opinion). Each pair is
  classified by metadata relation:
  * Same tree on a **different date**, and top-vs-side of the same tree/session, are *expected* to
    look alike. They are flagged only when nearly identical (distance ≤ `near_dup_strict` = 4).
  * Different trees in the same view are flagged up to `near_dup_loose` = 6.
  * All candidate pairs (≤ loose) are listed in `duplicate_pairs.csv` with empty
    `review_decision` / `review_notes` columns. Side-by-side sheets are produced.

## 5. Results of the first full run (`outputs/full_run_v2`)

| final_status | side | top |
|---|---|---|
| OK | 1350 | 1337 |
| REVIEW | 70 | 78 |
| POSSIBLE_NEAR_DUPLICATE | 15 | 14 |
| OVEREXPOSED | 4 | 7 |
| BLURRY | 0 | 3 |
| EXACT_DUPLICATE | 1 | 1 |
| CORRUPTED | 0 | 0 |

Points that need your attention:
* **`drought2_281.jpg` (top) and `drought_281.jpg` (side) are byte-identical** (tree C1L2,
  9 Apr). The same photo was filed as both views. The real top or side photo for that
  observation is missing.
* **About 16 top/side pairs are nearly identical** (pHash ≤ 4, e.g. 1125, 1142, 1229). They are
  different shots taken seconds apart from practically the same viewpoint. For these
  observations the two "perspectives" carry almost the same information. Keep this in mind for
  multimodal fusion.
* **LOW_CONTRAST (49)**: hazy / washed-out late-evening and dusty images, mostly Jun–Jul.
* **ORIENTATION_MISMATCH (35)**: photos shot in landscape. They are valid, but resizing them
  to a square CNN input frames them differently.
* **ROI weakness on defoliated trees**: strongly stressed trees with few green leaves (e.g.
  `drought_1102`, `drought_926`) get a tiny or wrong box (`ROI_LOW_CONFIDENCE`). These are
  biologically important. **Never exclude them because of the ROI.** Give them manual boxes
  if their scores matter.
* No corrupted files and no underexposed images (lowest ROI mean = 90).

## 6. Interpreting the report

`quality_report.csv` / `.xlsx` (sheet `report`) has one row per image:

* `final_status`: most severe category
  (CORRUPTED > EXACT_DUPLICATE > BLURRY > OVEREXPOSED > UNDEREXPOSED > POSSIBLE_NEAR_DUPLICATE > REVIEW > OK).
* `quality_flags`: **all** issues (`;`-separated), plus one boolean `flag_*` column each.
  REVIEW-level flags: `BLUR_BORDERLINE`, `BLUR_OUTLIER` (robust z < −3 within the view),
  `LOW_CONTRAST`, `ABNORMAL_DIMENSIONS`, `ORIENTATION_MISMATCH`, `METADATA_MISSING`, `VIEW_NAME_MISMATCH`.
* `info_notes`: not defects. `BRIGHT_BACKGROUND`, `ROI_LOW_CONFIDENCE`, `ROI_FALLBACK_CENTRAL_BOX`,
  `BLUR_NOT_ON_LEAVES`.
* Scores: `blur_score` (+ `blur_score_source`), `blur_lapvar_whole/roi/leaf`, `whole_*` and
  `roi_*` exposure values, ROI box (`roi_x…roi_h`, full-resolution pixels), hashes, `load_errors`.
* `manual_decision` / `manual_notes`: empty, for you to fill in.

Other outputs: `manual_review_list.csv` (all non-OK images), `duplicate_pairs.csv`, `summary_*.csv`
(stats per view, counts, not-OK per tree and per day), `plots/`, and `contact_sheets/<view>/<FLAG>.jpg`
(thumbnails with tree, date, scores and the yellow ROI box). `_reference_OK_*.jpg` sheets show
typical OK images for comparison.

## 7. Calibrating the thresholds

The defaults in `config.py` come from this dataset's distributions and from a visual check.
**They are starting points.** Calibrate each view separately:

1. Draw a stratified labelling sample. Half comes from the worst 15 % of scores, half from the rest:

```powershell
python calibrate.py sample --report outputs/full_run_v2/quality_report.csv --metric blur_score --n 40
```

2. Open each listed **original** at 100 % zoom (contact sheets are only an index). Fill `label`
   with 1 = unusable for this criterion and 0 = acceptable. If possible, use two people and
   resolve disagreements.
3. Fit the thresholds:

```powershell
python calibrate.py fit --labels calibration/labels_blur_score.csv --metric blur_score --direction below
```

   This prints, per view, sensitivity/specificity, the Youden-optimal threshold and the
   threshold reaching 95 % sensitivity. Suggested use: put the hard flag (`blur_threshold`) at the
   Youden value and the REVIEW band (`blur_review_threshold`) at the high-sensitivity value.
4. Repeat for `roi_clip_any_pct` / `roi_mean` (`--direction above`) and `roi_dynamic_range`.
5. Edit `config.py → thresholds["top"/"side"]`, re-run with `--from-report`, and keep the label
   files with the paper's supplementary material.

**Blur warning:** at 100 % zoom, sharp **extreme close-ups** of large leaves (e.g.
`drought2_438`, `drought_161`) score as low as truly blurred images (`drought_306`,
`drought_1257`). They contain little fine texture. The Laplacian score depends on content and
distance as well as focus. That is why the BLURRY threshold is low (300), the REVIEW band is wide
(<1500), and every BLURRY image must be confirmed by eye.

## 8. Inspecting flagged images and deciding on exclusion

1. Start with `contact_sheets/<view>/` per flag, `duplicate_pairs_flagged.jpg`, and the plots.
2. For each candidate, open the original. Decide on **biological usability**: can leaf colour,
   leaf density, canopy structure and branch distribution be judged?
   * **Exclude:** unreadable files; the second copy of an exact duplicate; foliage clearly out of
     focus or motion-smeared; foliage colour lost to clipping; wrong tree or wrong view
     (mislabelled).
   * **Keep:** bright soil or sky; harsh sun or shadows on a correctly exposed tree; weeds, pipes
     or a person in the background; landscape framing; a slightly soft image whose leaves are
     still distinguishable; **stressed, yellowed, defoliated or dead-looking trees**. These are
     the signal, not noise.
   * **When in doubt, keep it** and write a note. You can test its influence later as a
     sensitivity analysis.
3. Write `INCLUDE` or `EXCLUDE` in `manual_decision` (and a reason in `manual_notes`) in
   `quality_report.csv`. Fill `review_decision` in `duplicate_pairs.csv`.
4. Report counts and reasons per view in the methods section (`exclusion_log.csv`).

## 9. Preparing the cleaned dataset for EfficientNet-B0 / ResNet50

```powershell
python prepare_dataset.py --report outputs/full_run_v2/quality_report.csv
```

* By default only `CORRUPTED` files are excluded automatically. Your `manual_decision` values are
  applied, and extra copies of exact duplicates are dropped. Blur/exposure flags are excluded only
  if you confirm them, or if you opt in with `--exclude-statuses CORRUPTED BLURRY OVEREXPOSED`.
* Writes `manifest_included.csv`, `exclusion_log.csv`, and `paired_top_side.csv` (tree×date with both
  views, for fusion with SPAD, leaf temperature, height, trunk diameter and canopy cover from
  `Growth.xlsx`).
* **Splits are by tree** (all 18 dates of a tree in one split), stratified by variety × irrigation
  regime. A random image-level split would leak each tree's appearance into the test set and
  inflate accuracy.
* `--copy-to <folder>` makes byte-identical copies of the included originals. Originals are never touched.

Recommended input processing (in the training code, not on disk):
* Keep the EXIF orientation as decoded. Resize with aspect-ratio preservation (letterbox or pad)
  to 224×224 (EfficientNet-B0 / ResNet50) or 240–300 px. Optionally crop to the ROI box first,
  with a margin.
* Use **only** the ImageNet mean/std normalisation required by the pretrained weights.
* **No colour-altering preprocessing**: no histogram equalisation, CLAHE, white balance,
  saturation/hue jitter or contrast stretching on these images. Leaf colour and discoloration are
  measurement targets. Geometric augmentation (flip, small rotation, scale/crop) and mild
  brightness jitter are acceptable if they are reported.
* Track the image→tree→date link through `manifest_included.csv`, never through the file number.

## 10. Limitations of each metric

| Metric | Limitation |
|---|---|
| Laplacian variance | Content- and scale-dependent: close-ups, sparse trees and smooth leaves score low; noise inflates the score; cannot tell defocus from motion blur; depends on `reduce_factor` |
| Leaf-edge blur | Relies on the vegetation mask; few green pixels (defoliated or yellowed trees) → falls back to ROI |
| ROI (a\* mask) | Weeds near the tree can enlarge the box (Mar–Apr); yellowed or clipped-white leaves are not "green"; edge-framed trees are penalised |
| Exposure | Gray levels after the phone's tone curve (not radiometric); ROI box still contains soil; JPEG smears clipped areas |
| LOW_CONTRAST | Also triggered by genuinely uniform scenes; haze can be atmospheric (real conditions) |
| pHash | Insensitive to small changes by design; similar soil backgrounds give moderate distances between different trees |
| SHA-256 | Only finds byte-identical files (a re-saved copy is a near- not an exact duplicate) |
| Thresholds | Derived from this dataset's distributions; must be re-calibrated for other cameras or resolutions |
