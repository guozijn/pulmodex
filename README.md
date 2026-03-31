# Pulmodex

AI-powered lung nodule detection from CT scans using deep neural networks.

**Input:** DICOM CT scans
**Output:** Detection overlays · confidence maps · saliency maps · annotated slice PNGs · JSON report

---

## Models

| Model | Architecture | Loss | Patch |
|-------|-------------|------|-------|
| Baseline | 3D U-Net (encoder-decoder + skip connections) | Dice + BCE | 128³ |
| Hybrid | Res-U-Net + Swin Transformer bottleneck + deep supervision | Dice-Focal | 128³ |

Both expose `forward(x) -> {"seg": mask, "logits": raw}`.

**Project-native inference** is two-stage: sliding-window segmentation → false-positive reduction (3D CNN on 32³ patches, OHEM training).

**MONAI bundle inference** uses a pretrained 3D RetinaNet detector (e.g. `lung_nodule_ct_detection`) as the primary detector, followed by the same optional local FP reduction stage. Set `MODEL_CHECKPOINT` to a MONAI bundle directory to activate this path.

**Datasets:** LUNA16 (10-fold CV, primary) · LIDC-IDRI (≥3/4 radiologist consensus, secondary)

> **Note:** `LUNA16Dataset` and `LIDCDataset` classes are referenced in `src/train.py` and `src/evaluate.py` but are not yet implemented. The `src/data/` package currently provides preprocessing utilities only (`load_mhd`, `normalise_hu`, `resample_to_isotropic`, `extract_patch`). Dataset implementations are required before training from scratch.

---

## Quick Start

### Prerequisites

**System dependencies**

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Python | 3.11 | `python3 --version` |
| pip | 23+ | bundled with Python 3.11 |
| Node.js | 18 LTS | `node --version` |
| npm | 9+ | bundled with Node.js 18 |
| Docker | 24+ | `docker --version` |
| Docker Compose | v2 (plugin) | `docker compose version` |
| CUDA GPU | — | required for training; CPU-only inference is supported |

**Docker socket access (Linux)**

Your user must be in the `docker` group to run `make dev` or any `docker compose` target without `sudo`:

```bash
sudo usermod -aG docker $USER
exec su -l $USER   # apply in current shell, or log out and back in
```

**Python environment**

Recommended: create a local virtualenv instead of using a global Conda base env:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Frontend dependencies**

```bash
npm --prefix webapp install
```

**Environment file**

```bash
cp .env.example .env
# edit .env: set DEVICE, MODEL_CHECKPOINT, FP_CHECKPOINT
```

`MODEL_CHECKPOINT` accepts either a project `.ckpt` file or a MONAI bundle directory.

### Development workflow

- Install the project in editable mode with `pip install -e .`
- Use `pulmodex <command>` as the primary entry point
- Main commands: `train`, `evaluate`, `infer`, `detect`, `export-onnx`, `generate-mock-data`, `preprocess-cache`, `dicom-to-luna16`
- Detection is exposed as a grouped command namespace: `pulmodex detect <prepare|train|infer|evaluate>`
- Run tests with `python -m pytest`

### Prepare local LUNA16 data for MONAI detection

The repo already includes raw LUNA16 `.mhd/.raw` scans in `data/orig_datasets`, split JSONs in `data/LUNA16_datasplit/mhd_original`, and challenge annotations in `data/evaluationScript/annotations`. The MONAI detection prep step resolves those pieces into project-local manifests with absolute scan paths.

```bash
pulmodex detect prepare \
    --raw_data_dir data/orig_datasets \
    --split_dir data/LUNA16_datasplit/mhd_original \
    --output_dir data/monai_detection
```

This writes `data/monai_detection/dataset_fold*.json` and `dataset_index.json`.
Use the grouped form `pulmodex detect ...`; the old flat forms such as `detect-prepare` are no longer supported.

### Train the MONAI 3D detection model

```bash
pulmodex detect train \
    --fold 0 \
    --prepared_dir data/monai_detection \
    --checkpoint checkpoints/monai_detection_fold0.pt \
    --epochs 10 \
    --batch_size 2 \
    --patch_size 96 96 96
```

The training flow uses a MONAI 3D RetinaNet detector with random positive/negative crops from the prepared LUNA16 manifests.

### Run MONAI 3D detection inference

```bash
pulmodex detect infer \
    --checkpoint checkpoints/monai_detection_fold0.pt \
    --input_dir data/orig_datasets \
    --output_dir outputs/monai_detection
```

Each scan writes `ct_volume.nii.gz`, `seg_mask.nii.gz`, `confidence_map.nii.gz`, `saliency_map.nii.gz`, `candidates.csv`, and `report.json` under `outputs/monai_detection/<seriesuid>/`.

### Evaluate MONAI 3D detection on LUNA16

```bash
pulmodex detect evaluate \
    --checkpoint checkpoints/monai_detection_fold0.pt \
    --fold 0 \
    --prepared_dir data/monai_detection \
    --output outputs/detection_eval_fold0.json
```

This runs validation-fold inference, filters excluded annotations using the official LUNA16 exclusion list, and reports CPM plus sensitivity at the standard FROC FP/scan points.

### References

[1] Project MONAI Tutorials, Detection workflows and examples:
https://github.com/Project-MONAI/tutorials/tree/main/detection

[2] Cardoso MJ, Li W, Brown R, et al. MONAI: An open-source framework for deep learning in healthcare.
arXiv:2211.02701, 2022.
https://arxiv.org/abs/2211.02701

[3] Lin TY, Goyal P, Girshick R, He K, Dollar P. Focal Loss for Dense Object Detection.
Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.
https://arxiv.org/abs/1708.02002

[4] Lin TY, Dollar P, Girshick R, He K, Hariharan B, Belongie S. Feature Pyramid Networks for Object Detection.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
https://arxiv.org/abs/1612.03144

[5] Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge.
Medical Image Analysis, 42:1-13, 2017.
https://doi.org/10.1016/j.media.2017.06.015

### Convert DICOM to LUNA16 format

```bash
pulmodex dicom-to-luna16 \
    --input_dir data/raw_dicom \
    --output_dir data/processed
```

After conversion, edit `data/processed/candidates.csv` and `annotations.csv` to add ground-truth labels.

### Precompute training cache

The training dataset can cache each scan after isotropic resampling and HU normalization so repeated candidates from the same CT do not trigger full-volume preprocessing every time. This is most useful when you rerun training repeatedly on the same dataset and want to reduce CPU preprocessing overhead.

```bash
pulmodex preprocess-cache \
    --data_dir data/processed \
    --cache_dir data/processed/.cache/luna16_iso
```

Training configs already point `data.cache_dir` at this location by default.

For mock data, you can precompute a separate cache and pass it explicitly:

```bash
pulmodex preprocess-cache \
    --data_dir data/mock_luna16 \
    --cache_dir data/mock_luna16/.cache/luna16_iso
```

```bash
WANDB_MODE=disabled pulmodex train experiment=baseline \
    trainer.device=cpu \
    data_dir=data/mock_luna16 \
    data.cache_dir=data/mock_luna16/.cache/luna16_iso \
    data.patch_size=32 \
    data.batch_size=1 \
    data.num_workers=0 \
    trainer.max_epochs=1
```

Each cached scan writes:

```text
<seriesuid>_vol.npy
<seriesuid>_meta.json
```

Delete and rebuild the cache if preprocessing logic changes, such as resampling behavior, HU normalization, or metadata handling.

### Generate mock LUNA16 data

For local development and smoke tests, you can generate a tiny synthetic dataset that matches the project's LUNA16 reader layout.

```bash
pulmodex generate-mock-data --clean
```

This writes `data/mock_luna16/` with:

```text
subset0..subset9/
annotations.csv
candidates.csv
```

### Train

```bash
pulmodex train experiment=baseline
pulmodex train experiment=hybrid
pulmodex train experiment=fp_reduction
```

Checkpoints are saved to `checkpoints/`. W&B logging is enabled automatically if `wandb` is installed.

**Mac CPU smoke test** (MPS unsupported for `conv_transpose3d`):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 pulmodex train experiment=baseline \
    trainer.max_epochs=1 data.patch_size=32 data.batch_size=1
```

**Mock-data smoke test**:
```bash
source .venv/bin/activate
pulmodex generate-mock-data --clean
WANDB_MODE=disabled pulmodex train experiment=baseline \
    trainer.device=cpu \
    data_dir=data/mock_luna16 \
    data.patch_size=32 \
    data.batch_size=1 \
    data.num_workers=0 \
    trainer.max_epochs=1
```

This should complete a 1-epoch end-to-end training run without needing the full LUNA16 dataset.

### Evaluate

```bash
pulmodex evaluate --checkpoint checkpoints/baseline_best.ckpt --split test
```

Outputs CPM, per-FP-rate sensitivity, and mean Dice to `outputs/eval_results.json`.

### Run inference on new scans

```bash
pulmodex infer \
    --checkpoint checkpoints/hybrid_best.ckpt \
    --fp_checkpoint checkpoints/fp_reduction_best.ckpt \
    --candidate_threshold 0.5 \
    --min_candidate_voxels 10 \
    --primary_patch_size 256 \
    --fp_threshold 0.5 \
    --input_dir data/processed
```

`--checkpoint` accepts either:

- a project checkpoint such as `checkpoints/hybrid_best.ckpt`
- a MONAI bundle directory such as `checkpoints/monai_lung_nodule_ct_detection_0.6.8`

When using a MONAI bundle, the main detection path comes from the bundle and `--fp_checkpoint` is still used for the local false-positive reduction stage.

`candidate_threshold`, `min_candidate_voxels`, `primary_patch_size`, and `fp_threshold` are the main inference-time controls when using project-native checkpoints. In MONAI bundle mode, bundle-specific preprocessing and detector settings come from the bundle itself, while `fp_threshold` still applies to the local FP reduction stage.

If you want to retain more small nodules, start by lowering `candidate_threshold` and `min_candidate_voxels` on the validation split. Use training changes only if inference-time tuning still underperforms for the target size range.

Suggested tuning order for small-nodule sensitivity:

- Lower `candidate_threshold` first
- Lower `min_candidate_voxels` second
- Adjust `fp_threshold` after candidate recall is acceptable
- Increase or decrease `primary_patch_size` only if context or memory limits become the bottleneck

Artefacts written to `outputs/<seriesuid>/`:

```
seg_mask.nii.gz        binary detection mask / visualisation mask
confidence_map.nii.gz  confidence map
saliency_map.nii.gz    Grad-CAM / Swin attention; falls back to confidence map in bundle mode
ct_volume.nii.gz       CT proxy volume for slice rendering (bundle mode)
candidates.csv         detected nodules with coordinates and confidence
report.json            summary
slices/                annotated PNG slices (axial · coronal · sagittal)
```

### Export to ONNX

```bash
pulmodex export-onnx \
    --checkpoint checkpoints/baseline_best.ckpt \
    --model unet3d \
    --output checkpoints/model.onnx
```

`export-onnx` currently supports project checkpoints only. It does not export MONAI bundle directories.

---

## Web App

The inference web app can run fully in Docker Compose, while local development runs the app processes on the host and keeps Redis in Docker.

```bash
make docker-up
make docker-down
make docker-logs
```

The production-style API container uses environment-driven worker settings. Adjust the `.env` values below as needed.

Configuration precedence for the web inference stack:

1. `.env` and process environment variables
2. `configs/experiment/webapp.yaml`
3. `configs/config.yaml`

In other words, `webapp.yaml` provides web-specific defaults, and `.env` is the final override layer used by the running API and worker processes.

Important `.env` groups:

- Shared runtime:
  `DEVICE`, `LOG_LEVEL`, `CUDA_VISIBLE_DEVICES`
- Redis / async backend:
  `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
- API / worker process settings:
  `API_WORKERS`, `CELERY_WORKER_CONCURRENCY`, `CELERY_WORKER_LOGLEVEL`
- Primary detection model:
  `MODEL_CHECKPOINT`
  This can point to either a project `.ckpt` file or a MONAI bundle directory.
- False-positive reduction model:
  `FP_CHECKPOINT`, `FP_THRESHOLD`
- Project-native candidate generation knobs:
  `CANDIDATE_THRESHOLD`, `MIN_CANDIDATE_VOXELS`, `PRIMARY_PATCH_SIZE`

If `MODEL_CHECKPOINT` points at a MONAI bundle directory, the worker uses the bundle's own preprocessing and detection config and then applies the local FP reduction model.

For local development, start Redis in Docker and run the API, worker, and frontend on the host:

```bash
make redis-up
make dev-api
make dev-worker
make dev-frontend
```

Or run everything needed for development in one command:

```bash
make dev
```

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI — upload scans, poll jobs, fetch slices |
| Frontend | http://localhost:3000 | React — drag-drop upload, CT viewer, nodule list |

**API endpoints:**

```
POST /predict                              upload .zip DICOM series → {job_id, seriesuid}
GET  /status/{job_id}                      poll Celery task state → {state, progress, result, error}
GET  /slices/{uid}/{view}?idx=N&layer=...  fetch rendered PNG slice layer
GET  /slices/{uid}/{view}/index            list available slice indices for a view
GET  /scans                                list all completed scans (scan history), newest first
GET  /report/{uid}                         fetch JSON inference report for a scan
```

Supported slice layers:

- `layer=composite` renders the combined PNG
- `layer=base` returns the windowed CT slice with nodule circles drawn directly on it
- `layer=overlay` returns the transparent heatmap (JET colormap, signal pixels fully opaque; opacity controlled client-side)

Upload inputs:

- The frontend and API both only accept `.zip` upload for DICOM series
- `.zip` uploads are unpacked on the API side; the largest enclosed DICOM series is converted to a temporary `.mhd` before the worker runs

---

## Evaluation Metric

**CPM** (Competition Performance Metric) — LUNA16 standard:
mean sensitivity at FP/scan ∈ {0.125, 0.25, 0.5, 1, 2, 4, 8}.

TP = prediction centroid within `diameter_mm / 2` of annotation centroid.
Greedy matching by descending confidence. Per-scan average (never global pool).

**Target:** ≤ 1 FP/scan at sensitivity ≥ 0.85.

---

## Project Structure

```
src/
  data/             preprocessing utilities (load_mhd, normalise_hu, resample_to_isotropic, extract_patch)
  models/
    baseline/       3D U-Net
    hybrid/         Res-U-Net + Swin Transformer
    shared/         residual blocks, SE attention, losses
  training/         Trainer class
  evaluation/       FROC, Dice
  fp_reduction/     FP classifier + OHEM loss
  interpretability/ Grad-CAM (baseline), Swin attention (hybrid)
  inference/        sliding-window pipeline · MONAI bundle adapter · artefact writer
  webapp/           FastAPI · Celery tasks · slice renderer
configs/experiment/ baseline · hybrid · fp_reduction · webapp
tests/              mirrors src/
scripts/            dicom_to_luna16.py · export_onnx.py
webapp/             React frontend (Vite + JSX)
docker/             Dockerfiles (api · worker · frontend)
```

---

## Tests

```bash
python -m pytest tests/ -v
npm --prefix webapp test
```

Key test coverage:
- FROC sanity checks (all-zeros → 0.0, perfect → 1.0, sensitivity non-decreasing, radius matching)
- Model forward passes (UNet3D, HybridNet, FPClassifier)
- Loss functions (DiceBCELoss, DiceFocalLoss with deep supervision)
- Frontend component tests (upload flow, status banner, nodule list, viewer)

---

## Conventions

- **Coordinates:** LPS system
- **HU range:** clamp [−1000, 400] → normalise to [0, 1]
- **Patch size:** 128³ training / 256³ inference / 32³ FP classifier
- **Classes:** 0 background · 1 nodule · 2 lesion (multi-class)
- DICOM loading via `pydicom` only; no notebook-style code in `src/`

---

## License

MIT
