# Pulmodex

AI-powered lung nodule detection from CT scans using deep neural networks.

**Input:** DICOM CT scans
**Output:** Segmentation masks · confidence maps · saliency maps · annotated slice PNGs · JSON report

---

## Models

| Model | Architecture | Loss | Patch |
|-------|-------------|------|-------|
| Baseline | 3D U-Net (encoder-decoder + skip connections) | Dice + BCE | 128³ |
| Hybrid | Res-U-Net + Swin Transformer bottleneck + deep supervision | Dice-Focal | 128³ |

Both expose `forward(x) -> {"seg": mask, "logits": raw}`.

Inference is two-stage: sliding-window segmentation → false-positive reduction (3D CNN on 32³ patches, OHEM training).

**Datasets:** LUNA16 (10-fold CV, primary) · LIDC-IDRI (≥3/4 radiologist consensus, secondary)

---

## Quick Start

### Prerequisites

- Python 3.11+, CUDA GPU for training
- `pip install -r requirements.txt`
- Copy `.env.example` → `.env` and set `DEVICE`, checkpoint paths

### Convert DICOM to LUNA16 format

```bash
python scripts/dicom_to_luna16.py \
    --input_dir data/raw_dicom \
    --output_dir data/processed
```

After conversion, edit `data/processed/candidates.csv` and `annotations.csv` to add ground-truth labels.

### Train

```bash
python src/train.py experiment=baseline
python src/train.py experiment=hybrid
python src/train.py experiment=fp_reduction
```

Checkpoints are saved to `checkpoints/`. W&B logging is enabled automatically if `wandb` is installed.

**Mac CPU smoke test** (MPS unsupported for `conv_transpose3d`):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/train.py experiment=baseline \
    trainer.max_epochs=1 data.patch_size=32 data.batch_size=1
```

### Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best.ckpt --split test
```

Outputs CPM, per-FP-rate sensitivity, and mean Dice to `outputs/eval_results.json`.

### Run inference on new scans

```bash
python src/inference.py \
    --checkpoint checkpoints/best.ckpt \
    --fp_checkpoint checkpoints/best_fp.ckpt \
    --input_dir data/processed
```

Artefacts written to `outputs/<seriesuid>/`:

```
seg_mask.nii.gz        binary segmentation
confidence_map.nii.gz  probability map
saliency_map.nii.gz    Grad-CAM / Swin attention
candidates.csv         detected nodules with coordinates and confidence
report.json            summary
slices/                annotated PNG slices (axial · coronal · sagittal)
```

### Export to ONNX

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/best.ckpt \
    --model unet3d \
    --output checkpoints/model.onnx
```

---

## Web App

The inference web app can run fully in Docker Compose, while local development runs the app processes on the host and keeps Redis in Docker.

```bash
make docker-up
make docker-down
make docker-logs
```

The production-style API container uses environment-driven worker settings. Adjust `API_WORKERS`, `CELERY_WORKER_CONCURRENCY`, and `CELERY_WORKER_LOGLEVEL` in `.env` if needed.

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
POST /predict                         upload .mhd → returns {job_id, seriesuid}
GET  /status/{job_id}                 poll Celery task state
GET  /slices/{uid}/{view}?idx=N       fetch rendered PNG slice
GET  /slices/{uid}/{view}/index       list available slice indices
GET  /report/{uid}                    fetch JSON report
```

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
  data/             preprocessing, LUNA16 & LIDC-IDRI datasets
  models/
    baseline/       3D U-Net
    hybrid/         Res-U-Net + Swin Transformer
    shared/         residual blocks, SE attention, losses
  training/         Trainer class
  evaluation/       FROC, Dice
  fp_reduction/     FP classifier + OHEM loss
  interpretability/ Grad-CAM (baseline), Swin attention (hybrid)
  inference/        two-stage pipeline + artefact writer
  webapp/           FastAPI · Celery tasks · slice renderer
configs/experiment/ baseline · hybrid · fp_reduction · webapp
tests/              mirrors src/
scripts/            dicom_to_luna16.py · export_onnx.py
webapp/             React frontend
docker/             Dockerfiles (api · worker · frontend)
```

---

## Tests

```bash
pytest tests/ -v
```

Key test coverage:
- FROC sanity checks (all-zeros → 0.0, perfect → 1.0, sensitivity non-decreasing, radius matching)
- Model forward passes (UNet3D, HybridNet, FPClassifier)
- Loss functions (DiceBCELoss, DiceFocalLoss with deep supervision)

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
