# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — 2026-03-22 (patch)

### Added
- **Frontend tests**: Vitest + React Testing Library unit tests for all four components (`StatusBanner`, `UploadZone`, `NoduleList`, `Viewer`) and `App`; Playwright E2E tests covering upload flow, status transitions, nodule list, saliency slider, and view tab switching.
- **GitHub Actions workflow** (`.github/workflows/frontend.yml`): runs unit tests with coverage and E2E tests (Chromium) on every push/PR touching `webapp/`.

### Fixed
- **Critical** `src/data/luna16.py`, `lidc.py`: removed double `[::-1]` on `origin` when calling `world_to_voxel` — origin returned by `load_mhd` is already ZYX; extra reversal corrupted all voxel coordinates
- **Critical** `src/data/preprocessing.py`: `resample_to_isotropic` now sets `SetOutputOrigin` and `SetOutputDirection` on the SimpleITK resampler, preserving physical-space metadata
- **Critical** `src/webapp/api.py`: use `Path(file.filename).name` to prevent path traversal in file upload
- **Critical** `src/inference.py`, `src/webapp/tasks.py`: added `weights_only=True` to all `torch.load` calls to prevent arbitrary code execution from malicious checkpoints
- **Major** `scripts/dicom_to_luna16.py`: cast `RescaleSlope`/`RescaleIntercept` to `float` (not `int`) to avoid HU truncation
- **Major** `src/evaluation/froc.py`: replaced `assert` with `log.warning` so a non-monotonic sensitivity edge case does not crash the training loop
- **Major** `src/training/trainer.py`: log exception details when FROC computation fails instead of silently returning `cpm=0.0`; save `model` key in checkpoints to identify architecture unambiguously on load
- **Major** `src/inference.py`, `src/webapp/tasks.py`: model loading now uses the `model` checkpoint key instead of exception-driven architecture detection
- **Major** `src/inference/pipeline.py`: `_sliding_window_inference` now generates patches lazily per batch, eliminating the RAM spike from pre-collecting all patches
- **Major** `src/webapp/tasks.py`: replaced `raise self.retry(exc=exc, max_retries=0)` with plain `raise` so Celery marks failed tasks as `FAILURE` immediately
- **Minor** `src/interpretability/gradcam.py`: reset `_activations`/`_gradients` to `None` at the start of each `__call__` to prevent stale state on repeated calls
- **Minor** `src/models/hybrid/hybrid_net.py`: removed full-resolution `ds1` from `ds_logits`; deep supervision should only include lower-resolution outputs
- **Minor** `src/data/luna16.py`: moved `import os` to module level; `_find_mhd` now searches only the dataset's configured folds, preventing accidental access to val/test data during training

---

## [Unreleased] — 2026-03-22

### Added
- Project scaffold: directory structure, `docker-compose.yml`, Dockerfiles (api, worker, frontend), `requirements.txt`, `pyproject.toml`, `.env.example`
- Shared model components: `ResidualBlock`, `ResidualBlockSE`, `ChannelAttention`, `ConvBnRelu` (`src/models/shared/blocks.py`)
- Loss functions: `DiceBCELoss`, `DiceFocalLoss`, `FocalLoss` (`src/models/shared/losses.py`)
- Baseline 3D U-Net with 4-level encoder-decoder and skip connections (`src/models/baseline/unet3d.py`)
- Hybrid Res-U-Net + Swin Transformer bottleneck with deep supervision (`src/models/hybrid/hybrid_net.py`, `swin3d.py`)
- FP reduction classifier: 3D CNN on 32³ patches with OHEM loss (`src/fp_reduction/classifier.py`)
- Data pipeline: `LUNA16Dataset`, `LIDCDataset`, preprocessing utilities (`src/data/`)
- Training loop with W&B logging, top-k checkpoint saving, cosine LR scheduler (`src/training/trainer.py`, `src/train.py`)
- FROC evaluation: CPM at standard LUNA16 FP/scan thresholds, per-scan greedy matching (`src/evaluation/froc.py`, `src/evaluate.py`)
- Interpretability: Grad-CAM for baseline, Swin bottleneck attention extractor for hybrid (`src/interpretability/`)
- Two-stage inference pipeline: sliding-window segmentation → FP classifier, artefact writer (`src/inference/pipeline.py`, `src/inference.py`)
- Slice renderer: lung window → jet saliency overlay → candidate circles → annotated PNG (`src/webapp/renderer.py`)
- Web app: FastAPI API, Celery worker tasks, React + Vite frontend (`src/webapp/api.py`, `src/webapp/tasks.py`, `webapp/`)
- DICOM-to-LUNA16 converter: DICOM parse → 1 mm resample → lung segmentation → MHD export (`scripts/dicom_to_luna16.py`)
- ONNX export script for all model types (`scripts/export_onnx.py`)
- Hydra configs for `baseline`, `hybrid`, `fp_reduction`, `webapp` experiments (`configs/`)
- Tests: FROC sanity checks, model forward passes, loss functions (`tests/`)
- `README.md`: full project documentation with quickstart, API reference, metric definition
- `CHANGELOG.md`: this file
