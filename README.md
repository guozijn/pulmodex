# Pulmodex

AI-powered lung nodule detection from CT scans.

Pulmodex currently focuses on a MONAI-backed inference web app: upload a zipped
DICOM CT series or a `.nii.gz` volume, run asynchronous detection, and review
rendered CT slices, nodule candidates, confidence maps, and downloadable
artefacts.

**Input:** zipped DICOM CT series or `.nii.gz` CT volume  
**Output:** rendered slice PNGs · candidate boxes · confidence maps · saliency maps · JSON report

---

## Current Inference Stack

The web worker supports three primary detector sources through `MODEL_CHECKPOINT`
and `MODEL_BACKEND`:

| Backend | Model source | Notes |
|---------|--------------|-------|
| `monai_bundle` | MONAI bundle directory with `configs/inference.json` and `models/model.pt` | Preferred production path |
| `monai_tutorial` | standalone TorchScript `.pt` from MONAI LUNA16 tutorial `luna16_training.py` | Uses tutorial RetinaNet defaults |
| `native` | Pulmodex project checkpoint `.ckpt` | Uses local sliding-window segmentation |
| `auto` | path-based detection | Default; infers one of the above |

After the primary detector, the worker can apply the local false-positive
reduction model from `FP_CHECKPOINT`. If that checkpoint is missing, the FP
stage is skipped and detection still runs.

Project-native model checkpoints are still supported:

| Model | Architecture | Loss | Patch |
|-------|--------------|------|-------|
| Baseline | 3D U-Net encoder-decoder with skip connections | Dice + BCE | 128³ |
| Hybrid | Res-U-Net + Swin Transformer bottleneck + deep supervision | Dice-Focal | 128³ |
| FP reduction | 3D CNN classifier on candidate patches | OHEM | 32³ |

---

## Quick Start

### Prerequisites

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| Python | 3.11 | `python3 --version` |
| pip | 23+ | bundled with Python 3.11 |
| Node.js | 18 LTS | `node --version` |
| npm | 9+ | bundled with Node.js 18 |
| Docker | 24+ | required for Redis / Compose workflow |
| Docker Compose | v2 plugin | `docker compose version` |
| CUDA GPU | optional | recommended for inference and required for practical training |

On Linux, make sure your user can access Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
exec su -l $USER
```

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
npm --prefix webapp install
```

Create the runtime environment file:

```bash
cp .env.example .env
```

Set at least these values in `.env`:

```bash
DEVICE=cuda
MODEL_BACKEND=auto
MODEL_CHECKPOINT=checkpoints/monai_lung_nodule_ct_detection
FP_CHECKPOINT=checkpoints/fp_reduction_best.ckpt
FP_THRESHOLD=0.5
```

Use `DEVICE=cpu` for CPU-only local smoke testing. Use `DEVICE=cuda` in the
worker when CUDA is available.

---

## Web App

The app has three runtime pieces:

- FastAPI API at `http://localhost:8010`
- Celery worker for inference
- React/Vite frontend at `http://localhost:3000`

Redis is used as the Celery broker and result backend.

### Local Development

Start Redis in Docker and run API, worker, and frontend on the host:

```bash
make dev
```

Equivalent separate terminals:

```bash
make redis-up
make dev-api
make dev-worker
make dev-frontend
```

Open:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8010 |

### Docker Compose

Run the full stack in containers:

```bash
make docker-up
```

Stop or inspect it with:

```bash
make docker-down
make docker-logs
```

The Compose stack mounts:

- `./checkpoints` read-only into API/worker containers
- `./outputs` for generated reports and rendered images
- `./uploads` for staged uploaded scans

### Runtime Configuration

Configuration precedence for web inference:

1. `.env` and process environment variables
2. `configs/experiment/webapp.yaml`
3. `configs/config.yaml`

Important environment variables:

| Variable | Purpose |
|----------|---------|
| `DEVICE` | `cpu`, `cuda`, or a CUDA device such as `cuda:0` |
| `MODEL_CHECKPOINT` | project `.ckpt`, MONAI bundle directory, or MONAI tutorial `.pt` |
| `MODEL_BACKEND` | `auto`, `native`, `monai_bundle`, or `monai_tutorial` |
| `FP_CHECKPOINT` | optional local false-positive classifier checkpoint |
| `FP_THRESHOLD` | FP classifier probability threshold |
| `CANDIDATE_THRESHOLD` | native segmentation candidate threshold |
| `MIN_CANDIDATE_VOXELS` | native segmentation connected-component minimum size |
| `PRIMARY_PATCH_SIZE` | native sliding-window patch size |
| `CELERY_BROKER_URL` | Redis broker URL |
| `CELERY_RESULT_BACKEND` | Redis result backend URL |
| `API_WORKERS` | API process worker count in container mode |
| `CELERY_WORKER_CONCURRENCY` | Celery worker concurrency |

For MONAI bundle inference, `MODEL_CHECKPOINT` must point to a bundle directory
that contains `configs/inference.json` and `models/model.pt`. The bundle supplies
its own detector configuration; `FP_THRESHOLD` still controls the optional local
FP reduction stage.

For MONAI tutorial inference, `MODEL_CHECKPOINT` must point to a standalone
TorchScript `.pt` file. The adapter uses the LUNA16 tutorial defaults for
anchors, score threshold, NMS threshold, spacing, and sliding-window ROI size.

### Uploads

Supported upload formats:

- `.zip`: DICOM files are unpacked, the largest CT series is selected, then the
  API converts it to a staged `.nii.gz`
- `.nii.gz`: validated with SimpleITK and passed directly to the worker

The DICOM conversion preserves physical geometry, resolves slice spacing from
DICOM metadata when possible, and conservatively crops obvious air/background.

### API Endpoints

```text
POST   /predict                              upload .zip or .nii.gz -> {job_id, seriesuid}
GET    /status/{job_id}                      poll Celery state/result/error
GET    /scans                                list persisted scan history
DELETE /scans/{uid}                          delete scan artefacts and upload staging data
GET    /report/{uid}                         fetch report.json
GET    /volume/{uid}                         download persisted original_scan.nii.gz
GET    /markups/{uid}                        download detected boxes as OBJ
GET    /slices/{uid}/{view}?idx=N            fetch rendered PNG slice
GET    /slices/{uid}/{view}/index            list available slice indices
```

`view` must be one of `axial`, `coronal`, or `sagittal`.

### Inference Artefacts

Each completed scan writes to `outputs/<seriesuid>/`:

```text
meta.json             upload metadata
original_scan.nii.gz  persisted staged input volume
ct_volume.nii.gz      preprocessed CT proxy volume for rendering
seg_mask.nii.gz       detection mask / visualisation mask
confidence_map.nii.gz confidence map
saliency_map.nii.gz   Grad-CAM, Swin attention, or zero fallback for MONAI paths
candidates.csv        detected candidates and confidence values
report.json           summary and candidate payload
slices/               axial, coronal, and sagittal PNG slices
```

`/markups/{uid}` exports detected nodule boxes as an OBJ file in RAS world
coordinates for loading into tools such as 3D Slicer.

---

## Command-Line Inference

The same inference adapters are available from the CLI:

```bash
pulmodex infer \
    --checkpoint checkpoints/monai_lung_nodule_ct_detection \
    --fp_checkpoint checkpoints/fp_reduction_best.ckpt \
    --input_dir data/processed \
    --output_dir outputs \
    --fp_threshold 0.5
```

`--checkpoint` accepts:

- a MONAI bundle directory
- a MONAI tutorial TorchScript `.pt`
- a Pulmodex project `.ckpt`

For native checkpoints, these extra controls affect candidate generation:

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

---

## Data Preparation

### Convert DICOM to LUNA16 Layout

```bash
pulmodex dicom-to-luna16 \
    --input_dir data/raw_dicom \
    --output_dir data/processed
```

After conversion, edit:

```text
data/processed/annotations.csv
data/processed/candidates.csv
```

Expected LUNA16-style layout:

```text
data/processed/
  subset0..subset9/
    <seriesuid>.mhd
    <seriesuid>.raw
  annotations.csv
  candidates.csv
```

### Generate Mock Data

For local smoke tests:

```bash
pulmodex generate-mock-data --clean
```

This writes a small synthetic LUNA16-style dataset to `data/mock_luna16/`.

### Precompute Training Cache

Training can cache isotropically resampled, HU-normalised CT volumes so repeated
candidates from the same scan do not re-run full-volume preprocessing.

```bash
pulmodex preprocess-cache \
    --data_dir data/processed \
    --cache_dir data/processed/.cache/luna16_iso
```

Each cached scan writes:

```text
<seriesuid>_vol.npy
<seriesuid>_meta.json
```

Delete and rebuild the cache after changes to resampling, HU normalisation, or
metadata handling.

---

## Training

Training uses Hydra configs in `configs/` and LUNA16-style folds.

```bash
pulmodex train experiment=baseline
pulmodex train experiment=hybrid
pulmodex train experiment=fp_reduction
```

Checkpoints are saved to `checkpoints/`. W&B logging is enabled automatically if
`wandb` is installed; disable it with `WANDB_MODE=disabled`.

Mock-data smoke test:

```bash
source .venv/bin/activate
pulmodex generate-mock-data --clean
WANDB_MODE=disabled pulmodex train experiment=baseline \
    trainer.device=cuda \
    data_dir=data/mock_luna16 \
    data.patch_size=32 \
    data.batch_size=1 \
    data.num_workers=0 \
    trainer.max_epochs=1
```

With a precomputed mock cache:

```bash
pulmodex preprocess-cache \
    --data_dir data/mock_luna16 \
    --cache_dir data/mock_luna16/.cache/luna16_iso

WANDB_MODE=disabled pulmodex train experiment=baseline \
    trainer.device=cuda \
    data_dir=data/mock_luna16 \
    data.cache_dir=data/mock_luna16/.cache/luna16_iso \
    data.patch_size=32 \
    data.batch_size=1 \
    data.num_workers=0 \
    trainer.max_epochs=1
```

Mac CPU smoke test:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 pulmodex train experiment=baseline \
    trainer.max_epochs=1 \
    data.patch_size=32 \
    data.batch_size=1
```

---

## Evaluation

```bash
pulmodex evaluate \
    --checkpoint checkpoints/baseline_best.ckpt \
    --data_dir data/processed \
    --split test \
    --output outputs/eval_results.json
```

Evaluation reports CPM, sensitivity at LUNA16 false-positive rates, and mean
Dice.

**CPM** is the mean sensitivity at FP/scan values:

```text
0.125, 0.25, 0.5, 1, 2, 4, 8
```

Matching is greedy by descending confidence. A prediction is counted as a true
positive when its centroid falls within `diameter_mm / 2` of an annotation
centroid.

---

## Export to ONNX

```bash
pulmodex export-onnx \
    --checkpoint checkpoints/baseline_best.ckpt \
    --model unet3d \
    --output checkpoints/model.onnx
```

ONNX export currently supports Pulmodex project checkpoints only. It does not
export MONAI bundle directories or standalone MONAI tutorial TorchScript files.

---

## Tests

```bash
python -m pytest -q
npm --prefix webapp test
npm --prefix webapp run test:e2e
```

Or run the full configured suite:

```bash
make test
```

---

## Project Structure

```text
src/
  data/             LUNA16 dataset and preprocessing utilities
  models/
    baseline/       3D U-Net
    hybrid/         Res-U-Net + Swin Transformer
    shared/         residual blocks, SE attention, losses
  training/         Trainer class
  evaluation/       FROC and Dice metrics
  fp_reduction/     FP classifier and OHEM loss
  interpretability/ Grad-CAM and Swin attention helpers
  inference/        native pipeline, MONAI bundle adapter, MONAI tutorial adapter
  webapp/           FastAPI API, Celery tasks, slice renderer
configs/
  config.yaml
  experiment/       baseline, hybrid, fp_reduction, webapp
scripts/            data conversion, cache, mock data, ONNX export
webapp/             React frontend
docker/             API, worker, and frontend Dockerfiles
tests/              Python tests
```

---

## Conventions

- Web/MONAI reports use RAS world coordinates for candidate boxes.
- LUNA16 data utilities work with `.mhd` / `.raw` volumes and CSV annotations.
- HU preprocessing clamps lung CT values and normalises intensities for model input.
- Default native patch sizes: 128³ for training, 256³ for inference, 32³ for FP reduction.
- DICOM upload conversion uses `pydicom` and SimpleITK.

---

## License

MIT

---

## References

[1] Project MONAI Tutorials, Detection workflows and examples:  
https://github.com/Project-MONAI/tutorials/tree/main/detection

[2] Cardoso MJ, Li W, Brown R, et al. MONAI: An open-source framework for deep learning in healthcare.  
https://arxiv.org/abs/2211.02701

[3] Lin TY, Goyal P, Girshick R, He K, Dollar P. Focal Loss for Dense Object Detection.  
https://arxiv.org/abs/1708.02002

[4] Lin TY, Dollar P, Girshick R, He K, Hariharan B, Belongie S. Feature Pyramid Networks for Object Detection.  
https://arxiv.org/abs/1612.03144

[5] Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT: the LUNA16 challenge.  
https://doi.org/10.1016/j.media.2017.06.015
