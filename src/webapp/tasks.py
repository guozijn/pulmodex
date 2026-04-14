"""Celery task definitions for async inference."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from celery import Celery
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery("pulmodex", broker=BROKER_URL, backend=BACKEND_URL)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]


def _env_value(key: str, default: str) -> str:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.split("#", 1)[0].strip() or default


def _load_webapp_config() -> dict:
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    exp_cfg = OmegaConf.load(ROOT / "configs" / "experiment" / "webapp.yaml")
    merged = OmegaConf.merge(cfg, exp_cfg)
    return OmegaConf.to_container(merged, resolve=True)


def _resolve_device(requested: str) -> str:
    import torch

    normalized = requested.strip()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        log.warning("Requested device %s is unavailable in the worker; falling back to cpu", normalized)
        return "cpu"
    return normalized


def _get_pipeline():
    """Lazy-load the primary detection pipeline (heavy, only in worker process)."""
    import torch

    from src.inference import InferencePipeline, MONAIBundleDetectionPipeline, is_monai_bundle_path
    from src.models.loading import load_checkpoint_model

    webapp_cfg = _load_webapp_config().get("webapp", {})
    device = _resolve_device(_env_value("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    primary_checkpoint = _env_value(
        "MODEL_CHECKPOINT",
        str(webapp_cfg.get("primary_checkpoint", "checkpoints/hybrid_best.ckpt")),
    )
    fp_ckpt = _env_value(
        "FP_CHECKPOINT",
        str(webapp_cfg.get("fp_checkpoint", "checkpoints/fp_reduction_best.ckpt")),
    )
    fp_threshold = float(_env_value("FP_THRESHOLD", str(webapp_cfg.get("fp_threshold", 0.5))))
    candidate_threshold = float(
        _env_value(
            "CANDIDATE_THRESHOLD",
            str(webapp_cfg.get("candidate_threshold", 0.5)),
        )
    )
    min_candidate_voxels = int(
        _env_value(
            "MIN_CANDIDATE_VOXELS",
            str(webapp_cfg.get("min_candidate_voxels", 10)),
        )
    )
    primary_patch_size = int(
        _env_value(
            "PRIMARY_PATCH_SIZE",
            str(webapp_cfg.get("primary_patch_size", 256)),
        )
    )

    fp_model = None
    fp_ckpt_path = Path(fp_ckpt)
    if fp_ckpt_path.exists():
        fp_model, _ = load_checkpoint_model(fp_ckpt, device)
    else:
        log.warning("FP checkpoint not found at %s — FP reduction stage will be skipped", fp_ckpt)

    if is_monai_bundle_path(primary_checkpoint):
        return MONAIBundleDetectionPipeline(
            bundle_dir=primary_checkpoint,
            fp_model=fp_model,
            fp_threshold=fp_threshold,
            device=device,
        )

    primary_model, _ = load_checkpoint_model(primary_checkpoint, device)
    return InferencePipeline(
        primary_model=primary_model,
        fp_model=fp_model,
        fp_threshold=fp_threshold,
        candidate_threshold=candidate_threshold,
        min_candidate_voxels=min_candidate_voxels,
        device=device,
        primary_patch_size=primary_patch_size,
    )


_pipeline = None  # module-level singleton in worker


@celery_app.task(bind=True, name="pulmodex.predict")
def predict_task(self, mhd_path: str, output_dir: str, seriesuid: str) -> dict:
    """Run inference pipeline on a single CT scan.

    Args:
        mhd_path: absolute path to .mhd file (uploaded to worker volume)
        output_dir: base output directory
        seriesuid: scan identifier used for artefact sub-folder

    Returns:
        report dict from InferencePipeline.run()
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = _get_pipeline()

    try:
        self.update_state(state="PROGRESS", meta={"step": "detection"})
        report = _pipeline.run(mhd_path, output_dir, seriesuid)

        # Render slices after inference
        self.update_state(state="PROGRESS", meta={"step": "rendering"})
        from src.webapp.renderer import render_slices
        webapp_cfg = _load_webapp_config().get("webapp", {})
        render_slices(
            scan_output_dir=str(Path(output_dir) / seriesuid),
            fp_threshold=_pipeline.fp_threshold,
            confident_color=tuple(webapp_cfg.get("confident_color", [0, 255, 0])),
            uncertain_color=tuple(webapp_cfg.get("uncertain_color", [0, 180, 0])),
        )

        return {"status": "done", "report": report}

    except Exception:
        log.exception(f"predict_task failed for {seriesuid}")
        raise
