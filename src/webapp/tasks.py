"""Celery task definitions for async inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from celery import Celery

log = logging.getLogger(__name__)

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery("pulmodex", broker=BROKER_URL, backend=BACKEND_URL)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]


def _get_pipeline():
    """Lazy-load the inference pipeline (heavy, only in worker process)."""
    import torch
    from src.inference import InferencePipeline

    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    seg_ckpt = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best.ckpt")
    fp_ckpt = os.environ.get("FP_CHECKPOINT", "checkpoints/best_fp.ckpt")
    fp_threshold = float(os.environ.get("FP_THRESHOLD", "0.5"))

    def _load_seg(path, dev):
        ckpt = torch.load(path, map_location=dev, weights_only=True)
        model_name = ckpt.get("model", "unet3d")
        if model_name == "hybrid_net":
            from src.models.hybrid import HybridNet
            m = HybridNet()
        else:
            from src.models.baseline import UNet3D
            m = UNet3D()
        m.load_state_dict(ckpt["model_state_dict"])
        return m

    def _load_fp(path, dev):
        ckpt = torch.load(path, map_location=dev, weights_only=True)
        from src.fp_reduction import FPClassifier
        m = FPClassifier()
        m.load_state_dict(ckpt["model_state_dict"])
        return m

    seg_model = _load_seg(seg_ckpt, device)
    fp_model = _load_fp(fp_ckpt, device)

    return InferencePipeline(
        seg_model=seg_model,
        fp_model=fp_model,
        fp_threshold=fp_threshold,
        device=device,
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
        self.update_state(state="PROGRESS", meta={"step": "segmentation"})
        report = _pipeline.run(mhd_path, output_dir, seriesuid)

        # Render slices after inference
        self.update_state(state="PROGRESS", meta={"step": "rendering"})
        from src.webapp.renderer import render_slices
        render_slices(
            scan_output_dir=str(Path(output_dir) / seriesuid),
            fp_threshold=float(os.environ.get("FP_THRESHOLD", "0.5")),
        )

        return {"status": "done", "report": report}

    except Exception:
        log.exception(f"predict_task failed for {seriesuid}")
        raise
