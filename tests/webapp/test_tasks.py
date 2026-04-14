from __future__ import annotations

from pathlib import Path

from src.webapp import tasks


def test_resolve_device_keeps_cpu() -> None:
    assert tasks._resolve_device("cpu") == "cpu"


def test_resolve_device_falls_back_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert tasks._resolve_device("cuda") == "cpu"
    assert tasks._resolve_device("cuda:0") == "cpu"


def test_get_pipeline_uses_monai_tutorial_adapter_for_standalone_pt(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"weights")

    monkeypatch.setenv("MODEL_CHECKPOINT", str(model_path))
    monkeypatch.setenv("FP_CHECKPOINT", str(tmp_path / "missing.ckpt"))
    monkeypatch.setattr(tasks, "_load_webapp_config", lambda: {"webapp": {}})
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    import src.inference as inference_mod

    class FakePipeline:
        def __init__(self, model_path: str, fp_model, fp_threshold: float, device: str) -> None:
            self.model_path = model_path
            self.fp_model = fp_model
            self.fp_threshold = fp_threshold
            self.device = device

    monkeypatch.setattr(inference_mod, "MONAITutorialDetectionPipeline", FakePipeline)

    pipeline = tasks._get_pipeline()

    assert isinstance(pipeline, FakePipeline)
    assert Path(pipeline.model_path) == model_path
    assert pipeline.fp_model is None
    assert pipeline.device == "cpu"
