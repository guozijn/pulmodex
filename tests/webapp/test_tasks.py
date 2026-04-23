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


def test_get_pipeline_respects_model_backend_override(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"weights")

    monkeypatch.setenv("MODEL_CHECKPOINT", str(model_path))
    monkeypatch.setenv("MODEL_BACKEND", "native")
    monkeypatch.setenv("FP_CHECKPOINT", str(tmp_path / "missing.ckpt"))
    monkeypatch.setattr(tasks, "_load_webapp_config", lambda: {"webapp": {}})
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    captured: dict[str, object] = {}

    class FakePrimaryModel:
        pass

    class FakePipeline:
        def __init__(
            self,
            primary_model,
            fp_model,
            fp_threshold: float,
            candidate_threshold: float,
            min_candidate_voxels: int,
            device: str,
            primary_patch_size: int,
        ) -> None:
            captured["primary_model"] = primary_model
            captured["fp_model"] = fp_model
            captured["device"] = device

    import src.models.loading as loading_mod

    monkeypatch.setattr(loading_mod, "load_checkpoint_model", lambda path, device: (FakePrimaryModel(), {}))

    import src.inference as inference_mod

    monkeypatch.setattr(inference_mod, "InferencePipeline", FakePipeline)

    pipeline = tasks._get_pipeline()

    assert isinstance(pipeline, FakePipeline)
    assert isinstance(captured["primary_model"], FakePrimaryModel)
    assert captured["fp_model"] is None
    assert captured["device"] == "cpu"


def test_resolve_model_backend_rejects_unknown_value() -> None:
    try:
        tasks._resolve_model_backend("mystery")
    except ValueError as exc:
        assert "MODEL_BACKEND" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid MODEL_BACKEND")


def test_predict_task_persists_original_scan_before_render(monkeypatch, tmp_path) -> None:
    staged_scan = tmp_path / "upload" / "scan.nii.gz"
    staged_scan.parent.mkdir(parents=True)
    staged_scan.write_bytes(b"original-nifti")
    output_dir = tmp_path / "outputs"

    class FakePipeline:
        fp_threshold = 0.5

        def run(self, scan_path: str, output_dir_arg: str, seriesuid: str) -> dict:
            assert scan_path == str(staged_scan)
            assert output_dir_arg == str(output_dir)
            assert seriesuid == "scan-123"
            return {"seriesuid": seriesuid, "candidates": []}

    monkeypatch.setattr(tasks, "_pipeline", FakePipeline())
    monkeypatch.setattr(tasks, "_load_webapp_config", lambda: {"webapp": {}})

    render_calls = {}

    def fake_render_slices(scan_output_dir: str, **kwargs) -> list[str]:
        render_calls["scan_output_dir"] = scan_output_dir
        render_calls["kwargs"] = kwargs
        return []

    import src.webapp.renderer as renderer_mod

    monkeypatch.setattr(renderer_mod, "render_slices", fake_render_slices)

    updates = []
    monkeypatch.setattr(tasks.predict_task, "update_state", lambda state, meta: updates.append((state, meta)))
    result = tasks.predict_task.__wrapped__(str(staged_scan), str(output_dir), "scan-123")

    persisted = output_dir / "scan-123" / "original_scan.nii.gz"
    assert persisted.read_bytes() == b"original-nifti"
    assert render_calls["scan_output_dir"] == str(output_dir / "scan-123")
    assert result == {"status": "done", "report": {"seriesuid": "scan-123", "candidates": []}}
