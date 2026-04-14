from __future__ import annotations

import numpy as np

from src.inference.monai_tutorial import (
    _DEFAULT_BASE_ANCHOR_SHAPES,
    _DEFAULT_RETURNED_LAYERS,
    _DEFAULT_VAL_PATCH_SIZE,
    MONAITutorialDetectionPipeline,
    is_monai_tutorial_model_path,
)


def test_is_monai_tutorial_model_path_accepts_standalone_pt_file(tmp_path) -> None:
    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"weights")

    assert is_monai_tutorial_model_path(model_path) is True


def test_is_monai_tutorial_model_path_rejects_bundle_model_file(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "model.pt"
    model_path.write_bytes(b"weights")

    assert is_monai_tutorial_model_path(model_path) is False


def test_tutorial_defaults_match_luna16_training_config() -> None:
    assert _DEFAULT_RETURNED_LAYERS == [1, 2]
    assert _DEFAULT_BASE_ANCHOR_SHAPES == [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
    assert _DEFAULT_VAL_PATCH_SIZE == [512, 512, 208]


def test_detect_candidates_keeps_foreground_label_zero(monkeypatch) -> None:
    pipeline = object.__new__(MONAITutorialDetectionPipeline)
    pipeline.device = "cpu"
    pipeline.amp = False
    pipeline.val_patch_size = [512, 512, 208]

    class FakeDetector:
        target_box_key = "box"
        target_label_key = "label"
        pred_score_key = "score"

        def __call__(self, inputs, use_inferer=False):
            return [{
                "box": FakeTorchTensor(np.array([[1.0, 2.0, 3.0, 5.0, 6.0, 7.0]], dtype=np.float32)),
                "label": FakeTorchTensor(np.array([0], dtype=np.int64)),
                "score": FakeTorchTensor(np.array([0.8], dtype=np.float32)),
            }]

    class FakeTensor:
        def __init__(self, array):
            self.array = array

        def to(self, device):
            return self

        def __getitem__(self, idx):
            return FakeTensor(self.array[idx])

        def numel(self):
            return self.array.size

    class FakeTorchTensor:
        def __init__(self, array):
            self.array = array

        def to(self, dtype):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    pipeline.detector = FakeDetector()
    pipeline.postprocessing = lambda data: {
        "pred_box": FakeTorchTensor(np.array([[10.0, 20.0, 30.0, 4.0, 4.0, 4.0]], dtype=np.float32)),
        "pred_label": FakeTorchTensor(np.array([0], dtype=np.int64)),
        "pred_score": FakeTorchTensor(np.array([0.8], dtype=np.float32)),
    }

    item = {"image": FakeTensor(np.zeros((1, 8, 8, 8), dtype=np.float32))}
    candidates = MONAITutorialDetectionPipeline._detect_candidates(pipeline, item)

    assert len(candidates) == 1
    assert abs(candidates[0]["prob"] - 0.8) < 1e-6
