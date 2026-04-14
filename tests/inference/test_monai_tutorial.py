from __future__ import annotations

import numpy as np
import torch

import src.inference.monai_tutorial as monai_tutorial_mod
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
    pipeline.raw_box_clipper = lambda boxes, labels, spatial_size: (boxes, labels)

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
            self.shape = array.shape

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


def test_detect_candidates_preserves_raw_box_alignment_after_clipping() -> None:
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
                "box": FakeTorchTensor(np.array([
                    [-4.0, -4.0, -4.0, -1.0, -1.0, -1.0],
                    [10.0, 20.0, 30.0, 14.0, 24.0, 34.0],
                ], dtype=np.float32)),
                "label": FakeTorchTensor(np.array([0, 0], dtype=np.int64)),
                "score": FakeTorchTensor(np.array([0.2, 0.8], dtype=np.float32)),
            }]

    class FakeTensor:
        def __init__(self, array):
            self.array = array
            self.shape = array.shape

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
    pipeline.raw_box_clipper = lambda boxes, labels, spatial_size: (
        FakeTorchTensor(np.array([[10.0, 20.0, 30.0, 14.0, 24.0, 34.0]], dtype=np.float32)),
        labels,
    )
    pipeline.postprocessing = lambda data: {
        "pred_box": FakeTorchTensor(np.array([[100.0, 200.0, 300.0, 4.0, 4.0, 4.0]], dtype=np.float32)),
        "pred_label": FakeTorchTensor(np.array([0], dtype=np.int64)),
        "pred_score": FakeTorchTensor(np.array([0.8], dtype=np.float32)),
    }

    item = {"image": FakeTensor(np.zeros((1, 8, 8, 8), dtype=np.float32))}
    candidates = MONAITutorialDetectionPipeline._detect_candidates(pipeline, item)

    assert len(candidates) == 1
    assert np.allclose(candidates[0]["centre_zyx"], np.array([32.0, 22.0, 12.0], dtype=np.float32))


def test_clip_raw_boxes_to_image_passes_one_dummy_label_per_box() -> None:
    pipeline = object.__new__(MONAITutorialDetectionPipeline)

    captured: dict[str, np.ndarray] = {}

    def fake_clipper(boxes, labels, spatial_size):
        captured["boxes"] = np.asarray(boxes)
        captured["labels"] = np.asarray(labels)
        captured["spatial_size"] = np.asarray(spatial_size)
        return boxes, labels

    class FakeTensor:
        def __init__(self, array):
            self.shape = array.shape

    pipeline.raw_box_clipper = fake_clipper
    item = {"image": FakeTensor(np.zeros((1, 8, 8, 8), dtype=np.float32))}
    raw_boxes = np.array(
        [
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0],
            [3.0, 4.0, 5.0, 7.0, 8.0, 9.0],
            [4.0, 5.0, 6.0, 8.0, 9.0, 10.0],
        ],
        dtype=np.float32,
    )

    clipped = MONAITutorialDetectionPipeline._clip_raw_boxes_to_image(pipeline, item, raw_boxes)

    assert clipped.shape == (4, 6)
    assert np.array_equal(captured["boxes"], raw_boxes)
    assert np.array_equal(captured["labels"], np.zeros(4, dtype=np.int64))


def test_clip_raw_boxes_to_image_normalizes_single_box_shape() -> None:
    pipeline = object.__new__(MONAITutorialDetectionPipeline)

    def fake_clipper(boxes, labels, spatial_size):
        return boxes, labels

    class FakeTensor:
        def __init__(self, array):
            self.shape = array.shape

    pipeline.raw_box_clipper = fake_clipper
    item = {"image": FakeTensor(np.zeros((1, 8, 8, 8), dtype=np.float32))}

    clipped = MONAITutorialDetectionPipeline._clip_raw_boxes_to_image(
        pipeline,
        item,
        np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0], dtype=np.float32),
    )

    assert clipped.shape == (1, 6)


def test_detect_candidates_unwraps_meta_tensor_before_detector_call() -> None:
    pipeline = object.__new__(MONAITutorialDetectionPipeline)
    pipeline.device = "cpu"
    pipeline.amp = False
    pipeline.val_patch_size = [512, 512, 208]
    pipeline.raw_box_clipper = lambda boxes, labels, spatial_size: (boxes, labels)

    captured: dict[str, object] = {}

    class FakeDetector:
        target_box_key = "box"
        target_label_key = "label"
        pred_score_key = "score"

        def __call__(self, inputs, use_inferer=False):
            captured["input_type"] = type(inputs[0])
            return [{
                "box": FakeTorchTensor(np.array([[1.0, 2.0, 3.0, 5.0, 6.0, 7.0]], dtype=np.float32)),
                "label": FakeTorchTensor(np.array([0], dtype=np.int64)),
                "score": FakeTorchTensor(np.array([0.8], dtype=np.float32)),
            }]

    class FakeMetaTensor:
        def __init__(self, array):
            self.tensor = torch.from_numpy(array)
            self.shape = array.shape

        def as_tensor(self):
            return self.tensor

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

    item = {"image": FakeMetaTensor(np.zeros((1, 8, 8, 8), dtype=np.float32))}
    MONAITutorialDetectionPipeline._detect_candidates(pipeline, item)

    assert captured["input_type"] is torch.Tensor


def test_build_detector_matches_tutorial_anchor_scale_count(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeAnchorGenerator:
        def __init__(self, feature_map_scales, base_anchor_shapes):
            captured["feature_map_scales"] = list(feature_map_scales)
            captured["base_anchor_shapes"] = base_anchor_shapes

    class FakeNetwork:
        def to(self, device):
            captured["network_device"] = device
            return self

    class FakeDetector:
        target_box_key = "box"
        target_label_key = "label"
        pred_score_key = "score"

        def __init__(self, network, anchor_generator, debug=False):
            captured["detector_network"] = network
            captured["anchor_generator"] = anchor_generator

        def set_box_selector_parameters(self, **kwargs):
            captured["selector_kwargs"] = kwargs

        def set_sliding_window_inferer(self, **kwargs):
            captured["inferer_kwargs"] = kwargs

        def eval(self):
            captured["eval_called"] = True
            return self

    monkeypatch.setattr(monai_tutorial_mod, "AnchorGeneratorWithAnchorShape", FakeAnchorGenerator)
    monkeypatch.setattr(monai_tutorial_mod, "RetinaNetDetector", FakeDetector)
    monkeypatch.setattr(monai_tutorial_mod.torch.jit, "load", lambda *args, **kwargs: FakeNetwork())

    pipeline = object.__new__(MONAITutorialDetectionPipeline)
    pipeline.model_path = tmp_path / "detector.pt"
    pipeline.device = "cpu"
    pipeline.returned_layers = [2, 5]
    pipeline.base_anchor_shapes = [[6, 8, 4]]
    pipeline.score_thresh = 0.02
    pipeline.nms_thresh = 0.22
    pipeline.val_patch_size = [512, 512, 208]

    detector = MONAITutorialDetectionPipeline._build_detector(pipeline)

    assert isinstance(detector, FakeDetector)
    assert captured["feature_map_scales"] == [1, 2, 4]
