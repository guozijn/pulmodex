from __future__ import annotations

import torch

from src.inference.monai_bundle import _materialize_bundle_components, _resolve_bundle_paths, is_monai_bundle_path


class _FakeNetwork:
    def __init__(self) -> None:
        self.moved_to = None

    def to(self, device: torch.device) -> "_FakeNetwork":
        self.moved_to = device
        return self


class _FakeParser(dict):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.values = {
            "preprocessing": object(),
            "network_def": _FakeNetwork(),
            "detector": object(),
            "postprocessing": object(),
            "inferer": object(),
            "detector_ops": [],
        }

    def get_parsed_content(self, key: str):
        self.calls.append(key)
        if key == "network":
            raise AssertionError("network should be materialized from network_def, not parsed directly")
        return self.values[key]


def test_materialize_bundle_components_resolves_network_def_before_dependents() -> None:
    parser = _FakeParser()
    device = torch.device("cpu")

    preprocessing, network, detector, postprocessing, inferer = _materialize_bundle_components(parser, device)

    assert parser.calls == [
        "preprocessing",
        "network_def",
        "detector",
        "detector_ops",
        "inferer",
        "postprocessing",
    ]
    assert preprocessing is parser.values["preprocessing"]
    assert network is parser.values["network_def"]
    assert network.moved_to == device
    assert parser["network_def"] is network
    assert parser["network"] is network
    assert detector is parser.values["detector"]
    assert parser["detector"] is detector
    assert postprocessing is parser.values["postprocessing"]
    assert inferer is parser.values["inferer"]


def test_is_monai_bundle_path_accepts_bundle_directory(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "configs").mkdir(parents=True)
    (bundle_dir / "configs" / "inference.json").write_text("{}")

    assert is_monai_bundle_path(bundle_dir) is True


def test_is_monai_bundle_path_rejects_model_file_in_models_directory(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "configs").mkdir(parents=True)
    (bundle_dir / "models").mkdir(parents=True)
    (bundle_dir / "configs" / "inference.json").write_text("{}")
    model_path = bundle_dir / "models" / "model.pt"
    model_path.write_bytes(b"weights")

    assert is_monai_bundle_path(model_path) is False


def test_resolve_bundle_paths_rejects_model_file(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "configs").mkdir(parents=True)
    (bundle_dir / "models").mkdir(parents=True)
    (bundle_dir / "configs" / "inference.json").write_text("{}")
    model_path = bundle_dir / "models" / "model.pt"
    model_path.write_bytes(b"weights")

    try:
        _resolve_bundle_paths(model_path)
    except ValueError as exc:
        assert "Not a MONAI bundle path" in str(exc)
    else:
        raise AssertionError("Expected ValueError for direct model.pt bundle path")
