from __future__ import annotations

import torch

from src.inference.monai_bundle import _materialize_bundle_components


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
        "postprocessing",
        "inferer",
        "detector_ops",
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
