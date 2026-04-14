from __future__ import annotations

from src.webapp import tasks


def test_resolve_device_keeps_cpu() -> None:
    assert tasks._resolve_device("cpu") == "cpu"


def test_resolve_device_falls_back_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert tasks._resolve_device("cuda") == "cpu"
    assert tasks._resolve_device("cuda:0") == "cpu"
