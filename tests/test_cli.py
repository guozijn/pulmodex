from pathlib import Path

from src.cli import ROOT, _build_command_groups, _build_commands, _resolve_command


def test_resolve_grouped_detect_command() -> None:
    display, script, args = _resolve_command(
        "detect",
        ["train", "--epochs", "1"],
        _build_commands(),
        _build_command_groups(),
    )

    assert display == "detect train"
    assert script == ROOT / "src" / "detect_train.py"
    assert args == ["--epochs", "1"]
