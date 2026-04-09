from src.cli import _build_command_groups, _build_commands, _resolve_command


def test_resolve_top_level_train_command() -> None:
    display, script, args = _resolve_command(
        "train",
        ["--config-name", "baseline"],
        _build_commands(),
        _build_command_groups(),
    )

    assert display == "train"
    assert args == ["--config-name", "baseline"]
