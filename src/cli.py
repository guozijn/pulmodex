"""Unified command-line entry point for Pulmodex."""

from __future__ import annotations

import argparse
import logging
import runpy
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent


def _build_commands() -> dict[str, tuple[Path, str]]:
    return {
        "train": (ROOT / "src" / "train.py", "Train a model with Hydra overrides."),
        "evaluate": (ROOT / "src" / "evaluate.py", "Evaluate a checkpoint."),
        "infer": (ROOT / "src" / "inference.py", "Run inference on one or more scans."),
        "export-onnx": (
            ROOT / "scripts" / "export_onnx.py",
            "Export a checkpoint to ONNX.",
        ),
        "generate-mock-data": (
            ROOT / "scripts" / "generate_mock_luna16.py",
            "Generate a tiny LUNA16-style mock dataset.",
        ),
        "preprocess-cache": (
            ROOT / "scripts" / "preprocess_luna16_cache.py",
            "Precompute cached isotropic, normalized LUNA16 volumes.",
        ),
        "dicom-to-luna16": (
            ROOT / "scripts" / "dicom_to_luna16.py",
            "Convert raw DICOM studies to LUNA16-compatible data.",
        ),
    }


def _build_command_groups() -> dict[str, dict[str, tuple[Path, str]]]:
    return {
        "detect": {
            "prepare": (
                ROOT / "src" / "detect_prepare.py",
                "Prepare local LUNA16 splits for MONAI 3D detection.",
            ),
            "standardize": (
                ROOT / "src" / "detect_standardize.py",
                "Standardize raw MHD or DICOM inputs into NIfTI volumes.",
            ),
            "train": (
                ROOT / "src" / "detect_train.py",
                "Train a MONAI 3D RetinaNet detector.",
            ),
            "infer": (
                ROOT / "src" / "detect_infer.py",
                "Run MONAI 3D detection inference.",
            ),
            "evaluate": (
                ROOT / "src" / "detect_evaluate.py",
                "Run LUNA16 evaluation for a MONAI 3D detector.",
            ),
        }
    }


def _print_help(
    parser: argparse.ArgumentParser,
    commands: dict[str, tuple[Path, str]],
    groups: dict[str, dict[str, tuple[Path, str]]],
) -> None:
    parser.print_help()
    print("\nCommands:")
    for name, (_, help_text) in commands.items():
        print(f"  {name:<18} {help_text}")
    print("\nGrouped Commands:")
    for group_name, subcommands in groups.items():
        print(f"  {group_name}")
        for sub_name, (_, help_text) in subcommands.items():
            print(f"    {sub_name:<14} {help_text}")


def _resolve_command(
    command: str | None,
    remainder: list[str],
    commands: dict[str, tuple[Path, str]],
    groups: dict[str, dict[str, tuple[Path, str]]],
) -> tuple[str, Path, list[str]]:
    if command is None:
        raise ValueError("No command provided.")

    if command in groups:
        if not remainder:
            choices = ", ".join(sorted(groups[command]))
            raise ValueError(f"`{command}` requires a subcommand: {choices}")
        subcommand, *sub_args = remainder
        if subcommand not in groups[command]:
            choices = ", ".join(sorted(groups[command]))
            raise ValueError(f"Unknown `{command}` subcommand `{subcommand}`. Choices: {choices}")
        script_path, _ = groups[command][subcommand]
        return f"{command} {subcommand}", script_path, sub_args

    if command in commands:
        script_path, _ = commands[command]
        return command, script_path, remainder

    all_commands = sorted(list(commands) + list(groups))
    raise ValueError(f"Unknown command `{command}`. Choices: {', '.join(all_commands)}")


def main() -> None:
    commands = _build_commands()
    groups = _build_command_groups()

    parser = argparse.ArgumentParser(
        prog="pulmodex",
        description="Unified CLI for training, evaluation, inference, and data tooling.",
    )
    parser.add_argument("command", nargs="?")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    if parsed.command is None:
        _print_help(parser, commands, groups)
        raise SystemExit(1)

    try:
        display_name, script_path, forwarded_args = _resolve_command(
            parsed.command,
            parsed.args,
            commands,
            groups,
        )
    except ValueError as exc:
        parser.error(str(exc))

    sys.argv = [f"{parser.prog} {display_name}", *forwarded_args]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
