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


def main() -> None:
    commands = _build_commands()

    parser = argparse.ArgumentParser(
        prog="pulmodex",
        description="Unified CLI for training, evaluation, inference, and data tooling.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(commands))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    if parsed.command is None:
        parser.print_help()
        print("\nCommands:")
        for name, (_, help_text) in commands.items():
            print(f"  {name:<18} {help_text}")
        raise SystemExit(1)

    script_path, _ = commands[parsed.command]
    sys.argv = [f"{parser.prog} {parsed.command}", *parsed.args]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
