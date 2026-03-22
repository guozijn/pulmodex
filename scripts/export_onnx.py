"""Export a trained checkpoint to ONNX.

Usage:
    python scripts/export_onnx.py \
        --checkpoint checkpoints/best.ckpt \
        --output checkpoints/model.onnx \
        [--model hybrid_net|unet3d|fp_classifier] \
        [--patch_size 256]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

log = logging.getLogger(__name__)


def load_model(checkpoint: str, model_name: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)

    if model_name == "unet3d":
        from src.models.baseline import UNet3D
        model = UNet3D()
    elif model_name == "hybrid_net":
        from src.models.hybrid import HybridNet
        model = HybridNet()
    elif model_name == "fp_classifier":
        from src.fp_reduction import FPClassifier
        model = FPClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


class SegModelWrapper(torch.nn.Module):
    """Wrapper that returns only 'seg' (sigmoid mask) for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["seg"]


class FPModelWrapper(torch.nn.Module):
    """Wrapper that returns probability scalar for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["prob"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/model.onnx")
    parser.add_argument("--model", default="unet3d",
                        choices=["unet3d", "hybrid_net", "fp_classifier"])
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Spatial side length for dummy input")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    model = load_model(args.checkpoint, args.model, args.device)

    if args.model == "fp_classifier":
        wrapper = FPModelWrapper(model)
        ps = 32
    else:
        wrapper = SegModelWrapper(model)
        ps = args.patch_size

    dummy = torch.randn(1, 1, ps, ps, ps, device=args.device)

    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Exporting {args.model} to {output_path} (input {dummy.shape})")
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["ct_patch"],
        output_names=["output"],
        dynamic_axes={"ct_patch": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    log.info("ONNX model check passed.")

    # Quick ORT inference test
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    result = sess.run(None, {"ct_patch": dummy.cpu().numpy()})
    log.info(f"ORT output shape: {result[0].shape}")

    log.info(f"Exported to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
