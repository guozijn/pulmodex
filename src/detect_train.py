"""Train the MONAI 3D detection model."""

from __future__ import annotations

import argparse
import logging

from src.detection import train_detection_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MONAI 3D RetinaNet detector on LUNA16.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--prepared_dir", default="data/monai_detection")
    parser.add_argument("--checkpoint", default="checkpoints/monai_detection_fold0.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--target_spacing", type=float, default=1.0)
    parser.add_argument("--samples_per_image", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_detection_model(
        fold=args.fold,
        prepared_dir=args.prepared_dir,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=tuple(args.patch_size),
        target_spacing=args.target_spacing,
        samples_per_image=args.samples_per_image,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
