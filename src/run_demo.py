"""End to end demo: pretrain on MedMNIST, then linear probe the encoder.

Run with the project root on the path, for example:

    python -m src.run_demo --flag bloodmnist --pretrain-size 2000 --epochs 5

The script prints the pretraining loss history and compares the linear probe
accuracy of the pretrained encoder against a random encoder baseline.
"""

from __future__ import annotations

import argparse

import torch

from .data import load_medmnist
from .pretrain import pretrain_simclr
from .probe import linear_probe_accuracy, random_encoder_like


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flag", default="bloodmnist", help="MedMNIST dataset key")
    parser.add_argument("--pretrain-size", type=int, default=2000)
    parser.add_argument("--probe-train-size", type=int, default=500)
    parser.add_argument("--probe-test-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading MedMNIST '{args.flag}' ...")
    pre_images, _ = load_medmnist(args.flag, split="train", limit=args.pretrain_size)
    tr_images, tr_labels = load_medmnist(
        args.flag, split="train", limit=args.probe_train_size
    )
    te_images, te_labels = load_medmnist(
        args.flag, split="test", limit=args.probe_test_size
    )

    print(f"Pretraining SimCLR on {pre_images.shape[0]} unlabeled images ...")
    result = pretrain_simclr(
        pre_images, epochs=args.epochs, batch_size=args.batch_size
    )
    print("Pretraining loss per epoch:")
    for i, value in enumerate(result.loss_history):
        print(f"  epoch {i + 1}: {value:.4f}")

    pretrained_encoder = result.model.encoder
    random_encoder = random_encoder_like(pretrained_encoder)

    pretrained_acc = linear_probe_accuracy(
        pretrained_encoder, tr_images, tr_labels, te_images, te_labels
    )
    random_acc = linear_probe_accuracy(
        random_encoder, tr_images, tr_labels, te_images, te_labels
    )

    print(f"Linear probe accuracy (pretrained encoder): {pretrained_acc:.4f}")
    print(f"Linear probe accuracy (random encoder):     {random_acc:.4f}")
    print(f"Improvement: {pretrained_acc - random_acc:+.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
