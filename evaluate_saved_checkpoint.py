"""Evaluate an original saved CNN checkpoint on the six-class test subset.

Usage:
    python evaluate_saved_checkpoint.py --checkpoint path/to/original.ckpt

The checkpoint isn't included in this repository. Use a checkpoint you own and
trust. Checkpoint loading uses weights_only=True; PyTorch 2.2+ is recommended.
"""
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from fmnist import FMNIST
from networks import CNN


def evaluate_checkpoint(path, batch_size=256):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint:
        raise ValueError("Expected a PyTorch Lightning checkpoint with state_dict")
    state = {
        key.removeprefix("model."): tensor
        for key, tensor in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    if not state:
        raise ValueError("Checkpoint contains no CNN weights under 'model.'")
    model = CNN()
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = FMNIST(train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = defaultdict(int)
    correct = defaultdict(int)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            for cls in range(6):
                selected = (labels == cls)
                totals[cls] += int(selected.sum())
                correct[cls] += int((predictions[selected] == labels[selected]).sum())

    all_correct = sum(correct.values())
    all_total = sum(totals.values())
    print(f"Overall test accuracy: {all_correct}/{all_total} = {all_correct/all_total:.4%}")
    for cls, name in enumerate(FMNIST.selected_classes):
        print(f"{name:15s} {correct[cls]:4d}/{totals[cls]:4d} = "
              f"{correct[cls]/totals[cls]:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to trusted original checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.batch_size)
