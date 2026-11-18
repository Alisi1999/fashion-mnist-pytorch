"""Train and evaluate CNN/MLP models on a six-class Fashion-MNIST subset."""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

from fmnist import FMNIST
from networks import CNN, MLP
from networks_lightning import LightningModel


def get_loaders(args):
    # Separate dataset instances share sample indices but not augmentation state.
    # Validation images remain unaugmented even when training augmentation is enabled.
    train_source = FMNIST(train=True, rotate_train=args.rotate_train,
                          use_augmentations=args.use_augmentations)
    val_source = FMNIST(train=True)
    test_source = FMNIST(train=False, rotate_test=args.rotate_test)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    split = int(0.8 * len(indices))
    train_data = Subset(train_source, indices[:split])
    val_data = Subset(val_source, indices[split:])
    return (
        DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.workers),
        DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers),
        DataLoader(test_source, batch_size=args.batch_size, num_workers=args.workers),
    )


def make_model(model_type, learning_rate):
    network = CNN() if model_type == "CNN" else MLP()
    return LightningModel(network, lr=learning_rate, model_type=model_type)


def load_model(checkpoint, model_type, learning_rate):
    if not checkpoint:
        raise ValueError("Evaluation without training requires --checkpoint PATH")
    network = CNN() if model_type == "CNN" else MLP()
    return LightningModel.load_from_checkpoint(
        checkpoint, model=network, lr=learning_rate, model_type=model_type
    )


def plot_class_accuracy(accuracy, labels, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    classes = sorted(accuracy)
    plt.bar([labels[c] for c in classes], [accuracy[c] for c in classes])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Test accuracy by class")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--analyze-errors", action="store_true")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint for evaluation-only runs")
    parser.add_argument("--model-dir", type=str, default="model_checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0, help="0 is portable on Windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotate-test", action="store_true")
    parser.add_argument("--rotate-train", action="store_true")
    parser.add_argument("--model-type", choices=["CNN", "MLP"], default="CNN")
    parser.add_argument("--use-augmentations", action="store_true")
    args = parser.parse_args()

    if not (args.train or args.evaluate or args.analyze_errors):
        parser.error("Choose --train, --evaluate, or --analyze-errors")
    if not args.train and not args.checkpoint:
        parser.error("Evaluation-only runs require --checkpoint PATH")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_loader, val_loader, test_loader = get_loaders(args)
    experiment = (f"{args.model_type}_trainrot-{int(args.rotate_train)}_"
                  f"testrot-{int(args.rotate_test)}_aug-{int(args.use_augmentations)}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.model_dir) / experiment,
        filename="best-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc", mode="max", save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_acc", mode="max", patience=5)],
        logger=TensorBoardLogger("lightning_logs", name=experiment),
        accelerator="auto", devices=1, log_every_n_steps=10,
    )

    model = make_model(args.model_type, args.lr)
    if args.train:
        trainer.fit(model, train_loader, val_loader)
        checkpoint = checkpoint_callback.best_model_path
        print(f"Best checkpoint: {checkpoint}")
        if checkpoint:
            model = load_model(checkpoint, args.model_type, args.lr)
    else:
        model = load_model(args.checkpoint, args.model_type, args.lr)

    if args.evaluate:
        results = trainer.test(model, dataloaders=test_loader)
        print("Test metrics:", results)

    if args.analyze_errors:
        accuracy = model.compute_class_accuracy(test_loader)
        print("Balanced accuracy:", float(np.mean(list(accuracy.values()))))
        print("Per-class accuracy:", accuracy)
        plot_class_accuracy(
            accuracy, FMNIST.selected_classes,
            Path("figures") / f"{experiment}_accuracy_per_class.png",
        )


if __name__ == "__main__":
    main()
