"""PyTorch Lightning training and evaluation wrapper for six-class Fashion-MNIST."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy


class LightningModel(pl.LightningModule):
    def __init__(self, model, lr=0.001, model_type=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.model_type = model_type
        self.train_accuracy = Accuracy(task="multiclass", num_classes=6)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=6)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=6)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch):
        images, labels = batch
        logits = self(images)
        return F.cross_entropy(logits, labels), logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch)
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch)
        self.val_accuracy(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch)
        self.test_accuracy(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)

    def compute_class_accuracy(self, loader):
        """Compute accuracy per label on the current model device."""
        was_training = self.training
        self.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                logits = self(images.to(self.device))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                labels.extend(targets.cpu().tolist())
        self.train(was_training)
        predictions, labels = np.asarray(predictions), np.asarray(labels)
        return {
            cls: float((predictions[labels == cls] == cls).mean())
            for cls in range(6) if (labels == cls).any()
        }
