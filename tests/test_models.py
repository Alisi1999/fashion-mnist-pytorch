"""Small smoke tests that do not download Fashion-MNIST."""

import torch

from networks import CNN, MLP
from networks_lightning import LightningModel


def test_cnn_output_shape():
    logits = CNN()(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 6)


def test_mlp_output_shape():
    logits = MLP()(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 6)


def test_lightning_training_step_returns_loss():
    model = LightningModel(CNN(), model_type="CNN")
    images, targets = torch.randn(3, 1, 28, 28), torch.tensor([0, 2, 5])
    # Do not call training_step outside Trainer: logging requires Trainer context.
    loss, logits, labels = model._step((images, targets))
    assert loss.ndim == 0
    assert logits.shape == (3, 6)
    assert torch.equal(labels, targets)
