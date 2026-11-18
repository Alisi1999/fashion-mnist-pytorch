"""Six-class Fashion-MNIST dataset used in the original coursework project."""

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class FMNIST(Dataset):
    """Subset of Fashion-MNIST with labels remapped to six consecutive classes.

    Image rotations are applied per sample, so keep validation augmentation disabled
    to compare experiments fairly.
    """

    selected_classes = (
        "T-shirt/top", "Trouser", "Pullover", "Sneaker", "Bag", "Ankle Boot"
    )
    all_classes = (
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
    )

    def __init__(
        self, train=True, rotate_train=False, rotate_test=False,
        use_augmentations=False, data_dir="./data"
    ):
        self.rotate = rotate_train if train else rotate_test
        self.use_augmentations = use_augmentations
        selected_indices = [self.all_classes.index(cls) for cls in self.selected_classes]

        original = datasets.FashionMNIST(root=data_dir, train=train, download=True)
        original_targets = original.targets.numpy()
        mask = np.isin(original_targets, selected_indices)
        self.data = original.data[mask]
        remap = {original_label: i for i, original_label in enumerate(selected_indices)}
        self.targets = torch.tensor(
            [remap[int(label)] for label in original_targets[mask]], dtype=torch.long
        )
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if use_augmentations else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx].numpy())
        if self.augmentations is not None:
            image = self.augmentations(image)
        image = self.normalize(image)
        if self.rotate:
            image = transforms.functional.rotate(image, random.choice((0, 90, 180, 270)))
        return image, self.targets[idx]

    @property
    def label_dict(self):
        return dict(enumerate(self.selected_classes))
