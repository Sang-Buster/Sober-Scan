"""``LOSOTrainedCNN``: the intoxication CNN, trained correctly per LOSO fold.

This is the apples-to-apples comparison the report set up. Same
backbone (MobileNetV2) and same general architecture as the shipped
``intoxication_cnn.pt``, but with the methodology fixes the original
training pipeline lacked:

- **RGB input, no first-layer surgery.** The shipped model defaulted
  to ``infrared_mode=True`` and replaced MobileNetV2's first conv with
  a randomly-initialised single-channel one, destroying the
  pretrained features. We use 3-channel RGB so the backbone's
  ImageNet weights actually transfer.
- **Face crops with margin.** Background and lighting were a major
  confound in the previous training corpus; we crop to the detected
  face plus a 20% margin so the model has to look at the face.
- **Subject-disjoint per-fold training.** A fresh CNN is trained on
  the LOSO train fold and evaluated on the held-out subject. No
  random_split, no double-split bug.
- **Augmentation in tensor space** (horizontal flip, small rotations,
  light colour jitter) for regularisation.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sober_scan.corpus import IntoxicationCorpus, Photo
from sober_scan.models.baselines.imagenet import _face_crop
from sober_scan.models.cnn import IntoxicationCNN
from sober_scan.utils import load_image

_IMAGE_SIZE = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@lru_cache(maxsize=None)
def _face_tensor_for_path(path: Path) -> Optional[torch.Tensor]:
    """Load image, crop face, return a normalised RGB tensor (3, H, W).

    Cached across folds so each photo gets read + face-detected once
    per process. Returns ``None`` if the image won't load or has no face.
    """
    image = load_image(path)
    if image is None:
        return None
    crop = _face_crop(image)
    if crop is None:
        return None

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
    return preprocess(crop)


_TRAIN_AUGMENT = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    ]
)


class _PhotoTensorDataset(Dataset):
    """PyTorch Dataset wrapping (Photo, label) pairs with the face cache."""

    def __init__(self, photos: List[Photo], labels: List[int], *, augment: bool) -> None:
        self._photos = photos
        self._labels = labels
        self._augment = augment

    def __len__(self) -> int:
        return len(self._photos)

    def __getitem__(self, idx: int):
        photo = self._photos[idx]
        tensor = _face_tensor_for_path(photo.path)
        if tensor is None:
            # Black placeholder so the batch shape stays consistent. The
            # caller drops failed photos before constructing the dataset.
            tensor = torch.zeros(3, _IMAGE_SIZE, _IMAGE_SIZE)
        if self._augment:
            tensor = _TRAIN_AUGMENT(tensor)
        return tensor, torch.tensor(self._labels[idx], dtype=torch.float32)


class LOSOTrainedCNN:
    """Per-fold MobileNetV2 with frozen backbone + small classifier head."""

    def __init__(
        self,
        *,
        epochs: int = 8,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[IntoxicationCNN] = None
        self._fallback_proba: float = 0.5

    def fit(
        self, train: IntoxicationCorpus, *, threshold: float
    ) -> "LOSOTrainedCNN":
        labels = train.binary_labels(threshold=threshold)
        # Drop photos with no detectable face so the dataset doesn't carry
        # all-black placeholders into training.
        kept_photos: List[Photo] = []
        kept_labels: List[int] = []
        for photo, label in zip(train, labels):
            if _face_tensor_for_path(photo.path) is not None:
                kept_photos.append(photo)
                kept_labels.append(label)

        if not kept_photos:
            raise RuntimeError("no usable training photos for LOSOTrainedCNN")

        self._fallback_proba = sum(kept_labels) / len(kept_labels)

        dataset = _PhotoTensorDataset(kept_photos, kept_labels, augment=True)
        # drop_last=True prevents single-sample batches that crash BatchNorm.
        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, drop_last=True
        )

        self._model = IntoxicationCNN(in_channels=3).to(self._device)

        # Freeze the backbone features; train only the classifier head.
        for param in self._model.backbone.features.parameters():
            param.requires_grad_(False)
        head_params = list(self._model.backbone.classifier.parameters())

        # Class weighting for imbalance: BCELoss applied per-example.
        pos = sum(kept_labels)
        neg = len(kept_labels) - pos
        pos_weight_value = neg / pos if pos > 0 else 1.0
        criterion = nn.BCELoss(reduction="none")
        optimizer = torch.optim.AdamW(
            head_params, lr=self._learning_rate, weight_decay=self._weight_decay
        )

        self._model.train()
        for _ in range(self._epochs):
            for inputs, batch_labels in loader:
                inputs = inputs.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimizer.zero_grad()
                outputs = self._model(inputs).squeeze(-1)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                per_example_loss = criterion(outputs, batch_labels)
                # Up-weight the positive class so the model can't trivially
                # collapse to predicting all-sober.
                weights = torch.where(
                    batch_labels > 0.5,
                    torch.full_like(batch_labels, pos_weight_value),
                    torch.ones_like(batch_labels),
                )
                loss = (per_example_loss * weights).mean()
                loss.backward()
                optimizer.step()

        self._model.eval()
        return self

    def predict_proba(self, photo: Photo) -> float:
        if self._model is None:
            raise RuntimeError("LOSOTrainedCNN must be fit before predicting")

        tensor = _face_tensor_for_path(photo.path)
        if tensor is None:
            return self._fallback_proba

        with torch.no_grad():
            output = self._model(tensor.unsqueeze(0).to(self._device)).item()
        return float(output)
