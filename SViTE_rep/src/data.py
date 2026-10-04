"""ImageNet-1k image transforms for the first, 224-pixel training stage."""

from pathlib import Path
import csv

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.imagenet_info import ImageNetInfo
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


def build_transform(training: bool, image_size: int = 224):
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    common = dict(
        input_size=image_size,
        is_training=training,
        interpolation="bicubic",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    if training:
        return create_transform(
            **common,
            scale=(0.08, 1.0),
            ratio=(3 / 4, 4 / 3),
            hflip=0.5,
            color_jitter=0.3,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
        )
    return create_transform(**common, crop_pct=0.875)


class FlatImageNetValidation(Dataset):
    """Read Kaggle's flat validation directory using LOC_val_solution.csv.

    Synset strings are mapped through the training set's class mapping; numeric
    ILSVRC IDs must not be mistaken for ImageFolder's sorted class indices.
    """

    def __init__(self, root, labels_csv, class_to_idx, transform):
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform
        self.samples = []
        seen = set()
        with Path(labels_csv).open(newline="") as stream:
            for row in csv.DictReader(stream):
                image_id = row["ImageId"]
                annotations = row["PredictionString"].split()
                if not annotations or len(annotations) % 5:
                    raise ValueError(f"Invalid localization annotation for {image_id}.")
                synsets = set(annotations[::5])
                if len(synsets) != 1:
                    raise ValueError(f"Expected a single classification label for {image_id}.")
                if image_id in seen or Path(image_id).name != image_id:
                    raise ValueError(f"Invalid or duplicate validation image ID: {image_id}.")
                seen.add(image_id)
                path = Path(root) / f"{image_id}.JPEG"
                if not path.is_file():
                    raise FileNotFoundError(path)
                self.samples.append((path, self.class_to_idx[synsets.pop()]))
        if not self.samples:
            raise ValueError("Validation label CSV contains no images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(default_loader(path)), label


def build_imagenet_datasets(root: str | Path, image_size: int = 224, val_labels: str | Path | None = None):
    """Read root/{train,val}/<ImageNet synset>/*.JPEG, without downloading data.

    Use the standard 1,000 ILSVRC2012 synset directory names. Their sorted
    order must match the pretrained teacher's class order. Validation images
    must already be arranged into these class directories.
    """
    root = Path(root)
    train = ImageFolder(root / "train", transform=build_transform(True, image_size))
    if len(train.classes) != 1000:
        raise ValueError("The pretrained teacher requires all 1,000 ImageNet-1k classes.")
    expected_mapping = {name: index for index, name in enumerate(ImageNetInfo().label_names())}
    if train.class_to_idx != expected_mapping:
        raise ValueError("Training directories must use the canonical ImageNet-1k synset names and ordering.")
    if val_labels:
        val = FlatImageNetValidation(root / "val", val_labels, train.class_to_idx, build_transform(False, image_size))
    else:
        val = ImageFolder(root / "val", transform=build_transform(False, image_size))
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("Training and validation class mappings must match.")
    return train, val
