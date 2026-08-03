"""Paired RGB loader for the RICE1 and RICE2 cloud-removal datasets."""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from scipy import ndimage
from torch.utils import data


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _natural_key(name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


class RICEDataset(data.Dataset):
    """Load paired RICE RGB images with a deterministic dataset split.

    The public RICE folders do not define train/validation/test lists. This
    loader creates a reproducible split from paired filenames. RICE2 masks are
    color-coded, so every non-black pixel is treated as cloud-affected by
    default. RICE1 has no mask and uses the paired absolute RGB difference.
    """

    def __init__(
        self,
        datasets_dir,
        split="train",
        split_ratios=(0.8, 0.1, 0.1),
        split_seed=3407,
        image_size=256,
        augment=False,
        hflip_p=0.5,
        vflip_p=0.5,
        rot90_p=0.5,
        aux_mode="full",
        mask_mode="auto",
        mask_gain=4.0,
    ):
        super().__init__()
        if split not in ("train", "validation", "test", "predict", "all"):
            raise ValueError(f"unsupported RICE split: {split}")
        if aux_mode not in ("full", "luma_only"):
            raise ValueError(
                f"unsupported aux_mode {aux_mode}; expected 'full' or "
                "'luma_only'"
            )
        if mask_mode not in ("auto", "provided_nonzero", "difference"):
            raise ValueError(
                f"unsupported mask_mode {mask_mode}; expected 'auto', "
                "'provided_nonzero', or 'difference'"
            )

        ratios = tuple(float(value) for value in split_ratios)
        if len(ratios) != 3 or any(value < 0.0 for value in ratios):
            raise ValueError("split_ratios must contain three non-negative values")
        ratio_sum = sum(ratios)
        if ratio_sum <= 0.0:
            raise ValueError("split_ratios must have a positive sum")
        ratios = tuple(value / ratio_sum for value in ratios)

        self.datasets_dir = Path(datasets_dir)
        self.cloud_dir = self.datasets_dir / "cloud"
        self.label_dir = self.datasets_dir / "label"
        self.mask_dir = self.datasets_dir / "mask"
        if not self.cloud_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"RICE dataset must contain cloud/ and label/: "
                f"{self.datasets_dir}"
            )

        cloud_names = self._image_names(self.cloud_dir)
        label_names = self._image_names(self.label_dir)
        if set(cloud_names) != set(label_names):
            cloud_only = sorted(set(cloud_names) - set(label_names), key=_natural_key)
            label_only = sorted(set(label_names) - set(cloud_names), key=_natural_key)
            raise ValueError(
                "RICE cloud/label filenames do not match. "
                f"cloud_only={cloud_only[:5]}, label_only={label_only[:5]}"
            )

        names = sorted(cloud_names, key=_natural_key)
        if self.mask_dir.is_dir():
            mask_names = set(self._image_names(self.mask_dir))
            missing_masks = [name for name in names if name not in mask_names]
            if missing_masks and mask_mode == "provided_nonzero":
                raise ValueError(
                    f"RICE masks are missing for {missing_masks[:5]}"
                )

        if split != "all":
            names = self._split_names(
                names,
                "test" if split == "predict" else split,
                ratios,
                int(split_seed),
            )

        self.names = names
        self.split = split
        self.image_size = int(image_size) if image_size is not None else None
        self.augment = bool(augment and split == "train")
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.rot90_p = float(rot90_p)
        self.aux_mode = aux_mode
        self.mask_mode = mask_mode
        self.mask_gain = float(mask_gain)

    @staticmethod
    def _image_names(directory):
        return [
            item.name
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS
        ]

    @staticmethod
    def _split_names(names, split, ratios, seed):
        count = len(names)
        order = np.random.RandomState(seed).permutation(count)
        train_end = int(np.floor(count * ratios[0]))
        val_end = int(np.floor(count * (ratios[0] + ratios[1])))
        bounds = {
            "train": (0, train_end),
            "validation": (train_end, val_end),
            "test": (val_end, count),
        }
        start, end = bounds[split]
        indices = sorted(order[start:end].tolist())
        return [names[index] for index in indices]

    @staticmethod
    def _read_rgb(path):
        image = io.imread(str(path))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.shape[2] > 3:
            image = image[..., :3]
        if image.shape[2] != 3:
            raise ValueError(f"expected RGB image at {path}, got {image.shape}")
        return image.astype(np.float32)

    @staticmethod
    def _read_mask(path):
        mask = io.imread(str(path))
        if mask.ndim == 2:
            return mask > 0
        return np.any(mask[..., :3] > 0, axis=2)

    def _resize(self, image, is_mask=False):
        if self.image_size is None:
            return image
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
        return cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=interpolation,
        )

    def _augment_triplet(self, label, cloudy, mask):
        if np.random.rand() < self.hflip_p:
            label = np.flip(label, axis=1)
            cloudy = np.flip(cloudy, axis=1)
            mask = np.flip(mask, axis=1)
        if np.random.rand() < self.vflip_p:
            label = np.flip(label, axis=0)
            cloudy = np.flip(cloudy, axis=0)
            mask = np.flip(mask, axis=0)
        if np.random.rand() < self.rot90_p:
            k = np.random.randint(1, 4)
            label = np.rot90(label, k, axes=(0, 1))
            cloudy = np.rot90(cloudy, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
        return tuple(
            np.ascontiguousarray(value) for value in (label, cloudy, mask)
        )

    @staticmethod
    def _sobel_edges(image):
        edge_x = ndimage.sobel(image, axis=1, mode="reflect")
        edge_y = ndimage.sobel(image, axis=0, mode="reflect")
        return np.sqrt(edge_x * edge_x + edge_y * edge_y)

    @classmethod
    def _build_aux_cond(cls, cloudy, aux_mode):
        rgb = (np.clip(cloudy, -1.0, 1.0) + 1.0) * 0.5
        luma = (
            0.2989 * rgb[..., 0:1]
            + 0.5870 * rgb[..., 1:2]
            + 0.1140 * rgb[..., 2:3]
        )
        if aux_mode == "full":
            edge_rgb = cls._sobel_edges(rgb)
            edge_luma = cls._sobel_edges(luma)
        else:
            edge_rgb = np.zeros_like(rgb)
            edge_luma = np.zeros_like(luma)
        aux = np.concatenate([luma, edge_rgb, edge_luma], axis=2)
        return aux.astype(np.float32).transpose(2, 0, 1)

    def _cloud_mask(self, name, cloudy, label):
        use_provided = (
            self.mask_mode in ("auto", "provided_nonzero")
            and (self.mask_dir / name).is_file()
        )
        if use_provided:
            mask = self._read_mask(self.mask_dir / name).astype(np.float32)
            return self._resize(mask, is_mask=True).astype(np.float32)
        if self.mask_mode == "provided_nonzero":
            raise FileNotFoundError(f"missing RICE mask: {self.mask_dir / name}")
        diff = np.abs(cloudy - label).mean(axis=2) / 255.0
        return np.clip(diff * self.mask_gain, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, index):
        name = self.names[index]
        label = self._read_rgb(self.label_dir / name)
        cloudy = self._read_rgb(self.cloud_dir / name)
        if label.shape != cloudy.shape:
            raise ValueError(
                f"paired RICE shapes differ for {name}: "
                f"{label.shape} vs {cloudy.shape}"
            )

        label = self._resize(label)
        cloudy = self._resize(cloudy)
        mask = self._cloud_mask(name, cloudy, label)
        if self.augment:
            label, cloudy, mask = self._augment_triplet(label, cloudy, mask)

        label = label / 127.5 - 1.0
        cloudy = cloudy / 127.5 - 1.0
        aux_cond = self._build_aux_cond(cloudy, self.aux_mode)
        label = label.astype(np.float32).transpose(2, 0, 1)
        cloudy = cloudy.astype(np.float32).transpose(2, 0, 1)

        return {
            "cloudy": cloudy,
            "cond_image": cloudy,
            "aux_cond": aux_cond,
            "label": label,
            "M": mask.astype(np.float32),
            "image_path": name,
        }

    def __len__(self):
        return len(self.names)
