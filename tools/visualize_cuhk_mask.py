import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import skimage.io as io
from scipy import ndimage

from sgm.data.cuhk.imgproc import imresize


def read_float_image(path):
    return io.imread(str(path)).astype(np.float32)


def ensure_3d(image):
    if image.ndim == 2:
        return image[..., None]
    return image


def read_rgbnir(rgb_root, nir_root, split, name, nir_name=None, kind="label"):
    image = read_float_image(rgb_root / split / kind / name)
    image = ensure_3d(image)

    if nir_root is None:
        return image

    if nir_name is None:
        nir_name = name
    nir = read_float_image(nir_root / split / kind / nir_name)
    if nir.ndim == 3:
        nir = nir[..., 0]
    return np.concatenate([image, nir[..., None]], axis=2)


def current_mask(label, cloudy, gain=4.0):
    # Mirrors sgm/data/cuhk/image_datasets.py: abs-diff mean with gain so cloud
    # shadows are covered too.
    return np.clip(np.abs(cloudy - label).mean(axis=2) / 255.0 * gain, 0.0, 1.0).astype(np.float32)


def abs_diff_reference(label, cloudy, percentile=95.0):
    diff = np.abs(label - cloudy).sum(axis=2)
    scale = np.percentile(diff, percentile)
    return np.clip(diff / (scale + 1e-6), 0.0, 1.0).astype(np.float32)


def feather_mask(mask, kernel):
    if kernel <= 1:
        return mask
    return ndimage.uniform_filter(mask, size=kernel, mode="reflect")


def to_uint8_rgb(image):
    rgb = image[..., :3]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def gray_to_rgb(mask):
    gray = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def overlay_mask(rgb, mask, color=(255, 0, 0), alpha=0.45):
    rgb = rgb.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    mask = np.clip(mask[..., None], 0.0, 1.0)
    out = rgb * (1.0 - alpha * mask) + color_arr * (alpha * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_title(panel, title):
    out = panel.copy()
    h, w = out.shape[:2]
    bar_h = max(24, h // 12)
    titled = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    titled[bar_h:] = out
    cv2.putText(
        titled,
        title,
        (6, max(17, bar_h - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return titled


def make_grid(panels, titles, pad=6):
    titled = [add_title(panel, title) for panel, title in zip(panels, titles)]
    h = max(panel.shape[0] for panel in titled)
    normalized = []
    for panel in titled:
        if panel.shape[0] < h:
            extra = np.zeros((h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.concatenate([panel, extra], axis=0)
        normalized.append(panel)
    spacer = np.full((h, pad, 3), 255, dtype=np.uint8)
    grid = []
    for idx, panel in enumerate(normalized):
        if idx:
            grid.append(spacer)
        grid.append(panel)
    return np.concatenate(grid, axis=1)


def parse_indices(indices):
    if not indices:
        return None
    out = []
    for item in indices.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize CUHK cloud masks used by the current dataloader.")
    parser.add_argument("--datasets-dir", required=True, help="CUHK RGB dataset root, e.g. /path/CUHK-CR1")
    parser.add_argument("--nir-datasets-dir", default=None, help="CUHK NIR dataset root, e.g. /path/nir/CUHK-CR1")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--out-dir", default="outputs/mask_vis")
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma-separated sample indices; overrides --start/--num")
    parser.add_argument("--feather-kernel", type=int, default=0)
    parser.add_argument("--resize-scale", type=float, default=0.5)
    parser.add_argument("--mask-gain", type=float, default=4.0, help="Gain k in M = clip(|cloudy-label|.mean/255*k, 0, 1)")
    args = parser.parse_args()

    rgb_root = Path(args.datasets_dir)
    nir_root = Path(args.nir_datasets_dir) if args.nir_datasets_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_dir = rgb_root / args.split / "label"
    names = sorted(os.listdir(label_dir))
    if nir_root is not None:
        nir_names = sorted(os.listdir(nir_root / args.split / "label"))
        if len(names) != len(nir_names):
            raise ValueError(f"RGB/NIR sample counts differ: {len(names)} vs {len(nir_names)}")
    else:
        nir_names = [None] * len(names)

    indices = parse_indices(args.indices)
    if indices is None:
        indices = list(range(args.start, min(args.start + args.num, len(names))))

    for idx in indices:
        name = names[idx]
        nir_name = nir_names[idx]
        label = read_rgbnir(rgb_root, nir_root, args.split, name, nir_name, kind="label")
        cloudy = read_rgbnir(rgb_root, nir_root, args.split, name, nir_name, kind="cloud")

        if args.resize_scale != 1.0:
            label = imresize(label, args.resize_scale)
            cloudy = imresize(cloudy, args.resize_scale)

        mask = current_mask(label, cloudy, gain=args.mask_gain)
        mask_feather = feather_mask(mask, args.feather_kernel)
        abs_ref = abs_diff_reference(label, cloudy)

        cloudy_rgb = to_uint8_rgb(cloudy)
        label_rgb = to_uint8_rgb(label)
        overlay = overlay_mask(cloudy_rgb, mask_feather)

        panels = [
            cloudy_rgb,
            label_rgb,
            gray_to_rgb(mask),
            gray_to_rgb(mask_feather),
            overlay,
            gray_to_rgb(abs_ref),
        ]
        titles = [
            "cloudy RGB",
            "label RGB",
            "current M",
            f"feather M k={args.feather_kernel}",
            "M overlay",
            "abs-diff ref",
        ]
        grid = make_grid(panels, titles)

        stem = Path(name).stem
        out_path = out_dir / f"{idx:05d}_{stem}_mask_vis.png"
        io.imsave(str(out_path), grid)
        print(out_path)


if __name__ == "__main__":
    main()
