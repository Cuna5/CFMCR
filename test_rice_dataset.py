"""Smoke tests for the local RICE1/RICE2 data contract."""

from pathlib import Path

import numpy as np

from sgm.data.rice.image_datasets import RICEDataset


DATA_ROOT = Path(r"C:\Users\47651\OneDrive\Data\CODE\数据集")


def check_dataset(name, expected_count):
    root = DATA_ROOT / name
    datasets = {
        split: RICEDataset(root, split=split, image_size=256)
        for split in ("train", "validation", "test")
    }
    counts = {split: len(dataset) for split, dataset in datasets.items()}
    assert sum(counts.values()) == expected_count
    assert len(set().union(*(set(ds.names) for ds in datasets.values()))) == expected_count
    assert not (set(datasets["train"].names) & set(datasets["test"].names))

    sample = datasets["train"][0]
    assert sample["cond_image"].shape == (3, 256, 256)
    assert sample["label"].shape == (3, 256, 256)
    assert sample["aux_cond"].shape == (5, 256, 256)
    assert sample["M"].shape == (256, 256)
    assert np.isfinite(sample["cond_image"]).all()
    assert 0.0 <= float(sample["M"].min()) <= float(sample["M"].max()) <= 1.0
    print(f"[OK] {name}: splits={counts}, sample={sample['image_path']}")


if __name__ == "__main__":
    check_dataset("RICE1", 500)
    check_dataset("RICE2", 736)
    print("all RICE dataset smoke tests passed")
