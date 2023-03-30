import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl

from .mask import (
    bbox2mask,
    brush_stroke_mask,
    get_irregular_mask,
    random_bbox,
    get_file_mask,
)


def is_image_file(filename: str):
    return any(filename.lower().endswith(extension) for extension in [".jpg", ".png"])


def make_dataset(path: str):
    if os.path.isfile(path):
        images = list(np.genfromtxt(path, dtype=np.str, encoding="utf-8"))
        masks = [os.path.splitext(image)[0] + "_mask.png" for image in images]
    else:
        images = []
        masks = []
        assert os.path.isdir(path), "%s is not a valid directory" % path
        for root, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
                    masks.append(os.path.splitext(path)[0] + "_mask.png")
    return images, masks


class InpaintingDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        mask_mode: str,
        data_len: int,
        image_size: list,
    ):
        imgs, masks = make_dataset(data_path)
        if data_len > 0:
            self.imgs = imgs[: data_len]
            self.masks = masks[: data_len]
        else:
            self.imgs = imgs
            self.masks = masks
        self.tfs = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.mask_mode = mask_mode
        self.image_size = image_size
        self.len = len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(Image.open(path).convert("RGB"))
        mask = self.get_mask(mask_path=self.masks[index])
        cond_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask
        return {
            "gt_image": img,
            "cond_image": cond_image,
            "mask_image": mask_img,
            "mask": mask,
            "path": path.rsplit("/")[-1].rsplit("\\")[-1],
        }

    def __len__(self):
        return self.len

    def get_mask(self, mask_path=None):
        if self.mask_mode == "bbox":
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == "center":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == "irregular":
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == "free_form":
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == "hybrid":
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(
                self.image_size,
            )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == "file":
            mask = get_file_mask(mask_path)
        else:
            raise NotImplementedError(
                f"Mask mode {self.mask_mode} has not been implemented."
            )
        return torch.from_numpy(mask).permute(2, 0, 1)


class CelebaHQDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_conf: dict,
        validation_split,
        dataloader_conf: dict,
        val_dataloader_conf: dict,
        test_dataset_conf: dict,
        test_dataloader_conf: dict,
    ):
        super().__init__()
        self.dataset_conf = dataset_conf
        self.validation_split = validation_split
        self.dataloader_conf = dataloader_conf
        self.val_dataloader_conf = val_dataloader_conf
        self.test_dataset_conf = test_dataset_conf
        self.test_dataloader_conf = test_dataloader_conf

    def setup(self, stage: str):
        if stage == "fit":
            celeba_full = InpaintingDataset(**self.dataset_conf)
            data_len = len(celeba_full)
            if isinstance(self.validation_split, int):
                assert (
                    self.validation_split < data_len
                ), "Validation set size is configured to be larger than entire dataset."
                valid_len = self.validation_split
            else:
                valid_len = int(data_len * self.validation_split)
            data_len -= valid_len
            self.celeba_train, self.celeba_val = random_split(
                celeba_full, [data_len, valid_len]
            )
        if stage == "test":
            self.celeba_test = InpaintingDataset(**self.test_dataset_conf)
        if stage == "predict":
            self.celeba_predict = InpaintingDataset(**self.test_dataset_conf)

    def train_dataloader(self):
        return DataLoader(self.celeba_train, **self.dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.celeba_val, **self.val_dataloader_conf)

    def test_dataloader(self):
        return DataLoader(self.celeba_test, **self.test_dataloader_conf)

    def predict_dataloader(self):
        return DataLoader(self.celeba_predict, **self.test_dataloader_conf)
