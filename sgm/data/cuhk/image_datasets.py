"""
Original work Copyright (c) 2024 littlebeen
Credits to: https://github.com/littlebeen/Cloud-removal-model-collection
Modified by Ly403 at 2025-04-01. 
"""
import numpy as np
import os
from torch.utils import data
from sgm.data.cuhk.imgproc import imresize
import skimage.io as io
from scipy import ndimage

class TrainDataset(data.Dataset):

    def __init__(
        self,
        datasets_dir,
        nir_datasets_dir=None,
        isTrain=True,
        augment=False,
        hflip_p=0.5,
        vflip_p=0.5,
        rot90_p=0.5,
    ):
        super().__init__()
        self.isTrain = isTrain
        self.augment = augment and isTrain
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.rot90_p = float(rot90_p)
        if(isTrain):
            self.datasets_dir = datasets_dir+'/train' #change to the path of your dataset
        else:
            self.datasets_dir = datasets_dir+'/test'
        if nir_datasets_dir is not None:
            if(isTrain):
                self.nir_datasets_dir = nir_datasets_dir + '/train'
            else:
                self.nir_datasets_dir = nir_datasets_dir + '/test'

        self.imlistl = sorted(os.listdir(os.path.join(self.datasets_dir, 'label')))
        self.nir_imlistl = sorted(os.listdir(os.path.join(self.nir_datasets_dir, 'label'))) if nir_datasets_dir is not None else None
        assert len(self.imlistl) == len(self.nir_imlistl), 'The number of images in the RGB dataset and NIR dataset should be the same'

    def _augment_pair(self, t, x):
        if np.random.rand() < self.hflip_p:
            t = np.flip(t, axis=1)
            x = np.flip(x, axis=1)

        if np.random.rand() < self.vflip_p:
            t = np.flip(t, axis=0)
            x = np.flip(x, axis=0)

        if np.random.rand() < self.rot90_p:
            k = np.random.randint(1, 4)
            t = np.rot90(t, k, axes=(0, 1))
            x = np.rot90(x, k, axes=(0, 1))

        return np.ascontiguousarray(t), np.ascontiguousarray(x)

    @staticmethod
    def _sobel_edges(image):
        edge_x = ndimage.sobel(image, axis=1, mode="reflect")
        edge_y = ndimage.sobel(image, axis=0, mode="reflect")
        return np.sqrt(edge_x * edge_x + edge_y * edge_y)

    @classmethod
    def _build_aux_cond(cls, cond_image):
        rgbnir = (np.clip(cond_image, -1.0, 1.0) + 1.0) * 0.5
        red = rgbnir[..., 0:1]
        rgb = rgbnir[..., 0:3]
        nir = rgbnir[..., 3:4]

        ndvi = (nir - red) / (nir + red + 1e-4)
        edge_rgb = cls._sobel_edges(rgb)
        edge_nir = cls._sobel_edges(nir)
        aux_cond = np.concatenate([ndvi, edge_rgb, edge_nir], axis=2)
        return aux_cond.astype(np.float32).transpose(2, 0, 1)

    def __getitem__(self, index):
        # a dataset contain 4 bands. it read the nir band and RGB band separately
        t = io.imread(os.path.join(self.datasets_dir, 'label', str(self.imlistl[index]))).astype(np.float32)
        x = io.imread(os.path.join(self.datasets_dir, 'cloud', str(self.imlistl[index]))).astype(np.float32)
        if self.nir_datasets_dir is not None:
            nirt = io.imread(os.path.join(self.nir_datasets_dir, 'label', str(self.nir_imlistl[index]))).astype(np.float32)[:,:,0]
            nirx = io.imread(os.path.join(self.nir_datasets_dir, 'cloud', str(self.nir_imlistl[index]))).astype(np.float32)[:,:,0]
            t =np.concatenate([t,nirt[:,:,np.newaxis]],axis=2)
            x =np.concatenate([x,nirx[:,:,np.newaxis]],axis=2)
        t = imresize(t, 1/2)
        x = imresize(x, 1/2)

        if self.augment:
            t, x = self._augment_pair(t, x)

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)
        #M = io.imread(os.path.join(self.datasets_dir, 'mask', str(self.imlistl[index]))).astype(np.float32)
        # M[M>0.5]=1
        # M[M<=0.5]=0
        
        x = (x / 255) * 2 - 1
        t = (t / 255) * 2 - 1
        aux_cond = self._build_aux_cond(x)
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        cloudy = x[:3,...]
        filename = self.imlistl[index].split('.')[0]

        return {
            "cloudy": cloudy,
            "cond_image": x,
            "aux_cond": aux_cond,
            "label": t,
            "M": M,
            "image_path": filename + ".png"
        }

    def __len__(self):
        return len(self.imlistl)
