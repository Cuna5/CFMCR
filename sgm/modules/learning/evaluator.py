# code from https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal
import math

import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import lpips
import numpy as np

first_to_cuda = False
loss_fn = lpips.LPIPS(net='alex', version=0.1)

def caculate_ssim(imgA, imgB):
    imgA1 = np.tensordot(imgA.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    imgB1 = np.tensordot(imgB.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    score = SSIM(imgA1, imgB1, data_range=255)
    return score

def caculate_psnr(imgA, imgB):
    imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
    imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
    psnr = PSNR(imgA1, imgB1, data_range=255)
    return psnr

def caculate_lpips(img0, img1):
    # 换到GPU上来算
    im1 = img0.float()
    im2 = img1.float()
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    if not first_to_cuda:
        loss_fn.to(im1.device)    
    current_lpips_distance  = loss_fn.forward(im1, im2)
    return current_lpips_distance.item()

def _ssim_map(imgA, imgB):
    """Full SSIM map (HxW) on the luminance-weighted grayscale image."""
    imgA1 = np.tensordot(imgA.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    imgB1 = np.tensordot(imgB.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    _, full_map = SSIM(imgA1, imgB1, data_range=255, full=True)
    return full_map


def _region_metrics(target01, pred01, img_target, img_pred, region, suffix):
    """Cloud / non-cloud metrics over a boolean HxW region.

    target01/pred01: float numpy [C,H,W] in [0,1] (matches full-image RMSE).
    img_target/img_pred: uint8 torch [C,H,W] (matches full-image PSNR/SSIM convention).
    """
    out = {}
    if region.sum() == 0:
        return {
            f"PSNR_{suffix}": float("nan"),
            f"SSIM_{suffix}": float("nan"),
            f"RMSE_{suffix}": float("nan"),
        }
    diff01 = target01[:, region] - pred01[:, region]
    out[f"RMSE_{suffix}"] = math.sqrt(float(np.mean(diff01 ** 2)))

    a = img_target.cpu().numpy().astype(np.float32)
    b = img_pred.cpu().numpy().astype(np.float32)
    mse255 = float(np.mean((a[:, region] - b[:, region]) ** 2))
    out[f"PSNR_{suffix}"] = 10.0 * math.log10((255.0 ** 2) / max(mse255, 1e-12))

    if img_target.shape[0] == 4:
        ssim_maps = []
        for i in range(img_target.shape[0]):
            imA = img_pred[i].expand(3, *img_pred.shape[-2:])
            imB = img_target[i].expand(3, *img_target.shape[-2:])
            ssim_maps.append(_ssim_map(imA, imB))
        ssim_map = np.mean(np.stack(ssim_maps, axis=0), axis=0)
    else:
        ssim_map = _ssim_map(img_pred, img_target)
    out[f"SSIM_{suffix}"] = float(ssim_map[region].mean())
    return out


def img_metrics(target, pred, mask=None, mask_threshold=0.1):
    rmse = torch.sqrt(torch.mean((target - pred) ** 2)).item()
    imgA = pred.squeeze(0) * 2 - 1 # 0-1 to -1-1
    imgB = target.squeeze(0) * 2 - 1 # 0-1 to -1-1
    # the original code is for 0-255 images
    imgA = ((imgA+1)*127.5).clamp(0, 255).to(torch.uint8)
    imgB = ((imgB+1)*127.5).clamp(0, 255).to(torch.uint8)
    psnr = caculate_psnr(imgA, imgB)
    ssim=0
    lpips=0
    if(imgA.shape[0]==4):
        for i in range(imgA.shape[0]):
            imA = imgA[i]
            imA = imA.expand(3, *imA.shape[-2:])
            imB = imgB[i]
            imB = imB.expand(3, *imB.shape[-2:])
            ssim1 = caculate_ssim(imA, imB)
            lpips1 = caculate_lpips(imA, imB)
            ssim += ssim1
            lpips += lpips1
        ssim = ssim / imgA.shape[0]
        lpips = lpips / imgA.shape[0]
    else:
        ssim = caculate_ssim(imgA, imgB)
        lpips = caculate_lpips(imgA, imgB)
    metric_dict = {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips,
        "RMSE": rmse
    }

    if mask is not None:
        if torch.is_tensor(mask):
            m = mask.squeeze().cpu().numpy()
        else:
            m = np.asarray(mask).squeeze()
        cloudy_region = m > mask_threshold
        target01 = target.squeeze(0).cpu().numpy()
        pred01 = pred.squeeze(0).cpu().numpy()
        metric_dict.update(
            _region_metrics(target01, pred01, imgB, imgA, cloudy_region, "cloudy")
        )
        metric_dict.update(
            _region_metrics(target01, pred01, imgB, imgA, ~cloudy_region, "cloudfree")
        )
    return metric_dict

class avg_img_metrics():
    def __init__(self):
        self.metrics   = [
            'PSNR', 'SSIM', 'LPIPS', 'RMSE',
            'PSNR_cloudy', 'SSIM_cloudy', 'RMSE_cloudy',
            'PSNR_cloudfree', 'SSIM_cloudfree', 'RMSE_cloudfree',
        ]
        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()    

    def reset(self):
        for metric in self.metrics: 
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan
    
    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            # skip variables not registered
            if key not in self.metrics: continue
            # filter variables not translated to numpy yet
            if torch.is_tensor(val): continue
            if isinstance(val, tuple): val=val[0]

            # only keep a running mean of non-nan values
            if np.isnan(val): continue
            # if float("inf") == val: continue # 规避数值溢出

            if not self.running_nonan_count[key]: 
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else: 
                self.running_nonan_count[key]+= 1
                self.running_img_metrics[key] = (self.running_nonan_count[key]-1)/self.running_nonan_count[key] * self.running_img_metrics[key] \
                                                + 1/self.running_nonan_count[key] * val

    def value(self):
        return self.running_img_metrics

    
