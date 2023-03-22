# supporting data shape: (H, W), (T, H, W) and (T, D, H, W)
import numpy as np

from skimage.metrics import normalized_root_mse, structural_similarity


def NRMSE(img_pred: np.ndarray, img: np.ndarray):
    metric_val = normalized_root_mse(img_pred, img)

    return metric_val


def SSIM(img_pred: np.ndarray, img: np.ndarray):
    assert img_pred.ndim in [2, 3, 4]
    img_pred = img_pred[None, ...]
    img = img[None, ...]
    metric_val = structural_similarity(img_pred, img, channel_axis=0)

    return metric_val


REGISTERED_METRICS = {
    "NRMSE": NRMSE,
    "SSIM": SSIM
}
