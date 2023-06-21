import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import os
import yaml
import argparse
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from PIL import Image
from ImplicitNeuralRepr.models import SirenComplex
from ImplicitNeuralRepr.configs import FIGSIZE_UNIT
from einops import rearrange
from typing import Union


def vis_volume(data):
    # data: (D, H, W) or (D, W, H)
    if isinstance(data, torch.Tensor):
        data = ptu.to_numpy(data)
    img_viewer = sitk.ImageViewer()
    img_sitk = sitk.GetImageFromArray(data)
    img_viewer.Execute(img_sitk)


def vis_images(*images, **kwargs):
    """
    kwargs: if_save, save_dir, filename, titles, figsize_unit, normalize: {linear | none},
    vmins, vmaxs
    """
    normalize = kwargs.get("normalize", None)
    vmins = kwargs.get("vmins", None)
    vmaxs = kwargs.get("vmaxs", None)
    figsize_unit = kwargs.get("figsize_unit", 3.6)
    num_imgs = len(images)
    fig, axes = plt.subplots(1, num_imgs, figsize=(figsize_unit * num_imgs, figsize_unit))
    if num_imgs == 1:
        axes = [axes]
    titles = kwargs.get("titles", None)
    if titles is not None:
        assert len(titles) == len(images)
    for i, (img_iter, axis) in enumerate(zip(images, axes)):
        channel = 0
        # channel = 0 if img_iter.shape[0] == 1 else 1
        if isinstance(img_iter, torch.Tensor):
            img_iter = ptu.to_numpy(img_iter)
        img_iter = img_iter[channel]
        if normalize is None:
            handle = axis.imshow(img_iter, vmin=vmins[i], vmax=vmaxs[i], cmap="gray")
        else:
            handle = axis.imshow(img_iter, cmap="gray")
        plt.colorbar(handle, ax=axis)
        if titles is not None:
            axis.set_title(titles[i])

    fig.tight_layout()
    if_save = kwargs.get("if_save", False)
    if if_save:
        save_dir = kwargs.get("save_dir", "./outputs")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        assert "filename" in kwargs
        filename = kwargs.get("filename")
        filename_full = os.path.join(save_dir, filename)
        fig.savefig(filename_full)
    else:
        plt.show()
    plt.close()


def vis_signals(*signals, **kwargs):
    """
    signals: list[(C, T)]
    kwargs: if_save, save_dir, filename, titles, ylim, figsize_unit
    """
    figsize_unit = kwargs.get("figsize_unit", 3.6)
    num_imgs = len(signals)
    fig, axes = plt.subplots(num_imgs, 1, figsize=(figsize_unit * 3, figsize_unit * num_imgs))
    if num_imgs == 1:
        axes = [axes]
    titles = kwargs.get("titles", None)
    ylim = kwargs.get("ylim", [-1, 1])
    if titles is not None:
        assert len(titles) == len(signals)
    for i, (img_iter, axis) in enumerate(zip(signals, axes)):
        channel = 0
        # channel = 0 if img_iter.shape[0] == 1 else 1
        if isinstance(img_iter, torch.Tensor):
            img_iter = ptu.to_numpy(img_iter)
        img_iter = img_iter[channel]
        handle = axis.plot(img_iter)
        axis.set_ylim(ylim)
        if titles is not None:
            axis.set_title(titles[i])

    fig.tight_layout()
    if_save = kwargs.get("if_save", False)
    if if_save:
        save_dir = kwargs.get("save_dir", "./outputs")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        assert "filename" in kwargs
        filename = kwargs.get("filename")
        filename_full = os.path.join(save_dir, filename)
        fig.savefig(filename_full)
    else:
        plt.show()
    plt.close()


def vis_multi_channel_signal(signal: torch.Tensor, num_channels=None, **kwargs):
    """
    For only one signal: (C, T)
    """
    if num_channels is not None:
        signal = signal[:num_channels, :]
    signal_list = [signal[c:c + 1, :] for c in range(signal.shape[0])]
    titles = [f"channel_{i}" for i in range(signal.shape[0])]
    vis_signals(*signal_list, titles=titles, **kwargs)


def save_vol_as_gif(vol: Union[torch.Tensor, np.ndarray], save_dir: str, filename: str, **kwargs):
    """
    vol: (T, C, H, W)

    kwargs: duration, if_normalize
    """
    if_normalize = kwargs.get("if_normalize", False)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if isinstance(vol, torch.Tensor):
        vol = ptu.to_numpy(vol)
        
    assert ".gif" in filename
    T, C, H, W = vol.shape
    duration = kwargs.get("duration", T)
    loop = kwargs.get("loop", 0)
    assert C in [1, 3]
    save_path = os.path.join(save_dir, filename)
    if isinstance(vol, torch.Tensor):
        vol = ptu.to_numpy(vol)

    # vol = (vol * 255).astype(np.uint8)
    if if_normalize:
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        # vol = np.clip(vol, 0., 1.)
    vol *= 255
    if C == 1:
        vol = vol[:, 0, ...]  # (T, H, W)
    else:
        vol = np.transpose(vol, axes=(0, 2, 3, 1))  # (T, H, W, C)

    imgs = []
    for t in range(vol.shape[0]):
        frame = vol[t, ...]
        # frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
        imgs.append(Image.fromarray(frame.astype(np.uint8)))
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=duration, loop=loop)


def load_yml_file(filename: str):
    assert ".yml" in filename
    with open(filename, "r") as rf:
        data = yaml.load(rf, yaml.Loader)

    data = dict2namespace(data)

    return data


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dict2str(params: dict, delimiter=", "):
    out_str = ""
    for i, (key, val) in enumerate(params.items()):
        delimiter_iter = delimiter
        if i == len(params) - 1:
            delimiter_iter = ""
        out_str += f"{key}: {val}{delimiter_iter}"
    
    return out_str


def expand_dim_as(src: torch.Tensor, tgt: torch.Tensor):
    """
    src: (B,), tgt: (B, ...)
    """
    num_dims = tgt.ndim
    expand_shape = np.ones((num_dims), dtype=int)
    expand_shape[0] = src.shape[0]
    src_expanded = src.reshape(tuple(expand_shape))

    return src_expanded


def siren_param_hist(model: SirenComplex, **kwargs):
    """
    kwargs: figsize: tuple, weight_bins: int, bias_bins: int
    """
    # print("Creating param historam...")
    figsize = kwargs.get("figsize", None)
    bias_bins = kwargs.get("bias_bins", 20)
    weight_bins = kwargs.get("weight_bins", 80)
    layer_list = model.siren.net_list
    if figsize is None:
        figsize = (FIGSIZE_UNIT * len(layer_list), FIGSIZE_UNIT * 2)
    fig, axes = plt.subplots(2, len(layer_list), figsize=figsize)
    for i in range(len(layer_list)):
        # print(f"Current: {i + 1}/{len(layer_list)}")
        layer_iter = layer_list[i]
        if isinstance(layer_iter, nn.Linear):
            axes[0, i].hist(ptu.to_numpy(layer_iter.weight).flatten(), bins=weight_bins)
            axes[1, i].hist(ptu.to_numpy(layer_iter.bias), bins=bias_bins)
        else:
            axes[0, i].hist(ptu.to_numpy(layer_iter.linear.weight).flatten(), bins=weight_bins)
            axes[1, i].hist(ptu.to_numpy(layer_iter.linear.bias), bins=bias_bins)

    for axis in axes.flatten():
        axis.grid(axis="y")
    fig.tight_layout()

    return fig, axes  


# def nanmean(X: torch.Tensor, **kwargs):
#     if not torch.is_complex(X):
#         return torch.nanmean(**kwargs)

#     X_out = torch.nanmean(torch.real(X), **kwargs) + 1j * torch.nanmean(torch.imag(X), **kwargs)

#     return X_out


def temporal_interpolation(x: torch.Tensor, out_size: int, mode="bicubic"):
    """ 
    x: (B, T, H, W) or (T, H, W)
    """
    x_ndim = x.ndim
    if x_ndim == 3:
        x = x.unsqueeze(0)  # (1, T, H, W)
    B, T, H, W = x.shape
    x = rearrange(x, "B T H W -> B H W T")
    x = F.interpolate(x, size=(W, out_size), mode=mode)
    x = rearrange(x, "B H W T -> B T H W")
    if x_ndim == 3:
        x = x.squeeze(0)

    # the same shape as input
    return x
