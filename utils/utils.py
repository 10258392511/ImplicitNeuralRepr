import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import os
import yaml
import argparse
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from PIL import Image
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
    kwargs: if_save, save_dir, filename, titles, figsize_unit
    """
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
    if_normalize = kwargs.get("if_normalize", True)
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
