import sys
import os

path = os.path.abspath(__file__)
for _ in range(3):
    path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np
import torch
import pickle
import time
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from ImplicitNeuralRepr.configs import (
    load_config, 
    load_mask_config,
    IMAGE_KEY,
    ZF_KEY
)
from ImplicitNeuralRepr.datasets import CINEContDM
from torch.utils.data import DataLoader, Subset
from ImplicitNeuralRepr.models import TemporalTV
from ImplicitNeuralRepr.utils.utils import vis_images, save_vol_as_gif
from ImplicitNeuralRepr.pl_modules.trainers import TrainTemporalTV
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange
from typing import Dict
from ImplicitNeuralRepr.configs import (
    IMAGE_KEY,
    ZF_KEY,
    MASK_KEY,
    MEASUREMENT_KEY
)


def save_batch_and_recons(batch: Dict[str, torch.Tensor], recons: torch.Tensor, save_dir: str):
    # recons: (T', H, W)
    img = batch[IMAGE_KEY]
    zf = batch[ZF_KEY]  # (T, H, W)
    ksp = batch[MEASUREMENT_KEY]  # (T, H, W)
    mask = batch[MASK_KEY]  # (T, 1, W)
    recons, rel_mag_error = TrainTemporalTV.upsample_recons_and_compute_error(recons.unsqueeze(0).to(ptu.DEVICE), img.unsqueeze(0).to(ptu.DEVICE))
    recons = recons.squeeze(0)  # (T', H, W)
    tensors = [img, zf, recons]
    filenames = ["orig", "ZF", "recons"]
    for tensor_iter, filename_iter in zip(tensors, filenames):
        torch.save(tensor_iter.detach().cpu(), os.path.join(save_dir, f"{filename_iter}.pt"))
        tensor_iter = tensor_iter.unsqueeze(1)  # (T, C=1, H, W)
        save_vol_as_gif(torch.abs(tensor_iter), save_dir=save_dir, filename=f"{filename_iter}_mag.gif")

    # C = 1
    vis_images(rearrange(mask, "T C N -> C N T"), if_save=True, save_dir=save_dir, filename="mask.png", normalize=True)

    rel_mag_error = rel_mag_error.item()
    with open(os.path.join(save_dir, "logs.txt"), "a") as wf:
        wf.write(f"rel mag error: {rel_mag_error}\n")
    
    return rel_mag_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="2d+time_TVt")
    parser.add_argument("--log_dir", default="../2d_time_TVt")
    parser.add_argument("--output_dir", default="../outputs_implicit_neural_repr/2d_time_TVt")
    # DataModule
    parser.add_argument("--num_workers", type=int, default=0)
    # training
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--if_debug", action="store_true")
    parser.add_argument("--notes", default="no extra notes")

    args_dict = vars(parser.parse_args())
    config_dict = load_config(args_dict["task_name"])
    all_dict = args_dict.copy()
    all_dict.update(config_dict)

    for dir_name_key in ["log_dir", "output_dir"]:
        dir_name = args_dict[dir_name_key]
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    with open(os.path.join(args_dict["output_dir"], "args_dict.txt"), "w") as wf:
        for key, val in all_dict.items():
            wf.write(f"{key}: {val}\n")
    with open(os.path.join(args_dict["output_dir"], "args_dict.pkl"), "wb") as wf:
        pickle.dump(all_dict, wf)
    
    # load data
    dm_params = {
        "batch_size": 1,
        "test_batch_size": 1,
        "num_workers": args_dict["num_workers"]
    }
    dm_params.update(config_dict["dataset"])
    mask_config = load_mask_config()
    dm = CINEContDM(dm_params)
    dm.setup()
    ds = dm.test_ds

    # training
    all_errors = []
    for idx in range(len(ds)):
        print(f"Current: {idx + 1}/{len(ds)}")
        ds_iter = Subset(ds, [idx])
        batch = ds[idx]
        data_loader_iter = DataLoader(ds_iter, batch_size=1, num_workers=args_dict["num_workers"], pin_memory=True)
        model = TemporalTV(config_dict["model"], batch)
        lit_model = TrainTemporalTV(model, config_dict)

        time_stamp = f"sample_{idx}"
        log_dir = os.path.join(args_dict["log_dir"], time_stamp)
        output_dir = os.path.join(args_dict["output_dir"], time_stamp)
        logger = TensorBoardLogger(args_dict["log_dir"], name=None, version=time_stamp)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(log_dir, "desc.txt"), "w") as wf:
            for key, val in all_dict.items():
                wf.write(f"{key}: {val}\n")
    
        callbacks = []
        model_ckpt = ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="{epoch}_{val_loss: .2f}",
            monitor="val_loss",
            save_last=True,
        )
        callbacks.append(model_ckpt)

        lr_monitor = LearningRateMonitor("epoch", True)
        callbacks.append(lr_monitor)

        if args_dict["if_debug"]:
            trainer = Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=callbacks,
                fast_dev_run=3,
                # num_sanity_val_steps=-1
            )
        else:
            trainer = Trainer(
                accelerator="gpu",
                devices=1,
                logger=logger,
                # num_sanity_val_steps=-1,
                # check_val_every_n_epoch=50,
                max_epochs=args_dict["num_epochs"],
                callbacks=callbacks,
            )
        time_start = time.time()
        trainer.fit(lit_model, train_dataloaders=data_loader_iter)
        time_end = time.time()
        time_duration = time_end - time_start
        with open(os.path.join(output_dir, "args_dict.txt"), "a") as wf:
            wf.write(f"training time: {time_duration}\n")
        
        recons = model.kernel
        rel_mag_error = save_batch_and_recons(batch, recons, output_dir)
        all_errors.append(rel_mag_error)
        print("-" * 100)

    all_errors = np.array(all_errors)
    num_train_samples = len(dm.train_ds)
    train_errors = all_errors[:num_train_samples]
    val_errors = all_errors[num_train_samples:]
    with open(os.path.join(args_dict["output_dir"], "results.txt"), "w") as wf:
        wf.write(f"train_error: {train_errors.mean()}\n")
        wf.write(f"val_error: {val_errors.mean()}\n")
        wf.write(f"number of training samples: {num_train_samples}\n")
