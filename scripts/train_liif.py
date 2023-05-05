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

from ImplicitNeuralRepr.configs import load_config
from ImplicitNeuralRepr.linear_transforms import load_linear_transform
from ImplicitNeuralRepr.datasets import CINEImageKSPDM
from ImplicitNeuralRepr.models import load_model, reload_model
from ImplicitNeuralRepr.utils.utils import vis_images
from ImplicitNeuralRepr.pl_modules.trainers import TrainLIIF
from pytorch_lightning.callbacks import ModelCheckpoint
from ImplicitNeuralRepr.pl_modules.callbacks import TrainLIIFCallback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", choices=["2d+time_liif_param", "2d+time_liif_non_param"], default="2d+time_liif_param")
    parser.add_argument("--log_dir", default="../2d_time_liif_logs")
    parser.add_argument("--output_dir", default="../outputs_implicit_neural_repr/2d_time_liif")
    # DataModule
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    # Callback
    parser.add_argument("--temporal_res", type=int, default=50)
    parser.add_argument("--test_idx", type=int, default=0)
    # training
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--if_train", action="store_true")
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
    lin_tfm = load_linear_transform(args_dict["task_name"], "dc")
    sense_maps = [lin_tfm.sens_maps[i].unsqueeze(0) for i in range(lin_tfm.sens_maps.shape[0])]  # (K, H, W) -> [(1, H, W)...]
    vis_images(
        *sense_maps,
        if_save=True,
        save_dir=args_dict["output_dir"],
        filename="sens.png"
    )
    mask = rearrange(lin_tfm.random_under_fourier.mask, "T C N -> C N T")  # C = 1
    vis_images(
        mask,
        if_save=True,
        save_dir=args_dict["output_dir"],
        filename="mask.png"
    )

    dm_params = {
    "res": 127,
    "seed": config_dict["dataset"]["seed"],
    "train_test_split": config_dict["dataset"]["train_test_split"],
    "batch_size": args_dict["batch_size"],
    "num_workers": args_dict["num_workers"]
    }
    dm = CINEImageKSPDM(dm_params, lin_tfm)

    # training
    model = load_model(args_dict["task_name"])
    lit_model = TrainLIIF(model, config_dict, lin_tfm)

    if not args_dict["if_train"]:
        ckpt_path = reload_model(args_dict["task_name"])
        lit_model.load_from_checkpoint(ckpt_path, model=model, config_dict=config_dict, lin_tfm=lin_tfm)

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_dir = os.path.join(args_dict["log_dir"], time_stamp)
    logger = TensorBoardLogger(args_dict["log_dir"], name=None, version=time_stamp)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "desc.txt"), "w") as wf:
        for key, val in all_dict.items():
            wf.write(f"{key}: {val}\n")
    
    callbacks = []
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}_{epoch_val_loss: .2f}",
        monitor="epoch_val_loss",
        save_last=True,
    )
    callbacks.append(model_ckpt)

    train_callback_params = {
        "save_dir": args_dict["output_dir"],
        "save_interval": args_dict["val_interval"],
        "test_idx": args_dict["test_idx"],
        "temporal_res": args_dict["temporal_res"]
    }
    train_callback = TrainLIIFCallback(train_callback_params)
    callbacks.append(train_callback)

    if args_dict["if_debug"]:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            fast_dev_run=3,
            num_sanity_val_steps=-1
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            logger=logger,
            num_sanity_val_steps=-1,
            check_val_every_n_epoch=1,
            max_epochs=args_dict["num_epochs"],
            callbacks=callbacks,
        )
    if args_dict["if_train"]:
        time_start = time.time()
        trainer.fit(lit_model, datamodule=dm)
        time_end = time.time()
        time_duration = time_end - time_start
        with open(os.path.join(args_dict["output_dir"], "args_dict.txt"), "a") as wf:
            wf.write(f"training time: {time_duration}\n")
    