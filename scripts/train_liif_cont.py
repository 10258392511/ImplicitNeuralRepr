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
from ImplicitNeuralRepr.models import load_model, reload_model
from ImplicitNeuralRepr.utils.utils import vis_images, save_vol_as_gif, temporal_interpolation
from ImplicitNeuralRepr.pl_modules.trainers import TrainLIIF3DConv
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from ImplicitNeuralRepr.pl_modules.callbacks import TrainLIIF3DConvCallback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="2d+time_liif_cont")
    parser.add_argument("--log_dir", default="../2d_time_liif_3d_conv_logs")
    parser.add_argument("--output_dir", default="../outputs_implicit_neural_repr/2d_time_liif_3d_conv")
    # DataModule
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    # Callback
    parser.add_argument("--test_idx", type=int, default=-1)
    # training
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--if_train", action="store_true")
    parser.add_argument("--if_debug", action="store_true")
    parser.add_argument("--if_pred", action="store_true")
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
        "batch_size": args_dict["batch_size"],
        "test_batch_size": args_dict["test_batch_size"],
        "num_workers": args_dict["num_workers"]
    }
    dm_params.update(config_dict["dataset"])
    mask_config = load_mask_config()
    dm = CINEContDM(dm_params)

    # training
    model = load_model(args_dict["task_name"])
    lit_model = TrainLIIF3DConv(model, config_dict)

    if args_dict["if_pred"] and (not args_dict["if_train"]):
        ckpt_path = reload_model(args_dict["task_name"])
        lit_model.load_from_checkpoint(ckpt_path, model=model, config=config_dict)

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

    lr_monitor = LearningRateMonitor("epoch", True)
    callbacks.append(lr_monitor)

    train_callback_params = {
        "save_dir": args_dict["output_dir"],
        "save_interval": args_dict["val_interval"],
        "test_idx": args_dict["test_idx"],
    }
    train_callback = TrainLIIF3DConvCallback(train_callback_params)

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

    if args_dict["if_pred"]:
        print("-" * 100)
        print("Predicting...")
        all_preds = trainer.predict(lit_model, datamodule=dm)  # [((B_test, T, H, W), (B_test,))]
        all_recons = [item[0] for item in all_preds]
        all_recons = torch.cat(all_recons, axis=0)  # (N_all, T, H, W)
        all_errors = np.concatenate([item[1] for item in all_preds], axis=0)  # (N_all,)
        train_val_split_idx = len(dm.train_ds)
        metric_dict = {
            "train_error": all_errors[:train_val_split_idx].mean(),
            "val_error": all_errors[train_val_split_idx:].mean(),
            "all_error": all_errors.mean()
        }
        with open(os.path.join(args_dict["output_dir"], "args_dict.txt"), "a") as wf:
            wf.write("-" * 100)
            wf.write("\n")
            for key, val in metric_dict.items():
                wf.write(f"{key}: {val}\n")
        
        # val_idx = 0
        val_inds = [0, 2, 4, 7, 8]
        for val_idx in val_inds:
            data_idx = train_val_split_idx + val_idx
            data_dict = dm.test_ds[data_idx]
            img = data_dict[IMAGE_KEY]  # (T, H, W)
            T_query = img.shape[0]
            img_zf = data_dict[ZF_KEY]
            img_zf = temporal_interpolation(torch.abs(img_zf), T_query)
            output_dir = os.path.join(args_dict["output_dir"], f"idx_{data_idx}")
            save_vol_as_gif(torch.abs(img.unsqueeze(1)), save_dir=output_dir, filename="orig_mag.gif")
            save_vol_as_gif(torch.abs(img_zf.unsqueeze(1)), save_dir=output_dir, filename="zf_mag.gif")
            torch.save(img.detach().cpu(), os.path.join(output_dir, "orig.pt"))
            torch.save(img_zf.detach().cpu(), os.path.join(output_dir, "zf.pt"))
            
            recons = all_recons[data_idx, ...].detach().cpu().clone()  # (T, H, W)
            save_vol_as_gif(torch.abs(recons.unsqueeze(1)), save_dir=output_dir, filename="recons_mag.gif")            
            torch.save(recons, os.path.join(output_dir, "recons.pt"))
