import sys
import os

path = os.path.abspath(__file__)
for _ in range(3):
    path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

import argparse
import torch
import pickle
import time
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from ImplicitNeuralRepr.configs import load_config
from ImplicitNeuralRepr.datasets import (
    load_data,
    SpatialTemporalRegSamplingDM,
    FracSpatialTemporalRegDM,
    add_phase
)
from ImplicitNeuralRepr.utils.utils import vis_images, save_vol_as_gif, siren_param_hist
from ImplicitNeuralRepr.models import load_model, GridSample, reload_model
from ImplicitNeuralRepr.linear_transforms import load_linear_transform
from ImplicitNeuralRepr.pl_modules.trainers import Train2DTimeExplicitReg
from pytorch_lightning.callbacks import ModelCheckpoint
from ImplicitNeuralRepr.pl_modules.callbacks import Train2DTimeCallack
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="../2d_time_explicit_reg_logs")
    parser.add_argument("--output_dir", default="../outputs_implicit_neural_repr/2d_time_explicit_reg")

    parser.add_argument("--cine_ds_name", default="CINE127")
    parser.add_argument("--cine_idx", type=int, default=10)
    parser.add_argument("--cine_mode", default="val")
    parser.add_argument("--cine_ds_type", default="2d+time")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--spatial_batch_size", type=int, default=32)
    parser.add_argument("--pred_batch_size", type=int, default=32)
    parser.add_argument("--num_temporal_repeats", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--task_name", default="2d+time+explicit_reg")
    parser.add_argument("--noise_std", type=float, default=1e-3)

    parser.add_argument("--kernel_shape", type=int, nargs="+", default=[1, 1, 12, 32, 32])
    parser.add_argument("--lam_min", type=int, default=-3)
    parser.add_argument("--lam_max", type=int, default=0)
    parser.add_argument("--num_lams", type=int, default=4)  # number of lambda's per batch

    parser.add_argument("--siren_weight", type=float, default=1.)
    parser.add_argument("--grid_sample_weight", type=float, default=1.)
    parser.add_argument("--if_ZF", action="store_true")

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--val_interval", type=int, default=50)
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
    cine_ds = load_data(args_dict["cine_ds_name"], args_dict["cine_mode"], ds_type=args_dict["cine_ds_type"])
    img = cine_ds[args_dict["cine_idx"]][0]
    img_complex = add_phase(img[:, None, ...], init_shape=(5, 5, 5), seed=args_dict["seed"], mode="2d+time")
    img_complex = img_complex.squeeze(1)  # (T, H, W)
    T, H, W = img_complex.shape

    lam_tfm = lambda lam : 10 ** lam
    dm_params = {
        "in_shape": (T, H, W),
        "lam_min": args_dict["lam_min"],
        "lam_max": args_dict["lam_max"],
        "num_lams": args_dict["num_lams"],
        "lam_pred": -1.,
        "spatial_batch_size": args_dict["spatial_batch_size"],
        "pred_batch_size": args_dict["pred_batch_size"],
        "num_temporal_repeats": args_dict["num_temporal_repeats"],
        "num_workers": args_dict["num_workers"]
    }
    # dm = SpatialTemporalRegSamplingDM(dm_params, lam_tfm)
    dm = FracSpatialTemporalRegDM(dm_params, lam_tfm)
    # lam_grid = dm.spatial_ds.lam_grid
    lam_grid = torch.linspace(-1, 1, args_dict["lam_max"] - args_dict["lam_min"])

    torch.save(img_complex, os.path.join(args_dict["output_dir"], "original.pt"))
    save_vol_as_gif(torch.abs(img_complex.unsqueeze(1)), save_dir=args_dict["output_dir"], filename="orig_mag.gif")  # (T, H, W) -> (T, 1, H, W)
    save_vol_as_gif(torch.angle(img_complex.unsqueeze(1)), save_dir=args_dict["output_dir"], filename="orig_phase.gif")

    # load model
    siren = load_model(args_dict["task_name"])
    fig_hist, axes_hist = siren_param_hist(siren)
    fig_hist.savefig(os.path.join(args_dict["output_dir"], "siren_before.png"))

    grid_sample_params = {
        "kernel_shape": args_dict["kernel_shape"]
    }
    grid_sample = GridSample(grid_sample_params)
    lin_tfm = load_linear_transform(args_dict["task_name"], "dc")
    torch.manual_seed(args_dict["seed"])
    measurement = lin_tfm(img_complex)   # (T, H, W, K)
    measurement += args_dict["noise_std"] * torch.randn_like(measurement)
    zf = lin_tfm.conj_op(measurement)  # (T, H, W)

    torch.save(measurement, os.path.join(args_dict["output_dir"], "measurement.pt"))
    torch.save(zf, os.path.join(args_dict["output_dir"], "ZF.pt"))
    torch.save(lin_tfm.sens_maps, os.path.join(args_dict["output_dir"], "sens.pt"))  # (K, H, W)
    # visualize measurement of the first coil
    save_vol_as_gif(torch.log(torch.abs(measurement[:, None, :, :, 0]) + 1e-6), save_dir=args_dict["output_dir"], filename="measurement_mag.gif")
    save_vol_as_gif(torch.angle(measurement[:, None, :, :, 0]), save_dir=args_dict["output_dir"], filename="measurement_phase.gif")
    
    save_vol_as_gif(torch.abs(zf.unsqueeze(1)), save_dir=args_dict["output_dir"], filename="zf_mag.gif")  # (T, H, W) -> (T, 1, H, W)
    save_vol_as_gif(torch.angle(zf.unsqueeze(1)), save_dir=args_dict["output_dir"], filename="zf_phase.gif")
    
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

    lit_model_params = {
        "siren_weight": args_dict["siren_weight"],
        "grid_sample_weight": args_dict["grid_sample_weight"]
    }

    zf_in = None
    if args_dict["if_ZF"]:
        zf_in = zf.to(ptu.DEVICE)
    lit_model = Train2DTimeExplicitReg(
        siren, 
        grid_sample, 
        measurement.to(ptu.DEVICE), 
        config_dict, 
        lit_model_params,
        zf_in
    )

    if not args_dict["if_train"]:
        ckpt_path = reload_model(args_dict["task_name"])
        lit_model.load_from_checkpoint(ckpt_path, siren=siren, grid_sample=grid_sample, measurement=measurement.to(ptu.DEVICE), config=config_dict, params=lit_model_params)

    # training
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
        filename="{epoch}_{loss: .2f}",
        # monitor="epoch_loss", 
        monitor="loss",
        save_last=True,
        save_on_train_epoch_end=True
    )
    callbacks.append(model_ckpt)

    train_callback_params = {
        "save_dir": args_dict["output_dir"],
        "save_interval": args_dict["val_interval"]
    }
    train_callback = Train2DTimeCallack(train_callback_params)
    callbacks.append(train_callback)

    if args_dict["if_debug"]:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            fast_dev_run=3,
            reload_dataloaders_every_n_epochs=1
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            logger=logger,
            check_val_every_n_epoch=args_dict["val_interval"],
            max_epochs=args_dict["num_epochs"],
            callbacks=callbacks,
            reload_dataloaders_every_n_epochs=1
        )
    if args_dict["if_train"]:
        time_start = time.time()
        trainer.fit(lit_model, datamodule=dm)
        time_end = time.time()
        time_duration = time_end - time_start
        with open(os.path.join(args_dict["output_dir"], "args_dict.txt"), "a") as wf:
            wf.write(f"training time: {time_duration}\n")
    
    fig_hist, axes_hist = siren_param_hist(siren)
    fig_hist.savefig(os.path.join(args_dict["output_dir"], "siren_after.png"))

    print("Predicting...")
    for lam_iter in lam_grid:
        print(f"lam = {lam_iter}")

        dm_params["lam_pred"] = lam_iter
        # dm = SpatialTemporalRegSamplingDM(dm_params, lam_tfm)
        dm = FracSpatialTemporalRegDM(dm_params, lam_tfm)

        preds = trainer.predict(lit_model, datamodule=dm)
        pred = lit_model.pred2vol(preds).unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
        save_dir = os.path.join(args_dict["output_dir"], f"lam_{lam_iter: .3f}")
        save_vol_as_gif(torch.abs(pred), save_dir=save_dir, filename=f"recons_mag.gif")
        save_vol_as_gif(torch.angle(pred), save_dir=save_dir, filename=f"recons_phase.gif")
        torch.save(pred.detach().cpu().squeeze(1), os.path.join(save_dir, f"reconstructions.pt"))

        pred = pred.squeeze()  # (T, H, W)
        T, H, W = pred.shape
        pred_iter = pred[None, T // 2, :, :] 
        vis_images(torch.abs(pred_iter), torch.angle(pred_iter), if_save=True, save_dir=save_dir, filename="half_T.png")
        pred_iter = pred[None, :, H // 2, :].transpose(-1, -2)  # x-t
        vis_images(torch.abs(pred_iter), torch.angle(pred_iter), if_save=True, save_dir=save_dir, filename="half_H.png")
        pred_iter = pred[None, :, :, W // 2].transpose(-1, -2)  # y-t
        vis_images(torch.abs(pred_iter), torch.angle(pred_iter), if_save=True, save_dir=save_dir, filename="half_W.png")
