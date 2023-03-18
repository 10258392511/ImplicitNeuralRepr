import os
import sys

path = os.path.abspath(__file__)
for _ in range(3):
    path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

import argparse
import torch
import pickle
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from ImplicitNeuralRepr.datasets import (
    load_data, 
    CoordDataset, 
    MetaCoordDM, 
    add_phase
)
from ImplicitNeuralRepr.utils.utils import (
    vis_images
)
from ImplicitNeuralRepr.configs import load_config
from ImplicitNeuralRepr.models import load_model, reload_model
from ImplicitNeuralRepr.linear_transforms import load_linear_transform
from ImplicitNeuralRepr.pl_modules.trainers import TrainSpatial
from pytorch_lightning.callbacks import StochasticWeightAveraging as SWA, ModelCheckpoint
from ImplicitNeuralRepr.pl_modules.callbacks import TrainSpatialCallack
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--if_train", action="store_true")
    parser.add_argument("--log_dir", default="./spatial_logs")
    parser.add_argument("--output_dir", default="../outputs_implicit_neural_repr/spatial")

    parser.add_argument("--cine_ds_name", default="CINE127")
    parser.add_argument("--cine_idx", type=int, default=300)
    parser.add_argument("--cine_mode", default="val")
    parser.add_argument("--cine_ds_type", default="spatial")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", nargs="+", type=int, default=[1])

    parser.add_argument("--task_name", default="spatial")
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=5)
    parser.add_argument("--noise_std", type=float, default=0.001)

    parser.add_argument("--if_pred_res", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lamda_reg", type=float, default=0.001)
    # StepLR
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.2)
    # SWA
    parser.add_argument("--if_swa", action="store_true")
    parser.add_argument("--swa_lrs", type=float, default=5e-4)
    parser.add_argument("--swa_epoch_start", type=float, default=0.5)

    parser.add_argument("--val_interval", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--notes", default="no extra notes")

    args_dict = vars(parser.parse_args())
    if not os.path.isdir(args_dict["output_dir"]):
        os.makedirs(args_dict["output_dir"])
    
    with open(os.path.join(args_dict["output_dir"], "args_dict.pkl"), "wb") as wf:
        pickle.dump(args_dict, wf)
    
    config_dict = load_config(args_dict["task_name"])
    config_dict["model"].update({
        "hidden_features": args_dict["hidden_features"],
        "hidden_layers": args_dict["hidden_layers"]
    })

    torch.manual_seed(args_dict["seed"])
    # load data
    cine_ds = load_data(args_dict["cine_ds_name"], args_dict["cine_mode"], ds_type=args_dict["cine_ds_type"])
    img = cine_ds[args_dict["cine_idx"]][0]  # (H, W)
    img_complex = add_phase(img[None, None, ...], seed=args_dict["seed"], mode="spatial")
    img_complex = img_complex.squeeze()  # (H, W)
    vis_images(
        torch.abs(img_complex[None, ...]), 
        torch.angle(img_complex[None, ...]), 
        if_save=True, 
        save_dir=args_dict["output_dir"], 
        filename="original.png", 
        titles=["mag", "phase"]
    )
    T = 1  # dummy dimension
    H, W = img_complex.shape
    spatial_ds = CoordDataset((T, H, W), (1, 2))
    dm_params = {
        "batch_size": args_dict["batch_size"],
        "num_workers": args_dict["num_workers"]
    }
    dm = MetaCoordDM(dm_params, [spatial_ds], args_dict["num_samples"], spatial_ds)

    # load model & generate measurement and ZF
    model = load_model(args_dict["task_name"], config_dict=config_dict)
    lin_tfm = load_linear_transform(args_dict["task_name"], "dc")
    measurement = lin_tfm(img_complex)  # (H, W)
    measurement += args_dict["noise_std"] * torch.randn_like(measurement)
    zf = lin_tfm.conj_op(measurement)  # (H, W)
    torch.save(measurement, os.path.join(args_dict["output_dir"], "measurement.pt"))
    torch.save(zf, os.path.join(args_dict["output_dir"], "ZF.pt"))
    torch.save(lin_tfm.sens_maps, os.path.join(args_dict["output_dir"], "sens.pt"))  # (K, H, W)
    vis_images(
        torch.log(torch.abs(measurement[..., 0][None, ...]) + 1e-6), 
        torch.angle(measurement[..., 0][None, ...]),
        if_save=True,
        save_dir=args_dict["output_dir"],
        filename="measurement.png",
        titles=["mag", "phase"]
    )
    vis_images(
        torch.abs(zf[None, ...]), 
        torch.angle(zf[None, ...]),
        if_save=True,
        save_dir=args_dict["output_dir"],
        filename="ZF.png",
        titles=["mag", "phase"]
    )
    sense_maps = [lin_tfm.sens_maps[i].unsqueeze(0) for i in range(lin_tfm.sens_maps.shape[0])]  # (K, H, W) -> [(1, H, W)...]
    vis_images(
        *sense_maps,
        if_save=True,
        save_dir=args_dict["output_dir"],
        filename="sens.png"
    )

    lit_model_params = {
        "lr": args_dict["lr"],
        "lamda_reg": args_dict["lamda_reg"],
        "if_pred_res": args_dict["if_pred_res"],
        "step_size": args_dict["step_size"],
        "gamma": args_dict["gamma"]
    }
    lit_model = TrainSpatial(model, measurement.to(ptu.DEVICE), lin_tfm, lit_model_params)

    if not args_dict["if_train"]:
        ckpt_path = reload_model(args_dict["task_name"])
        lit_model.load_from_checkpoint(ckpt_path, model=model, measurement=measurement.to(ptu.DEVICE), lin_tfm=lin_tfm, params=lit_model_params)

    # training
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_dir = os.path.join(args_dict["log_dir"], time_stamp)
    logger = TensorBoardLogger(args_dict["log_dir"], name=None, version=time_stamp)

    callbacks = []
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}_{loss: .2f}",
        monitor="loss", 
        save_last=True
    )
    callbacks.append(model_ckpt)

    if args_dict["if_swa"]:
        swa_params = {
            "swa_lrs": args_dict["swa_lrs"],
            "swa_epoch_start": args_dict["swa_epoch_start"],
            "device": None
        }
        swa = SWA(**swa_params)
        callbacks.append(swa)

    train_spatial_callback_params = {
        "save_dir": args_dict["output_dir"],
        "save_interval": args_dict["val_interval"]
    }
    train_spatial_callback = TrainSpatialCallack(train_spatial_callback_params, img_complex)
    callbacks.append(train_spatial_callback)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "desc.txt"), "w") as wf:
        for key, val in args_dict.items():
            wf.write(f"{key}: {val}\n")
    
    # training
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        check_val_every_n_epoch=args_dict["val_interval"],
        max_epochs=args_dict["num_epochs"],
        callbacks=callbacks
    )
    if args_dict["if_train"]:
        trainer.fit(lit_model, datamodule=dm)
    
    preds = trainer.predict(lit_model, datamodule=dm)
    pred = preds[0][0].unsqueeze(0)  # (1, H, W)
    img_gt = img_complex
    error_img = pred - img_gt
    torch.save(pred.detach().cpu(), os.path.join(args_dict["output_dir"], f"recons.pt"))
    vis_images(
        torch.abs(pred), 
        torch.angle(pred), 
        torch.abs(error_img), 
        torch.angle(error_img),
        if_save=True, 
        save_dir=args_dict["output_dir"], 
        filename=f"recons.png", 
        titles=["mag", "phase", "error mag", "error phase"]
    )
