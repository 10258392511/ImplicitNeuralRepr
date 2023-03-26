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
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from ImplicitNeuralRepr.configs import load_config
from ImplicitNeuralRepr.datasets import (
    load_data, 
    CoordDataset, 
    MetaCoordDM, 
    add_phase
)
from ImplicitNeuralRepr.utils.utils import vis_images, save_vol_as_gif
from ImplicitNeuralRepr.models import load_model, reload_model
from ImplicitNeuralRepr.linear_transforms import load_linear_transform
from ImplicitNeuralRepr.pl_modules.trainers import Train2DTimeReg
from pytorch_lightning.callbacks import ModelCheckpoint
from ImplicitNeuralRepr.pl_modules.callbacks import Train2DTimeRegCallack
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="../2d_time_reg_logs")
    parser.add_argument("--output_dir", default="../outputs_implicit_neural_repr/2d_time_reg")

    parser.add_argument("--cine_ds_name", default="CINE127")
    parser.add_argument("--cine_idx", type=int, default=10)
    parser.add_argument("--cine_mode", default="val")
    parser.add_argument("--cine_ds_type", default="2d+time")
    parser.add_argument("--Lambda", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--task_name", default="2d+time+reg")
    parser.add_argument("--noise_std", type=float, default=1e-3)

    parser.add_argument("--lamda_reg_profile", type=float, default=0.01)

    parser.add_argument("--num_epochs", type=int, default=500)
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
    Lambda = args_dict["Lambda"]
    spatial_ds = CoordDataset((Lambda, T, H, W), (2, 3))
    temporal_ds = CoordDataset((Lambda, T, H, W), (1,))
    reg_ds = CoordDataset((Lambda, T, H, W), (0,))
    dm_params = {
    "batch_size": args_dict["batch_size"],
    "num_workers": args_dict["num_workers"]
    }
    dm = MetaCoordDM(
        dm_params, 
        [spatial_ds, temporal_ds, reg_ds], 
        [len(spatial_ds), len(temporal_ds), len(reg_ds)], 
        spatial_ds
    )
    save_to_subdir_dict = {
        "original": img_complex
    }

    torch.save(img_complex, os.path.join(args_dict["output_dir"], "original.pt"))
    save_vol_as_gif(torch.abs(img_complex.unsqueeze(1)), save_dir=args_dict["output_dir"], filename="orig_mag.gif")  # (T, H, W) -> (T, 1, H, W)
    save_vol_as_gif(torch.angle(img_complex.unsqueeze(1)), save_dir=args_dict["output_dir"], filename="orig_phase.gif")

    # load model
    siren = load_model(args_dict["task_name"])
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

    lit_model_params = {
        "lamda_reg_profile": args_dict["lamda_reg_profile"],
    }

    lit_model = Train2DTimeReg(
        siren,
        measurement.to(ptu.DEVICE), 
        config_dict, 
        lit_model_params
    )

    if not args_dict["if_train"]:
        ckpt_path = reload_model(args_dict["task_name"])
        lit_model.load_from_checkpoint(ckpt_path, siren=siren, measurement=measurement.to(ptu.DEVICE), config=config_dict, params=lit_model_params)

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
        monitor="epoch_loss", 
        save_last=True,
        save_on_train_epoch_end=True
    )
    callbacks.append(model_ckpt)

    train_2d_time_callback_params = {
        "save_dir": args_dict["output_dir"],
        "save_interval": args_dict["val_interval"]
    }
    train_2d_time_callback = Train2DTimeRegCallack(train_2d_time_callback_params)
    callbacks.append(train_2d_time_callback)
    
    if args_dict["if_debug"]:
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            callbacks=callbacks,
            fast_dev_run=10
        )
    else:
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
    Train2DTimeReg.save_preds(preds, args_dict["output_dir"])
