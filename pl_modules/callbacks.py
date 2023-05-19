from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import pytorch_lightning as pl
import os
import sys
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from pytorch_lightning.callbacks import Callback
from ImplicitNeuralRepr.utils.utils import vis_images, save_vol_as_gif
from einops import rearrange
from ImplicitNeuralRepr.configs import IMAGE_KEY, MEASUREMENT_KEY, ZF_KEY, COORD_KEY
from typing import Union


class TrainSpatialCallack(Callback):
    """
    Saves residual and reconstructed images.
    """
    def __init__(self, params: dict, img_gt: torch.Tensor):
        """
        params: save_dir, save_interval
        """
        super().__init__()
        self.params = params
        self.img_gt = img_gt  # (H, W)
        self.counter = -1
        self.params["save_dir"] = os.path.join(self.params["save_dir"], "screenshots/")
        if not os.path.isdir(self.params["save_dir"]):
            os.makedirs(self.params["save_dir"])
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.counter += 1
        if self.counter % self.params["save_interval"] != 0 and self.counter != trainer.max_epochs - 1:
            return
        # preds = trainer.predict(pl_module, datamodule=trainer.datamodule)  # [(B, H, W)...]
        preds = []
        model = pl_module.model.to(ptu.DEVICE)
        for batch in trainer.datamodule.predict_dataloader():
            H, W = pl_module.zf.shape
            x = batch[..., 1:]  # remove axis 0 (T) 
            x = x.to(ptu.DEVICE)
            x = rearrange(x, "B (H W) D -> B H W D", H=H)  # (B, H, W, 2)
            # Note we only have one image and we don't take slices, so there's no need to call val2idx(.)
            pred_res = model(x).squeeze(-1)  # (B, H, W, 1) -> (B, H, W)
            pred = pl_module.collate_pred(pred_res)  # (B, H, W)
            # pred = pred_res
            preds.append(pred)

        pred = preds[0][0].unsqueeze(0)  # (1, H, W)
        img_gt = self.img_gt.to(pred.device)
        error_img = pred - img_gt
        torch.save(pred.detach().cpu(), os.path.join(self.params["save_dir"], f"recons_{self.counter + 1}.pt"))
        vis_images(torch.abs(pred), torch.angle(pred), torch.abs(error_img), torch.angle(error_img),
                   if_save=True, save_dir=self.params["save_dir"], filename=f"screenshot_epoch_{self.counter + 1}.png", 
                   titles=["mag", "phase", "error mag", "error phase"])


class Train2DTimeCallack(Callback):
    """
    Saves reconstructed images. Also used for Train2DTimeExplicitReg
    """
    def __init__(self, params: dict):
        """
        params: save_dir, save_interval
        """
        super().__init__()
        self.params = params
        self.counter = -1
        self.params["save_dir"] = os.path.join(self.params["save_dir"], "screenshots/")
        if not os.path.isdir(self.params["save_dir"]):
            os.makedirs(self.params["save_dir"])
    
    @torch.no_grad()
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.counter += 1
        trainer.datamodule.resample()

        if self.counter % self.params["save_interval"] != 0 and self.counter != trainer.max_epochs - 1:
            return
        # preds = trainer.predict(pl_module, datamodule=trainer.datamodule)  # [(B, H, W)...]
        preds = []

        all_models = [pl_module.siren, pl_module.grid_sample]
        for model_iter in all_models:
            model_iter.to(ptu.DEVICE)
            model_iter.eval()
        if_train_dict = {model_iter: model_iter.training for model_iter in all_models}

        for batch in trainer.datamodule.predict_dataloader():
            # T, H, W = pl_module.in_shape
            x_s = batch  # (B, H, W, 3)
            x_s = x_s.to(ptu.DEVICE)
            pred_s = pl_module.predict_step(x_s, None)
            preds.append(pred_s)

        pred = pl_module.pred2vol(preds).unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
        torch.save(pred.detach().cpu(), os.path.join(self.params["save_dir"], f"recons_{self.counter + 1}.pt"))
        save_vol_as_gif(torch.abs(pred), save_dir=self.params["save_dir"], filename=f"mag_{self.counter + 1}.gif")
        save_vol_as_gif(torch.angle(pred), save_dir=self.params["save_dir"], filename=f"phase_{self.counter + 1}.gif")

        low_res = pl_module.grid_sample.get_low_res().detach().cpu()  # (1, 1, T0, H0, W0)
        low_res = low_res.squeeze().unsqueeze(1)  # (T0, 1, H0, W0)
        duration = pred.shape[0]  # set duration equal to "pred"
        save_vol_as_gif(torch.abs(low_res), save_dir=self.params["save_dir"], filename=f"mag_low_res_{self.counter + 1}.gif", duration=duration)
        save_vol_as_gif(torch.angle(low_res), save_dir=self.params["save_dir"], filename=f"phase_low_res_{self.counter + 1}.gif", duration=duration)

        for model_iter in all_models:
            if if_train_dict[model_iter]:
                model_iter.train()
            else:
                model_iter.eval()


class Train2DTimeRegCallack(Callback):
    """
    Saves reconstructed images.
    """
    def __init__(self, params: dict):
        """
        params: save_dir, save_interval
        """
        super().__init__()
        self.params = params
        self.counter = -1
        self.params["save_dir"] = os.path.join(self.params["save_dir"], "screenshots/")
        if not os.path.isdir(self.params["save_dir"]):
            os.makedirs(self.params["save_dir"])
    
    @torch.no_grad()
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.counter += 1
        if self.counter % self.params["save_interval"] != 0 and self.counter != trainer.max_epochs - 1:
            return
        # preds = trainer.predict(pl_module, datamodule=trainer.datamodule)  # [(B, H, W)...]
        preds = []
        siren = pl_module.siren.to(ptu.DEVICE)
        siren.eval()

        for batch in trainer.datamodule.predict_dataloader():
            Lambda, T, H, W = pl_module.in_shape
            x_s = batch
            x_s = x_s.to(ptu.DEVICE)
            x_s = rearrange(x_s, "B (H W) D -> B H W D", H=H)  # (B', H, W, 4)
            pred_siren = siren(x_s).squeeze(-1)  # (B', H, W)
            pred_s = pred_siren

            # pred_s = pl_module.predict_step(batch, None)
            preds.append(pred_s)

        preds = pl_module.pred2vol(preds)  # (Lambda, T, H, W)
        save_dir = os.path.join(self.params["save_dir"], f"epoch_{self.counter}")
        pl_module.save_preds(preds, save_dir)


class TrainLIIFCallback(Callback):
    """
    Selects one test image and saves the reconstruction.
    """
    def __init__(self, params: dict):
        """
        params: save_dir, save_interval, test_idx, temporal_res: int = 50
        """
        super().__init__()
        self.params = params
        self.counter = -1
        self.params["save_dir"] = os.path.join(self.params["save_dir"], "screenshots/")
        self.t_coord = torch.linspace(-1, 1, self.params["temporal_res"])
        if not os.path.isdir(self.params["save_dir"]):
            os.makedirs(self.params["save_dir"])
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.counter += 1
        img_test, measurement_test = trainer.datamodule.test_ds[self.params["test_idx"]]  # (T, H, W), (T, H, W, num_sens)
        if self.counter == 0:
            img_test_save = img_test.unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
            torch.save(img_test_save.detach().cpu(), os.path.join(self.params["save_dir"], f"orig.pt"))
            save_vol_as_gif(torch.abs(img_test_save), save_dir=self.params["save_dir"], filename=f"orig_mag.gif")
            save_vol_as_gif(torch.angle(img_test_save), save_dir=self.params["save_dir"], filename=f"orig_phase.gif")

            img_zf = trainer.datamodule.train_ds.lin_tfm.conj_op(measurement_test).unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
            torch.save(img_zf.detach().cpu(), os.path.join(self.params["save_dir"], f"zf.pt"))
            save_vol_as_gif(torch.abs(img_zf), save_dir=self.params["save_dir"], filename=f"zf_mag.gif")
            save_vol_as_gif(torch.angle(img_zf), save_dir=self.params["save_dir"], filename=f"zf_phase.gif")
        
        if self.counter % self.params["save_interval"] != 0 and self.counter != trainer.max_epochs - 1:
            return
        
        img_test, measurement_test = img_test.unsqueeze(0), measurement_test.unsqueeze(0)  # (1, T, H, W), (1, T, H, W, num_sens)
        if_train = pl_module.model.training
        pred = pl_module.predict_step(measurement_test.to(ptu.DEVICE), None).squeeze(0)  # (1, T, H, W) -> (T, H, W)
        pred = pred.unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
        torch.save(pred.detach().cpu(), os.path.join(self.params["save_dir"], f"recons_{self.counter + 1}.pt"))
        save_vol_as_gif(torch.abs(pred), save_dir=self.params["save_dir"], filename=f"mag_{self.counter + 1}.gif")
        save_vol_as_gif(torch.angle(pred), save_dir=self.params["save_dir"], filename=f"phase_{self.counter + 1}.gif")

        if if_train:
            pl_module.model.train()
        else:
            pl_module.model.eval()
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self.on_train_epoch_end(trainer, pl_module)


class TrainLIIF3DConvCallback(Callback):
    def __init__(self, params: dict):
        """
        params: save_dir, save_interval, test_idx = -1, upsample_rates: Sequence[float], roi_size: int, overlap: float
        """
        super().__init__()
        self.params = params
        self.counter = -1
        self.params["save_dir"] = os.path.join(self.params["save_dir"], "screenshots/")
        if not os.path.isdir(self.params["save_dir"]):
            os.makedirs(self.params["save_dir"])
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        torch.set_grad_enabled(True)
        pl_module.model.to(ptu.DEVICE)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,) -> None:
        torch.set_grad_enabled(False)
        pl_module.model.to(ptu.DEVICE)
        if_training = pl_module.training
        pl_module.model.eval()

        self.counter += 1
        data_dict = trainer.datamodule.test_ds[self.params["test_idx"]]
        for key, val in data_dict.items():
            data_dict[key] = val.to(ptu.DEVICE)

        if self.counter == 0:
            img_test = data_dict[IMAGE_KEY]  # (T, H, W)
            img_zf = data_dict[ZF_KEY]
            
            img_test = img_test.unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
            torch.save(img_test.detach().cpu(), os.path.join(self.params["save_dir"], f"orig.pt"))
            save_vol_as_gif(torch.abs(img_test), save_dir=self.params["save_dir"], filename=f"orig_mag.gif")
            save_vol_as_gif(torch.angle(img_test), save_dir=self.params["save_dir"], filename=f"orig_phase.gif")

            img_zf = img_zf.unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)
            torch.save(img_zf.detach().cpu(), os.path.join(self.params["save_dir"], f"zf.pt"))
            save_vol_as_gif(torch.abs(img_zf), save_dir=self.params["save_dir"], filename=f"zf_mag.gif")
            save_vol_as_gif(torch.angle(img_zf), save_dir=self.params["save_dir"], filename=f"zf_phase.gif")
        
        if self.counter % self.params["save_interval"] != 0 and self.counter != trainer.max_epochs - 1:
            return
        
        batch = {
            IMAGE_KEY: data_dict[IMAGE_KEY].unsqueeze(0),  # (T, H, W) -> (1, T, H, W),
            ZF_KEY: data_dict[ZF_KEY].unsqueeze(0)
        }
        predict_step_params = {
            "roi_size": self.params["roi_size"],
            "overlap": self.params["overlap"]
        }

        for upsample_rate_iter in self.params["upsample_rates"]:
            predict_step_params["upsample_rate"] = upsample_rate_iter
            pred, error_val = pl_module.predict_step(batch, None, **predict_step_params)  # (1, T, H, W), (1,)
            pred = pred.transpose(0, 1)  # (1, T, H, W) -> (T, 1, H, W)
            error_val = [-1.] if error_val is None else error_val
            error_val = error_val[0]
            torch.save(pred.detach().cpu(), os.path.join(self.params["save_dir"], f"recons_{self.counter + 1}_upsample_{upsample_rate_iter: .1f}_error_{error_val: .4f}.pt"))
            save_vol_as_gif(torch.abs(pred), save_dir=self.params["save_dir"], filename=f"mag_{self.counter + 1}_upsample_{upsample_rate_iter: .1f}.gif")
            save_vol_as_gif(torch.angle(pred), save_dir=self.params["save_dir"], filename=f"phase_{self.counter + 1}_upsample_{upsample_rate_iter: .1f}.gif")

        data_dict = trainer.datamodule.val_ds[self.params["test_idx"]]
        batch = {key: val.unsqueeze(0).to(ptu.DEVICE) for key, val in data_dict.items()}
        loss, pred = pl_module.shared_step(batch, None)  # (1, T, H, W)
        pred = pred.transpose(0, 1)
        torch.save(pred.detach().cpu(), os.path.join(self.params["save_dir"], f"recons_{self.counter + 1}_val_error_{loss: .4f}.pt"))
        save_vol_as_gif(torch.abs(pred), save_dir=self.params["save_dir"], filename=f"mag_{self.counter + 1}_val.gif")
        save_vol_as_gif(torch.angle(pred), save_dir=self.params["save_dir"], filename=f"phase_{self.counter + 1}_val.gif")

        if if_training:
            pl_module.model.train()


    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        return self.on_train_epoch_end(trainer, pl_module)


class TrainLIIF3DConvDebugCallback(Callback):
    def __init__(self, params: dict):
        """
        params: save_dir
        """
        super().__init__()
        self.params = params
        self.params["save_dir"] = os.path.join(self.params["save_dir"], "debug_dir/")
        if not os.path.isdir(self.params["save_dir"]):
            os.makedirs(self.params["save_dir"])
    
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: Union[int, None] = 0) -> None:
        torch.set_grad_enabled(False)
        if_training = pl_module.model.training
        pl_module.model.eval()

        img = batch[IMAGE_KEY]  # (B, T0, H, W)
        img_zf = batch[ZF_KEY]
        t_coord = batch[COORD_KEY]  # (B, T0)
        img_pred, intermediates_dict = pl_module.model(img_zf, t_coord, if_debug=True)  # (B, T0, H, W)

        save_vol_as_gif(torch.abs(img.unsqueeze(2)[0]), save_dir=self.params["save_dir"], filename=f"mag_img.gif")
        save_vol_as_gif(torch.abs(img_zf.unsqueeze(2)[0]), save_dir=self.params["save_dir"], filename=f"mag_img_zf.gif")
        save_vol_as_gif(torch.abs(img_pred.unsqueeze(2)[0]), save_dir=self.params["save_dir"], filename=f"mag_img_pred.gif")

        save_vol_as_gif(intermediates_dict["x_enc"][0][0, None, :, 0, ...].transpose(0, 1),
                        save_dir=self.params["save_dir"], filename=f"x_enc.gif")  # (B, C, T, H, W) -> (1, C, H, W) -> (C, 1, H, W)
        
        for idx, pred_iter in enumerate(intermediates_dict["pred"]):
            # (B, T, H, W) -> (B, T, 1, H, W) -> (T, 1, H, W)
            save_vol_as_gif(torch.abs(pred_iter.unsqueeze(2)[0]), save_dir=self.params["save_dir"], filename=f"mag_img_pred_{idx}.gif")
        
        for idx, q_feats_iter in enumerate(intermediates_dict["q_feats"]):
            # (B, C, T, H, W) -> (B, 1, C, H, W) -> (1, C, H, W) -> (C, 1, H, W)
            save_vol_as_gif(torch.abs(q_feats_iter[0, None, :, 0, ...].transpose(0, 1)), save_dir=self.params["save_dir"], filename=f"mag_q_feats_{idx}.gif")
        
        with open(os.path.join(self.params["save_dir"], "log.txt"), "w") as wf:
            key = "feat_coord"
            print(f"{key}", file=wf)
            print(intermediates_dict[key][0][0, 0], file=wf)  # (T0,)
            print("-" * 200, file=wf)
            
            key = "t_coord"
            for idx, ele_iter in enumerate(intermediates_dict[key]):
                print(f"{key}_{idx}", file=wf)
                print(ele_iter[0, :], file=wf)  # (T',)
            print("-" * 200, file=wf)

            key = "t_coord_"
            for idx, ele_iter in enumerate(intermediates_dict[key]):
                print(f"{key}_{idx}", file=wf)
                print(ele_iter[0, :, 0], file=wf)  # (T',)
            print("-" * 200, file=wf)
            
            key = "q_coord"
            for idx, ele_iter in enumerate(intermediates_dict[key]):
                print(f"{key}_{idx}", file=wf)
                print(ele_iter[0, :], file=wf)  # (T',)
            print("-" * 200, file=wf)
            
            key = "rel_coord"
            for idx, ele_iter in enumerate(intermediates_dict[key]):
                print(f"{key}_{idx}", file=wf)
                print(ele_iter[0, :, 0, 0, 0], file=wf)  # (T',)
            print("-" * 200, file=wf)

        if if_training:
            pl_module.model.train()
        torch.set_grad_enabled(True)

        user_input = input("If continue? Y/N: ")
        if user_input.upper() == "N":
            sys.exit(1)
