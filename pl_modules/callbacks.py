import torch
import pytorch_lightning as pl
import os
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from pytorch_lightning.callbacks import Callback
from ImplicitNeuralRepr.utils.utils import vis_images, save_vol_as_gif
from einops import rearrange


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
