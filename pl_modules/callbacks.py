import torch
import pytorch_lightning as pl
import os
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from pytorch_lightning.callbacks import Callback
from ImplicitNeuralRepr.utils.utils import vis_images
from einops import rearrange


class TrainSpatialCallack(Callback):
    """
    Saves residual and reconstructed images.
    """
    # def __init__(self, params: dict, img_gt: torch.Tensor, dm: pl.LightningDataModule):
    def __init__(self, params: dict, img_gt: torch.Tensor):
        """
        params: save_dir, save_interval
        """
        super().__init__()
        self.params = params
        self.img_gt = img_gt  # (H, W)
        # self.dm = dm
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
