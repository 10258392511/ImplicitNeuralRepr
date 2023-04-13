import torch
import torch.nn as nn
import abc
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Dict, Any


class BaseTrainer(abc.ABCMeta):
    def __init__(self, params: dict, model_dict: Dict[str, nn.Module], config: dict):
        """
        params: num_epochs, if_notebook, log_dir
        logging:
        + log_dir
            + time_stamp
                - desc.txt (to be saved via scripts)
                - event.out
                - ckpt.pt
        """
        assert "num_epochs" in params
        assert "log_dir" in params
        self.params = params
        if_notebook = self.params.get("if_notebook", False)
        if if_notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        self.models = model_dict
        self.config = config
        self.opt_configs = self.configure_optimizers()
        self.time_stamp = self.generate_timestamp()
        log_dir = os.path.join(self.params["log_dir"], self.time_stamp)
        self.logger = SummaryWriter(log_dir)
        self.callback_states = {"epoch_metrics": None, "epoch": 0}

    def train(self, *args, **kwargs):
        assert self.opt_configs is not None
        pbar = trange(self.params["num_epochs"], leave=True)
        for epoch in pbar:
            for model_iter in self.models.values():
                model_iter.train()
            
            epoch_metrics = self.train_epoch()
            self.callback_states["epoch_metrics"] = epoch_metrics
            self.callback_states["epoch"] = epoch

            for opt_config_iter in self.configure_optimizers:
                scheduler = opt_config_iter.get("lr_scheduler", None)
                if scheduler is not None:
                    scheduler.step()
            
            for tag, val in epoch_metrics.items():
                self.logger.add_scalar(tag, val, epoch)
            desc = self.dict2str(epoch_metrics)
            pbar.set_description(desc)

            self.callbacks()

    @abc.abstractmethod
    def train_epoch(self, *args, **kwargs) -> dict:
        """
        - Iterate through multiple DataLoaders
        - Backward propagation with optimizers.
        - Logging

        returns: dict of epoch metrics to be logged
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def callbacks(self) -> None:
        """
        Read only. Calls multiple callbacks, e.g. save ckpt, save visualizations &etc.
        """
        for model_iter in self.models.values():
            model_iter.eval()
    
    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Any:
        for model_iter in self.models.values():
            model_iter.eval()
        return None

    def configure_optimizers(self) -> Dict[str, dict]:
        # str: model.__repr__()
        return None
    
    def save_models(self, save_path: str):
        assert ".pt" in save_path
        ckpt_dict = {key: self.models[key].state_dict() for key in self.models}
        torch.save(ckpt_dict, save_path)
    
    def load_models(self, ckpt_path: str):
        """
        model_dict: models already sent to DEVICE
        """
        assert ".pt" in ckpt_path
        ckpt_dict = torch.load(ckpt_path)
        for model_repr, model_iter in self.models.items():
            assert model_repr in ckpt_dict
            model_iter.load_state_dict(ckpt_dict[model_repr])
            model_iter.eval()

    @staticmethod
    def generate_timestamp():
        time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        return time_stamp
    
    @staticmethod
    def dict2str(metric_dict: dict):
        out_str = ""
        for key, val in metric_dict.items():
            out_str += f"{key}: {val: .3f}, "
        
        return out_str[:-2]
    