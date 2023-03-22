import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle
import ImplicitNeuralRepr.utils.pytorch_utils as ptu

from .metrics import REGISTERED_METRICS
from collections import defaultdict


def metric_vs_one_hyperparam(root_dirs: list, metrics: list, param_tune: str, param_defaults: dict,
                             orig_filename="original.pt", recons_filename="reconstructions.pt",
                             args_filename="args_dict.pkl", selection_func = None, *args, **kwargs):
    """
    kwargs: figsize_unit, if_logscale_x: bool, save_dir: str, save_filename: str, if_compute_metrics_only: bool
    metric_vals: {param: {metric: val...}...}
    """
    figsize_unit = kwargs.get("figsize_unit", 3.6)

    def selection(args_dict: dict):
        """
        Customize criterion for selecting parameters.
        """
        return True

    def dict2str(args_dict: dict):
        out_str = ""
        for i, (key, val) in enumerate(args_dict.items()):
            prefix = ", " if i > 0 else ""
            out_str += prefix + f"{key}: {val: .2e}"

        return out_str

    if selection_func is None:
        selection_func = selection

    metric_vals = defaultdict(dict)
    for root_dir_iter in root_dirs:
        print(f"current: {root_dir_iter}")
        img = torch.load(os.path.join(root_dir_iter, orig_filename))
        recons = torch.load(os.path.join(root_dir_iter, recons_filename))
        img = ptu.to_numpy(img)  # (T, C, H, W)
        recons = ptu.to_numpy(recons)[0]  # (T, C, H, W)

        with open(os.path.join(root_dir_iter, args_filename), "rb") as rf:
            args_dict = pickle.load(rf)
        if not selection_func(args_dict):
            continue
        local_metric_dict = {}
        for metric_iter in metrics:
            assert metric_iter in REGISTERED_METRICS
            metric_func = REGISTERED_METRICS[metric_iter]
            metric_vals[args_dict[param_tune]][metric_iter] = metric_func(np.abs(recons), np.abs(img))
            local_metric_dict[metric_iter] = metric_vals[args_dict[param_tune]][metric_iter]
        with open(os.path.join(root_dir_iter, "metrics.txt"), "w") as wf:
            wf.write(f"{local_metric_dict}")

    if kwargs.get("if_compute_metrics_only", False):
        return

    params = sorted(list(metric_vals.keys()))
    print(f"params:\n{params}")
    metric_dict = defaultdict(list)  # {metric: list...}
    for param in params:
        for metric_iter in metrics:
            metric_dict[metric_iter].append(metric_vals[param][metric_iter])

    fig, axes = plt.subplots(1, len(metrics), figsize=(figsize_unit * len(metrics), figsize_unit))
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics):
        metric_list = metric_dict[metric]
        print(f"{metric}:\n{metric_list}")
        axis.plot(params, metric_list)
        axis.set_xlabel(param_tune)
        axis.set_title(metric)
        axis.grid(axis="y")
        if kwargs.get("if_logscale_x", True):
            axis.set_xscale("log")

    fig.suptitle(dict2str(param_defaults))
    fig.tight_layout()

    save_dir = kwargs.get("save_dir", None)
    if save_dir is not None:
        save_filename = kwargs.get("save_filename", f"{param_tune}.png")
        assert ".png" in save_filename
        fig.savefig(os.path.join(save_dir, save_filename))

    return fig
