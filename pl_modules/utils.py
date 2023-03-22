import torch.nn as nn

from ImplicitNeuralRepr.configs import OPTIMIZER_MAP, SCHEDULER_MAP


def load_optimizer(opt_dict: dict, model: nn.Module):
    """
    Load optimizer and scheduler. Refer to 2d_time.yml for standard config.
    """
    opt_name = opt_dict["opt_name"]
    opt_ctor = OPTIMIZER_MAP[opt_name]
    opt = opt_ctor(model.parameters(), **opt_dict["opt_params"])
    scheduler = None
    scheduler_name = opt_dict.get("scheduler_name", None)
    if scheduler_name is not None:
        scheduler_ctor = SCHEDULER_MAP[scheduler_name]
        scheduler = scheduler_ctor(opt, **opt_dict["scheduler_params"])
    
    return opt, scheduler
