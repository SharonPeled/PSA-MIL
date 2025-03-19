import numpy as np
import torch
import random
from src.components.objects.Logger import Logger
import datetime
import pytorch_lightning as pl
from torch.multiprocessing import set_start_method, set_sharing_strategy


def set_global_configs(verbose, log_file_args, log_importance, log_format, random_seed):
    set_sharing_strategy('file_system')
    set_start_method("spawn")
    Logger.set_default_logger(verbose, log_file_args, log_importance, log_format)
    set_random_seed(random_seed)


def get_time():
    now = datetime.datetime.now()
    return now.strftime("%d-%m-%y_%H_%M_%S")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)





