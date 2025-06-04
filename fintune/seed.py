import random
import numpy as np
import torch
import os
import accelerate
from accelerate import Accelerator

def set_seeds(seed: int, deterministic: bool = False) -> int:
    if hasattr(Accelerator, "_shared_state") and Accelerator._shared_state:
        accelerate.utils.set_seed(seed)  # Accelerate 会自动处理多进程种子
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return seed

    # 常规单机种子设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU适配
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed