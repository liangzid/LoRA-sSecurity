
# ------------------------ Code --------------------------------------
import torch
import random
import numpy as np

def set_random_seed(seed, ifcuda=True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if ifcuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

