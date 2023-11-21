import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from deformable import *
from SPT import *
from Edge_ssim_loss import *

fixed = np.load('/media/user_gou/Elements/Shi/Attention-Reg-main/cropeddata/01_01.npy')
moving = np.load('/media/user_gou/Elements/Shi/Attention-Reg-main/cropeddata/02_01.npy')

fixed = torch.tensor(fixed,dtype=torch.float64).unsqueeze(0).unsqueeze(0)
moving = torch.tensor(moving,dtype=torch.float64).unsqueeze(0).unsqueeze(0)

# result = total_loss(fixed,moving) # tensor(1.0539, dtype=torch.float64)
# result = Edge_Loss(fixed,moving) # tensor(0.0892, dtype=torch.float64)
result = ssim_loss(fixed,moving) # tensor(0.0424, dtype=torch.float64)
# result = F.l1_loss(fixed,moving) # tensor(0.0071, dtype=torch.float64)
print(result)