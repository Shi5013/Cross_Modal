import numpy as np
from scipy.ndimage import convolve
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

# 构建sobel算子
device = torch.device("cuda:1") 
sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],
                        [[-2,0,2],[-4,0,4],[-2,0,2]],
                        [[-1,0,1],[-2,0,2],[-1,0,1]]],dtype=torch.float32)
sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
sobel_x = sobel_x.to(device)

sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]],
                        [[-2,-4,-2],[0,0,0],[2,4,2]],
                        [[-1,-2,-1],[0,0,0],[1,2,1]]],dtype=torch.float32)
sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
sobel_y = sobel_y.to(device)

sobel_z = torch.tensor([[[1,2,1],[2,4,2],[1,2,1]],
                        [[0,0,0],[0,0,0],[0,0,0]],
                        [[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]],dtype=torch.float32)
sobel_z = sobel_z.unsqueeze(0).unsqueeze(0)
sobel_z = sobel_z.to(device)

def Edge_Loss(moved,label):
    # moved和label都是五维的数组 1,1,64,256,256
    moved_grad_x = F.conv3d(moved, sobel_x, padding=1)
    moved_grad_y = F.conv3d(moved, sobel_y, padding=1)
    moved_grad_z = F.conv3d(moved, sobel_z, padding=1)

    label_grad_x = F.conv3d(label, sobel_x, padding=1)
    label_grad_y = F.conv3d(label, sobel_y, padding=1)
    label_grad_z = F.conv3d(label, sobel_z, padding=1)

    loss_x = F.l1_loss(moved_grad_x, label_grad_x)
    loss_y = F.l1_loss(moved_grad_y, label_grad_y)
    loss_z = F.l1_loss(moved_grad_z, label_grad_z)

    loss_x = torch.abs(loss_x)
    loss_y = torch.abs(loss_y)
    loss_z = torch.abs(loss_z)

    return (loss_x + loss_y + loss_z) / 3.0

def ssim_loss(moved,label):
    # 只接受np数组，所以要把tensor转化为numpy数据类型
    moved = moved.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    moved = moved.squeeze(0).squeeze(0)
    label = label.squeeze(0).squeeze(0)
    data_range = max(moved.max() - moved.min(), label.max() - label.min())
    ssi_index, _ = ssim(moved, label, full=True,data_range=data_range)
    ssi_index = 1.0-ssi_index
    ssi_index = torch.tensor(ssi_index,dtype = torch.float32)
    return ssi_index

def total_loss(moved,label):
    edge = Edge_Loss(moved,label)
    ssim = ssim_loss(moved,label)
    l1 = F.l1_loss(moved,label)
    return edge + ssim + l1

