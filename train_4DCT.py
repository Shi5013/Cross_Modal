import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from deformable import *
from SPT import *
from Edge_ssim_loss import *
import nibabel as nib


# 超参数
lr = 1e-4
batch_size = 1
epochs = 300
# criterion = torch.nn.MSELoss()
device = torch.device("cuda:1") 
save_folder = './draft/models_4DCT/'
save_prefix = 'model_'
save_interval = 20  # 每隔20个 epoch 保存一次模型

model = deformable()
model = model.to(device) # model -> GPU

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

class Fixed_Moving(Dataset):
    def __init__(self, txt_file):
        with open(txt_file, 'r') as f:
            self.file_paths = f.readlines()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path1 = self.file_paths[idx].strip()
        
        # 随机选择另一个文件
        idx2 = np.random.randint(len(self))
        path2 = self.file_paths[idx2].strip()

        # 检查是否来自同一个文件夹
        while not self.check_same_folder(path1, path2):
            # 如果不是同一个文件夹，则重新选择第二个文件
            idx2 = np.random.randint(len(self))
            path2 = self.file_paths[idx2].strip()
        # 使用nibabel加载NIfTI文件
        data1_nifti = nib.load(path1)
        data2_nifti = nib.load(path2)

        # 获取NIfTI数据的数组
        data1 = data1_nifti.get_fdata()
        data2 = data2_nifti.get_fdata()

        # 重新排列数组维度
        data1 = np.transpose(data1, (2, 0, 1))  # 将维度从 (256, 256, 96) 转换为 (96, 256, 256)
        data2 = np.transpose(data2, (2, 0, 1))
        Fixed = torch.from_numpy(data1).float()
        Moving = torch.from_numpy(data2).float()

        input_data = torch.stack((Fixed, Moving), dim=0)
        label_data = torch.unsqueeze(Fixed,dim=0).float()

        return {
            'input_data': input_data,
            'label': label_data
        }

    def check_same_folder(self, path1, path2):
        folder1 = os.path.dirname(path1)
        folder2 = os.path.dirname(path2)
        return folder1 == folder2

# 使用示例
txt_file_path = '/media/user_gou/Elements/Shi/nii_resample/list_all.txt'  # 替换为你的txt文件路径
dataset = Fixed_Moving(txt_file_path)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

for epoch in range(epochs):
    # 设置模型为训练模式
    model.train()
    losss = 0

    # 遍历数据加载器，获取每个批次的数据
    for batch in data_loader:
        # 获取输入数据和标签
        input_data = batch['input_data']
        input_data = input_data.to(device) # input_data -> GPU
        label_data = batch['label']
        label_data = label_data.to(device) # label_data -> GPU

        # 将梯度归零
        optimizer.zero_grad()

        # 前向传播
        output = model(input_data) # (1,3,96,256,256)

        # 计算损失
        trans = SpatialTransformer((96,256,256))
        moving = torch.unsqueeze(input_data[:, 1, :, :, :], 1)
        # moving = moving.to(device)
        moved = trans(moving,output) # 这里的moving是input的第二个维度
        # loss = criterion(moved, label_data) # MSE
        loss = total_loss(moved,label_data) # l1 + sobel + ssim
        # loss = Edge_Loss(moved,label_data) # sobel
        losss = losss + loss.item()

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()
    if (epoch + 1) % save_interval == 0:
        # 构建保存路径，包含有关模型和训练的信息
        save_path = f"{save_folder}{save_prefix}epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Model saved as {save_path}")

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {losss}")

