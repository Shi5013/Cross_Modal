import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from deformable import *
from SPT import *
from Edge_ssim_loss import *

# 超参数
lr = 1e-4
batch_size = 1
epochs = 300
# criterion = torch.nn.MSELoss()
device = torch.device("cuda:1") 
save_folder = './draft/models_random/'
save_prefix = 'model_'
save_interval = 20  # 每隔20个 epoch 保存一次模型

model = deformable()
model = model.to(device) # model -> GPU

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


class Fixed_Moving(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)
        np.random.shuffle(self.file_list) # 在这里随机

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 随机选择两个文件
        # file1, file2 = np.random.choice(self.file_list, size=2, replace=False)
        
        # 顺序选择两个文件
        file1, file2 = self.file_list[idx], self.file_list[(idx + 1) % len(self.file_list)]
        # 从文件加载数据并转换为张量
        Fixed = torch.from_numpy(np.load(os.path.join(self.data_folder, file1))).float()
        Moving = torch.from_numpy(np.load(os.path.join(self.data_folder, file2))).float()
        # unsqueezed_data1 = torch.unsqueeze(Fixed, dim=0)
        # unsqueezed_data2 = torch.unsqueeze(Moving, dim=0)

        # stacked_data = torch.cat((unsqueezed_data1, unsqueezed_data2), dim=0)

        # 在 dim=0 维度上连接两个张量
        input = torch.stack((Fixed, Moving), dim=0) 
        # 这拼完后其实是个四维的数据
        label_data = torch.unsqueeze(Fixed,dim=0).float()

        return {
            'input_data': input,
            'label': label_data
        }

# 示例用法
data_folder_path = '/media/user_gou/Elements/Shi/Attention-Reg-main/cropeddata'
dataset = Fixed_Moving(data_folder_path)
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
        output = model(input_data) # (1,3,64,256,256)

        # 计算损失
        trans = SpatialTransformer((64,256,256))
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

