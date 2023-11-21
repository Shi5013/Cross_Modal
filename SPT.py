import torch
import torch.nn as nn
import torch.nn.functional as nnf

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0) # 增加一个维度 
        grid = grid.type(torch.FloatTensor)
        # train:
        # grid = grid.to("cuda:1")


        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):  
        # new locations
        new_locs = self.grid + flow # 逐元素相加 torch.Size([1, 3, (size)])
        shape = flow.shape[2:] # 这是又返回原始图像的尺寸

        # need to normalize grid values to [-1, 1] for resampler
        # 标准化到[-1,1]之间
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        # 这尼玛的作者也不知道为啥，就转换一下维度。。
        # grid_sample 函数的要求是 (batch_size, height, width, N)，
        # 其中 N 表示坐标的维度。
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

# 这个SPT的核心就是nnf.grid_sample() , 难点就是张量维度的变换

# size = (16,32,32)
# trans = SpatialTransformer(size)

# srcimg = torch.randn(1,1,16,16,32)
# flow_field = torch.randn(1,3,16,32,32)

# result = trans(srcimg,flow_field)
# print(result.size())

# 可以，最终还是跑通了。输出的结果会和scrimg的维度一样。