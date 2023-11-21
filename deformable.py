import torch
import torch.nn as nn
import torch.nn.functional as F

class NLBlockND_cross(nn.Module):
    # Our implementation of the attention block referenced https://github.com/tea1528/Non-Local-NN-Pytorch
    # 这一块儿就是夸模态注意力模块
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND_cross, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        # 为非局部块的最后一个卷积层添加批量归一化，以提高模型的训练稳定性，并确保初始状态是一个恒等映射
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        #x_thisBranch for g and theta
        #x_otherBranch for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        # print(x_thisBranch.shape)

        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else: #default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch 

        return z


class deformable(nn.Module):
    def __init__(self):
        super(deformable, self).__init__()
        # Cross-Modal Attention 模块里面经过Cross Attention之后的操作
        self.relu_after_cross = nn.ReLU(inplace=True)
        self.batchnorm_after_cross = nn.BatchNorm3d(64)
        self.convsize1_after_cross = nn.Conv3d(in_channels=64,
                                out_channels=32,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        
        # 一个完整的block块：
        self.ConvBlock1 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            # # 池化：注意这里，论文里面是222，但是这里用的是333
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        )
        self.ConvBlock2 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            # # 池化：注意这里，论文里面是222，但是这里用的是333 
            # # 严格来说，这一个并不属于Block，先注释掉吧
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        )
        self.ConvBlock3 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            # # 池化：注意这里，论文里面是222，但是这里用的是333
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        )
        # 最大池化kernelsize是3，3，3，虽然论文里面说的是2，2，2
        # 但是实际上好像有一定的不同，这里就暂时按照论文说的
        self.MaxPooling1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,2,2), padding=0)
        self.MaxPooling2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)
        self.MaxPooling3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)

        self.Conv_size1_for_cat =nn.Conv3d(in_channels=64,
            out_channels=32,
            kernel_size=1,
            stride=(1, 1, 1),
            padding=0,# padding=1 还是会改变输入图像的大小的
            bias=False)

        self.ConvBlock4 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
            # # 池化：注意这里，论文里面是222，但是这里用的是333
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        )
        self.ConvBlock5 = nn.Sequential(
            # 第一层：
            nn.Conv3d(in_channels=64,
            out_channels=16,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            # BatchNorm + ReLU
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # 第二层：
            nn.Conv3d(in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
            # # 池化：注意这里，论文里面是222，但是这里用的是333
            # nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        )

        self.Final_Conv =nn.Conv3d(in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)        

        self.decoder1 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？先用转置卷积
            nn.ConvTranspose3d(in_channels=32, 
                               out_channels=32, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, 
                               padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？
            nn.ConvTranspose3d(in_channels=32, 
                               out_channels=32, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, # 在第二个里面是不是要给它放大四倍？因为前面是三个池化 不可以，因为要拼接
                               padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            # deconvolution:
            # voxelmorph里面用的是上采样最近邻插值，我们用哪一个？
            nn.ConvTranspose3d(in_channels=16, 
                               out_channels=32, 
                               kernel_size=4, # kernel_size=2的话padding=0
                               stride=2, # 在第二个里面是不是要给它放大四倍？因为前面是三个池化 不可以，因为要拼接
                               padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.Cross_attention = NLBlockND_cross(32) # NLBlockND_cross并不会改变通道数，输出还是32通道
    def forward(self, x):
        # 这一步操作是将x的两个通道的数据分别取出来，x是一个5D的数据:(N,C,T,H,W)

        fixed_0 = torch.unsqueeze(x[:, 0, :, :, :], 1) # 1,1,64,256,256
        moving_0 = torch.unsqueeze(x[:, 1, :, :, :], 1) # 1,1,64,256,256

        # 这里写的有问题，不能这样无脑套，因为有一些中间的结果被拿出来了，然后去和其他的做了拼接，
        # 所以要在每一代都创建一个新的变量
        fixed_1      = self.ConvBlock1(fixed_0) # 1,1,64,256,256 -> 1,32,64,256,256
        fixed_1_pool = self.MaxPooling1(fixed_1) # 1,32,64,256,256 -> 1, 32, 32, 128, 128
        fixed_2      = self.ConvBlock2(fixed_1_pool) # 1, 32, 32, 128, 128->1, 32, 32, 128, 128
        fixed_2_pool = self.MaxPooling2(fixed_2) # 1,32,32,128,128 -> 1, 32, 16, 64, 64
        fixed_3      = self.ConvBlock3(fixed_2_pool) # 1, 32, 16, 64, 64->1, 32, 16, 64, 64
        fixed_3_pool = self.MaxPooling3(fixed_3) # 1, 32, 16, 64, 64 -> 1, 32, 8, 32, 32
        fixed_cross_input = fixed_3_pool

        moving_1      = self.ConvBlock1(moving_0)
        moving_1_pool = self.MaxPooling1(moving_1)
        moving_2      = self.ConvBlock2(moving_1_pool)
        moving_2_pool = self.MaxPooling2(moving_2)
        moving_3      = self.ConvBlock3(moving_2_pool)
        moving_3_pool = self.MaxPooling3(moving_3)
        moving_cross_input = moving_3_pool
        # moving_cross_output = self.Cross_attention(moving_cross_input)
        # 运行了这个cross_attention之后，得到32channel的方块
        # ((1, 32, 8, 32, 32),(1, 32, 8, 32, 32))->(1, 32, 8, 32, 32)
        fixed_cross_output = self.Cross_attention(fixed_cross_input,moving_cross_input)
        # fixed_cross_output = self.relu_after_cross(fixed_cross_output)
        # 代码这里又relu一下，论文里面没有啊
        moving_cross_output = self.Cross_attention(moving_cross_input,fixed_cross_input)
        # moving_cross_output = self.relu_after_cross(moving_cross_output)
        # 这里其实就是得到那个拼接之后方块了。
        # ((1, 32, 8, 32, 32),(1, 32, 8, 32, 32)) -> (1, 64, 8, 32, 32)
        fixed_moving_cross = torch.cat((fixed_cross_output, moving_cross_output), 1)
        fixed_moving_cross = self.batchnorm_after_cross(fixed_moving_cross)

        # (1, 64, 8, 32, 32) -> (1, 32, 8, 32, 32)
        fixed_moving_cross = self.convsize1_after_cross(fixed_moving_cross)# 改变通道数，缩2
        # Cross-Modal Attention模块结束，下面进行deformable reg
        # 在进行deformable之前，先给下面的部分给做了,编码器的特征拼接
        cat1 = torch.cat((fixed_1,moving_1),1)# ((1,32,64,256,256),(1,32,64,256,256))->(1,64,64,256,256)
        cat1 = self.Conv_size1_for_cat(cat1) # (1,64,64,256,256)->(1,32,64,256,256)
        cat_for_deformable1 = cat1

        cat2 = torch.cat((fixed_2,moving_2),1) #1, 32, 32, 128, 128 -> 1, 64, 32, 128, 128
        cat2 = self.Conv_size1_for_cat(cat2)  # 1, 64, 32, 128, 128 -> 1, 32, 32, 128, 128
        cat_for_deformable2 = cat2

        cat3 = torch.cat((fixed_3,moving_3),1) # 1, 32, 16, 64, 64 -> 1, 64, 16, 64, 64
        cat3 = self.Conv_size1_for_cat(cat3) # 1, 64, 16, 64, 64 -> 1, 32, 16, 64, 64
        cat_for_deformable3 = cat3

        reg_input = fixed_moving_cross # (1, 32, 8, 32, 32)
        reg_input = self.relu_after_cross(reg_input)
        # 这里要进行一个反卷积
        reg_decoder1 = self.decoder1(reg_input) # (1, 32, 8, 32, 32) -> (1, 32, 16, 64, 64)
        CB4_input = torch.cat((reg_decoder1,cat_for_deformable3),1)# (1, 32, 16, 64, 64)->(1, 64, 16, 64, 64)
        ConvBlock4 = self.ConvBlock4(CB4_input)# 输入64，输出32  1, 64, 16, 64, 64 -> 1, 32, 16, 64, 64
        # 第二次反卷积
        reg_decoder2 = self.decoder2(ConvBlock4) # 输入输出都是32，图像放大两倍 1, 32, 16, 64, 64 -> 1, 32, 32, 128, 128
        CB5_input = torch.cat((reg_decoder2,cat_for_deformable2),1) # 1, 32, 32, 128, 128 -> 1, 64, 32, 128, 128
        ConvBlock5 = self.ConvBlock5(CB5_input) # 1, 64, 32, 128, 128 -> 1, 16, 32, 128, 128
        # 第三次反卷积
        # 这里第三个的通道数需要修改，有两种改法，1.只修改decoder的，2.conv5和decoder，我先选择简单的，只修改decoder
        # reg_decoder3 = self.decoder3(ConvBlock5) # 1, 16, 32, 128, 128 -> 1, 16, 64, 256, 256
        reg_decoder3 = self.decoder3(ConvBlock5) # 1, 16, 32, 128, 128 -> 1, 32, 64, 256, 256
        final_conv_input = torch.cat((reg_decoder3,cat_for_deformable1),1) 
        # (1, 32, 64, 256, 256),(1,32,64,256,256) -> ???
        output = self.Final_Conv(final_conv_input)
        
        return output
    
    