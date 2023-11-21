import torch
import torchvision.models as models
from deformable import *
from SPT import *
import numpy as np
import nibabel as nib

# 

# load and set model
model_test = deformable()
model_test.load_state_dict(torch.load('/media/user_gou/Elements/Shi/Attention-Reg-macos/draft/models_4DCT/model_epoch60.pth'))
model_test.eval()

# read fixed file
fixed_file = np.load('/media/user_gou/Elements/Shi/Attention-Reg-main/cropeddata/01_01.npy')
fixed = torch.from_numpy(fixed_file)
fixed = fixed.unsqueeze(0).unsqueeze(0) # 1,1,64,256,256
fixed = torch.tensor(fixed,dtype=torch.float32)

# read moving file
moving_file = np.load('/media/user_gou/Elements/Shi/Attention-Reg-main/cropeddata/01_02.npy')
moving = torch.from_numpy(moving_file)
moving = moving.unsqueeze(0).unsqueeze(0)
moving = torch.tensor(moving,dtype=torch.float32)

# warp
input_data = torch.cat((fixed,moving),dim=1) # 1,2,64,256,256
warp = model_test(input_data) # 1,3,64,256,256

# warp -> nii
warp = warp.squeeze(0) # 3,64,256,256
warp = warp.permute(2,3,1,0) # 256,256,64,3
affine = np.eye(4)
warp_array = warp.detach().numpy()
warp_image = nib.Nifti1Image(warp_array, affine)
nib.save(warp_image, 'warp.nii.gz')

# # # moved
# trans = SpatialTransformer((64,256,256))
# moved = trans(moving,warp) # 1,1,64,256,256
# moved = moved.squeeze(0).squeeze(0) # 256,256,64
# moved = moved.permute(1,2,0)

# # NIfTI
# moved_array = moved.detach().numpy()
# affine = np.eye(4)
# nii_image = nib.Nifti1Image(moved_array, affine)
# nib.save(nii_image, 'moved.nii.gz')