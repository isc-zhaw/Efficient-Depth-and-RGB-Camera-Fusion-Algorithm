
import os
import torch
import numpy as np
import cv2
from torch.nn.functional import interpolate

from dkn.models import *



class Dkn:
    def __init__(self, model='FDKN', kernel_size=3, filter_size=15):
        if model == 'FDKN':
          self.model = FDKN(kernel_size=kernel_size, filter_size=filter_size, residual=True).cuda()
          self.resume(path='dkn/parameter/FDKN_8x')
        elif model == 'DKN':
          self.model = DKN(kernel_size=kernel_size, filter_size=filter_size, residual=True).cuda()
          self.resume(path='dkn/parameter/DKN_8x')
        self.model.eval()

    def __call__(self, rgb, depth):
        scale = int(rgb.shape[0] / depth.shape[0])
        img_max = np.max(depth)
                
        depth = (depth.astype(float) / img_max)
        lr_up = cv2.resize(depth, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_CUBIC)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda().float()
        lr_up = torch.from_numpy(lr_up).unsqueeze(0).unsqueeze(0).cuda().float()
    
        rgb = torch.from_numpy(np.transpose(rgb, (2, 0, 1)) / 255).unsqueeze(0).cuda().float()
        with torch.no_grad():
          out_img = self.model((rgb, lr_up))
        
        out_img = out_img.detach().squeeze().cpu().numpy()
        out_img = (out_img * img_max).round().astype(np.float32)
        
        return out_img, rgb

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
        print(f'Checkpoint \'{path}\' loaded.')

