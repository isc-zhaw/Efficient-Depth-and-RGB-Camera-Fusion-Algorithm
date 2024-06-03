
import sys
sys.path.insert(0, 'AHMF')
import os

import numpy as np
import torch
import cv2

from AHMF.ahmf import AHMF
from torch import nn


class Ahmf:
    def __init__(self, checkpoint_path, scale=16):
        self.model = AHMF(scale=scale)
        self.model = nn.DataParallel(self.model.cuda())
        self.resume(path=checkpoint_path)
        self.model.cuda().eval()

    def __call__(self, rgb, depth):
        img_max = np.max(depth)

        depth = (depth.astype(float) / img_max)
        lr_up = cv2.resize(depth, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_CUBIC)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda().float()
        lr_up = torch.from_numpy(lr_up).unsqueeze(0).unsqueeze(0).cuda().float()
    
        rgb = torch.from_numpy(np.transpose(rgb, (2, 0, 1)) / 255).unsqueeze(0).cuda().float()
        with torch.no_grad():
            out_img = self.model(lr=depth, rgb=rgb, lr_up=lr_up)[0]
        
        out_img = out_img.detach().squeeze().cpu().numpy()
        out_img = (out_img * img_max).round().astype(np.float32)
        
        return out_img, rgb

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state'], strict=False)
        print(f'Checkpoint \'{path}\' loaded.')



