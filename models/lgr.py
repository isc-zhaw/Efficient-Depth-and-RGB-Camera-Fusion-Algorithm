
import sys
sys.path.insert(0, 'graph_super_resolution')
import os

import numpy as np
import torch
import cv2

from graph_super_resolution.model.graph_sr_net import GraphSuperResolutionNet
from torch import nn


class Lgr:
    def __init__(self, scale: int, crop_size: int = 128):
        self.crop_size = crop_size
        self.model = GraphSuperResolutionNet(scaling=scale, crop_size=self.crop_size, feature_extractor='Color')

    def __call__(self, rgb, depth):
        upscaling_factor = int(rgb.shape[0] / depth.shape[0])
        depth_out = np.zeros((rgb.shape[0], rgb.shape[1]))

        depth = depth.astype(float)
        lr_up = cv2.resize(depth, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_CUBIC)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda().float()
        lr_up = torch.from_numpy(lr_up).unsqueeze(0).unsqueeze(0).cuda().float()

        rgb = torch.from_numpy(np.transpose(rgb, (2, 0, 1)) / 255).unsqueeze(0).cuda().float()
        
        rgb_patch_size = self.crop_size
        depth_patch_size = rgb_patch_size // upscaling_factor
        x_patches = np.ceil(rgb.shape[2] / rgb_patch_size)
        y_patches = np.ceil(rgb.shape[3] / rgb_patch_size)

        for x in range(int(x_patches-1)):
            for y in range(int(y_patches-1)):
                rgb_patch = rgb[:, :, x*rgb_patch_size:(x+1)*rgb_patch_size, y*rgb_patch_size:(y+1)*rgb_patch_size]
                depth_patch = depth[:, :, x*depth_patch_size:(x+1)*depth_patch_size, y*depth_patch_size:(y+1)*depth_patch_size]
                lr_up_patch = lr_up[:, :, x*rgb_patch_size:(x+1)*rgb_patch_size, y*rgb_patch_size:(y+1)*rgb_patch_size]
                sample = {'guide': rgb_patch, 'source': depth_patch, 'mask_lr': torch.ones_like(depth_patch), 'y_bicubic': lr_up_patch}
                with torch.no_grad():
                    out = self.model(sample)['y_pred']
                depth_out[x*rgb_patch_size:(x+1)*rgb_patch_size, y*rgb_patch_size:(y+1)*rgb_patch_size] = out.detach().squeeze().cpu().numpy()

        for x in range(int(x_patches-1)):
            rgb_patch = rgb[:, :, x*rgb_patch_size:(x+1)*rgb_patch_size, -rgb_patch_size:]
            depth_patch = depth[:, :, x*depth_patch_size:(x+1)*depth_patch_size, -depth_patch_size:]
            lr_up_patch = lr_up[:, :, x*rgb_patch_size:(x+1)*rgb_patch_size, -rgb_patch_size:]
            sample = {'guide': rgb_patch, 'source': depth_patch, 'mask_lr': torch.ones_like(depth_patch), 'y_bicubic': lr_up_patch}
            with torch.no_grad():
              out = self.model(sample)['y_pred']
            depth_out[x*rgb_patch_size:(x+1)*rgb_patch_size, -rgb_patch_size:] = out.detach().squeeze().cpu().numpy()

        for y in range(int(y_patches-1)):
            rgb_patch = rgb[:, :, -rgb_patch_size:, y*rgb_patch_size:(y+1)*rgb_patch_size]
            depth_patch = depth[:, :, -depth_patch_size:, y*depth_patch_size:(y+1)*depth_patch_size]
            lr_up_patch = lr_up[:, :, -rgb_patch_size:, y*rgb_patch_size:(y+1)*rgb_patch_size]
            sample = {'guide': rgb_patch, 'source': depth_patch, 'mask_lr': torch.ones_like(depth_patch), 'y_bicubic': lr_up_patch}
            with torch.no_grad():
              out = self.model(sample)['y_pred']
            depth_out[-rgb_patch_size:, y*rgb_patch_size:(y+1)*rgb_patch_size] = out.detach().squeeze().cpu().numpy()

        rgb_patch = rgb[:, :, -rgb_patch_size:, -rgb_patch_size:]
        depth_patch = depth[:, :, -depth_patch_size:, -depth_patch_size:]
        lr_up_patch = lr_up[:, :, -rgb_patch_size:, -rgb_patch_size:]
        sample = {'guide': rgb_patch, 'source': depth_patch, 'mask_lr': torch.ones_like(depth_patch), 'y_bicubic': lr_up_patch}
        with torch.no_grad():
          out = self.model(sample)['y_pred']
        depth_out[-rgb_patch_size:, -rgb_patch_size:] = out.detach().squeeze().cpu().numpy()

        return depth_out, rgb



