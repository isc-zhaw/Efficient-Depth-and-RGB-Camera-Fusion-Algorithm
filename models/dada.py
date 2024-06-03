
import sys
sys.path.insert(0, 'diffusion_super_resolution')
import os

import numpy as np
import torch
import cv2

from model import GADBase
from utils import to_cuda

sys.path.remove('diffusion_super_resolution')

class Dada:

    def __init__(self, checkpoint_path, feature_extractor='UNet', Npre=8000, Ntrain = 1024):
        self.model = GADBase(feature_extractor, Npre=Npre, Ntrain=Ntrain)
        self.resume(path=checkpoint_path)
        self.model.cuda().eval()

    def __call__(self, rgb, tof):
        self.model.eval()

        upscaled_depth = torch.from_numpy(np.expand_dims(np.expand_dims(cv2.resize(tof, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_CUBIC), 0), 0)).to(torch.float32)

        rgb = torch.from_numpy(np.expand_dims(np.moveaxis(rgb, 2, 0), 0)).to(torch.float32)
        tof = torch.from_numpy(np.expand_dims(np.expand_dims(tof, 0), 0)).to(torch.float32)

        mask_lr = (~torch.isnan(tof)).float()

        sample = {'guide': rgb, 'source': tof, 'mask_lr': mask_lr, 'y_bicubic': upscaled_depth}

        with torch.no_grad():
            sample = to_cuda(sample)

            output = self.model(sample)

        return output['y_pred'].detach().squeeze().cpu().numpy(), rgb

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            model_dict = checkpoint['model']
            model_dict.pop('logk2', None) # in case of using the old codebase, pop unneccesary keys
            model_dict.pop('mean_guide', None)
            model_dict.pop('std_guide', None)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Checkpoint \'{path}\' loaded.')