from utilities.file_readers import read_png_color_image, read_rgb_color_image, read_bgr_color_image, read_png_depth, read_exr_depth, read_binary_mask, read_uint8_disparity, read_uint16_disparity, read_npy_depth, read_mat_depth, LiveDownsamplingReader

from glob import glob
import os
from math import ceil
import numpy as np

class Dataset:
    def __init__(self, rgb_reader=read_png_color_image, tof_reader=read_png_depth, gt_reader=read_png_depth, mask_reader=read_binary_mask, segmentation_reader=None) -> None:
        self._rgb_reader = rgb_reader
        self._tof_reader = tof_reader
        self._gt_reader = gt_reader
        self._mask_reader = mask_reader
        self._segmentation_reader = segmentation_reader
        self._rgb_paths = None
        self._tof_paths = None
        self._gt_paths = None
        self._mask_paths = None
        self._segmentation_paths = None
        self.max_distance = 255 # For Middlebury 2005
        self.unit = 'mm'
        self.name = 'Parent Class'

    def __len__(self):
        return len(self._tof_paths)

    def __iter__(self):
        return DatasetIterator(self)

    def __getitem__(self, index):
        if isinstance(index, int) or len(index) < 2:
            return self.get_data(index)
        else:
            result = []
            for i in index:
                result.append(self.get_data(i))
            return result

    def get_data(self, index):
        rgb_image = self._rgb_reader(self._rgb_paths[index])
        tof_image = self._tof_reader(self._tof_paths[index])
        tof_image = np.clip(tof_image, 0, self.max_distance)
        gt = self._gt_reader(self._gt_paths[index])

        sampling_ratio = int(rgb_image.shape[0] / tof_image.shape[0])
        hl, wl = tof_image.shape
        hl = hl - np.mod(hl, 2)
        wl = wl - np.mod(wl, 2)
        tof_image = tof_image[:hl, :wl]

        h = hl * sampling_ratio
        w = wl * sampling_ratio
        
        rgb_image = rgb_image[:h, :w]

        if gt is None:
            gt = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
        else:
            gt = gt[:h, :w]

        if len(self._mask_paths) == len(self._tof_paths):
            mask = self._mask_reader(self._mask_paths[index])[:h, :w]
        else:
            mask = np.where(np.logical_or(gt > self.max_distance, gt < 0), 0, 1)

        if self._segmentation_paths and len(self._segmentation_paths) == len(self._tof_paths):
            segmentation = self._segmentation_reader(self._segmentation_paths[index])
        else:
            segmentation = None

        gt = np.where(np.logical_or(gt > self.max_distance, gt < 0), 0, gt)

        return {'tof': tof_image, 'rgb': rgb_image, 'gt': gt, 'mask': mask, 'segmentation': segmentation, 'rgb_path': self._rgb_paths[index]}


class DatasetIterator:
    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self._index = 0

    def __next__(self):
        if self._index < len(self._dataset):
            result = self._dataset[self._index]
            self._index += 1
            return result

        raise StopIteration


class TofDataset(Dataset):
    def __init__(self, upscaling_factor='x16', rgb_reader=LiveDownsamplingReader(1, read_bgr_color_image), tof_reader=read_png_depth, gt_reader=lambda x: None) -> None:
        super().__init__(rgb_reader, tof_reader, gt_reader)
        self.name = '3D ToF Dataset'
        root = 'data/3D_ToF/'

        N = int(upscaling_factor.replace('x', ''))
        if N == 32:
            N = 1
            self._tof_reader = LiveDownsamplingReader(2, tof_reader)
        elif N > 32 or N <= 1:
            raise Exception('Upscaling factor must be > 1 and <= 32')
        else:
            N = 16 / N
        self._rgb_reader.downscaling_factor = N

        self._tof_paths = []
        files = glob(root + '/*_depth.png')
        self._tof_paths.extend(sorted(files, key=lambda x: os.path.basename(x).split('.png')[0]))

        self._rgb_paths = [tof_path.replace('_depth', '_rgb') for tof_path in self._tof_paths]
        self._gt_paths = [None] * len(self._tof_paths)
        self._mask_paths = []

        self.max_distance = 12500



class Middlebury2014Dataset(Dataset):
    def __init__(self, input_scale='x16', output_scale='x1', rgb_reader=read_png_color_image, tof_reader=read_png_depth, gt_reader=read_png_depth, mask_reader=read_binary_mask) -> None:
        super().__init__(rgb_reader, tof_reader, gt_reader, mask_reader)

        self.name = 'Middlebury 2014'

        self._rgb_paths = glob(f'data/middlebury_2014/{output_scale}/rgb_*.png')
        self._gt_paths = [rgb_path.replace('rgb_', 'depth_') for rgb_path in self._rgb_paths]
        self._tof_paths = [tof_path.replace(output_scale, input_scale) for tof_path in self._gt_paths]
        self._mask_paths = [rgb_path.replace('rgb_', 'mask_') for rgb_path in self._rgb_paths]

        self.max_distance = 10000


class Middlebury2005Dataset(Dataset):
    def __init__(self, input_scale='x16', output_scale='x1', dataset_root='data/middlebury_2005', rgb_reader=read_png_color_image, tof_reader=read_uint16_disparity, gt_reader=read_uint16_disparity, mask_reader=read_binary_mask) -> None:
        super().__init__(rgb_reader, tof_reader, gt_reader, mask_reader)

        self.name = 'Middlebury 2005'
        self.unit = 'pixels'

        self._rgb_paths = glob(f'{dataset_root}/{output_scale}/rgb_*.png')
        self._gt_paths = [rgb_path.replace('rgb_', 'depth_') for rgb_path in self._rgb_paths]
        self._tof_paths = [tof_path.replace(output_scale, input_scale) for tof_path in self._gt_paths]
        self._mask_paths = [rgb_path.replace('rgb_', 'mask_') for rgb_path in self._rgb_paths]

        self.max_distance = 255


class Diml(Dataset):
    def __init__(self, length=None, dataset_root='data/diml', rgb_reader=read_png_color_image, tof_reader=LiveDownsamplingReader(downscaling_factor=16, reader=read_png_depth), gt_reader=read_png_depth, mask_reader=read_binary_mask) -> None:
        super().__init__(rgb_reader, tof_reader, gt_reader, mask_reader)

        self.name = 'DIML'

        self._rgb_paths = glob(f'{dataset_root}/*/color/*.png')
        self._gt_paths = [rgb_path.replace('/color/', '/depth_filled/').replace('_c.png', '_depth_filled.png') for rgb_path in self._rgb_paths]
        self._tof_paths = self._gt_paths[:]
        self._mask_paths = []

        if length is None:
            length = len(self._rgb_paths)

        indices = np.linspace(0, len(self), length, endpoint=False, dtype=int)

        self._rgb_paths = [self._rgb_paths[index] for index in indices]
        self._gt_paths = [self._gt_paths[index] for index in indices]
        self._tof_paths = [self._tof_paths[index] for index in indices]

        self.max_distance = np.iinfo(np.uint16).max


class TartanAir(Dataset):
    def __init__(self, dataset_root='data/tartanair', split=None, rgb_reader=read_png_color_image, tof_reader=LiveDownsamplingReader(downscaling_factor=16, reader=read_npy_depth), gt_reader=read_npy_depth, mask_reader=read_binary_mask) -> None:
        super().__init__(rgb_reader, tof_reader, gt_reader, mask_reader)

        self.name = 'TartanAir'

        if not isinstance(split, list):
            split = ['*']

        for split_folder in split:
            split_wildcard = f'*{split_folder}*'
            self._rgb_paths = glob(f'{dataset_root}/{split_wildcard}/{split_wildcard}/Easy/*/image_left/*.png')
            self._gt_paths = [rgb_path.replace('image_left', 'depth_left').replace('.png', '_depth.npy') for rgb_path in self._rgb_paths]
            self._tof_paths = self._gt_paths[:]
            self._mask_paths = []

        self.max_distance = 25


class TartanAirEval(TartanAir):
    def __init__(self, length=30, dataset_root='data/tartanair', split=None, rgb_reader=read_png_color_image, tof_reader=LiveDownsamplingReader(downscaling_factor=16, reader=read_npy_depth), gt_reader=read_npy_depth, mask_reader=read_binary_mask) -> None:
        super().__init__(dataset_root, split, rgb_reader, tof_reader, gt_reader, mask_reader)

        self.name = 'TartanAirEval'
        self.unit = 'm'

        if length is None:
            length = len(self._rgb_paths)

        indices = np.linspace(0, len(self), length, endpoint=False, dtype=int)

        self._rgb_paths = [self._rgb_paths[index] for index in indices]
        self._gt_paths = [self._gt_paths[index] for index in indices]
        self._tof_paths = [self._tof_paths[index] for index in indices]

        self.max_distance = 25