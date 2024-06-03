import argparse
import numpy as np
import os
import imageio
import re
from pathlib import Path
import skimage
import cv2


class MiddleburyDataset:
    def __init__(self, root='./dataset_raw/', output_folder_name='data/middlebury_2014') -> None:
        self.root = root
        self.output_folder_name = output_folder_name
        self.image_list = []
        self.disparity_list = []
        self.calibration_list = []
        self.sampling_ratios = [1, 2, 4, 8 ,16, 32]

    def create(self):
        self.get_file_list()
        self.create_folder_structure()
        self.read_files()

    def get_file_list(self):
        scenes = list(Path(self.root).glob('*'))
        for scene in scenes:
            self.image_list += [str(scene / 'im0.png')]
            self.disparity_list += [ str(scene / 'disp0.pfm') ]
            self.calibration_list += [str(scene / 'calib.txt')]

    def create_folder_structure(self):
        if not os.path.exists(self.output_folder_name):
            os.mkdir(self.output_folder_name)
        for sampling_ratio in self.sampling_ratios:
            path = os.path.join(self.output_folder_name, f'x{sampling_ratio}')
            if not os.path.exists(path):
                os.mkdir(path)

    def read_files(self):
        for im_path, disp_path, calib_path in zip(self.image_list, self.disparity_list, self.calibration_list):
            name = re.search(r'[/\\]([\w\-]+)[/\\]im\d.png', im_path).group(1)
            im = cv2.imread(im_path)
            disp, occ = read_disp_middlebury(disp_path)
            calib = read_calibration(calib_path)
            depth = calib['baseline'] * calib['cam0'][0, 0] / (disp + calib['doffs'])
            self.downsample_data(im.astype(np.float32), depth, occ, name)

    def downsample_data(self, im, depth, occ, name, depth_dtype=np.uint16):
        max_sampling_ratio = np.max(self.sampling_ratios)
        h, w = depth.shape
        h = h - np.mod(h, max_sampling_ratio)
        w = w - np.mod(w, max_sampling_ratio)
        for sampling_ratio in self.sampling_ratios:
            depth_downsampled = skimage.transform.rescale(depth[:h, :w], 1/sampling_ratio, anti_aliasing=True, order=3).astype(depth_dtype)
            rgb_downsampled = skimage.transform.rescale(im[:h, :w], (1/sampling_ratio, 1/sampling_ratio, 1), anti_aliasing=True, order=3).astype(np.uint8)
            occ_downsampled = skimage.transform.rescale(255 * occ[:h, :w].astype(np.float32), 1/sampling_ratio, anti_aliasing=True, order=3).astype(np.uint8)
            cv2.imwrite(os.path.join(self.output_folder_name, f'x{sampling_ratio}', f'depth_{name}.png'), depth_downsampled)
            cv2.imwrite(os.path.join(self.output_folder_name, f'x{sampling_ratio}', f'rgb_{name}.png'), rgb_downsampled)
            cv2.imwrite(os.path.join(self.output_folder_name, f'x{sampling_ratio}', f'mask_{name}.png'), occ_downsampled)



class Middlebury2005Dataset(MiddleburyDataset):
    def __init__(self, root='./dataset_raw/', output_folder_name='data/middlebury_2005') -> None:
        super().__init__(root, output_folder_name)

    def create(self):
        self.get_file_list()
        super().create_folder_structure()
        self.read_files()

    def get_file_list(self):
        scenes = list(Path(self.root).glob('*'))
        for scene in scenes:
            self.image_list += [str(scene / 'view1.bmp')]
            self.disparity_list += [ str(scene / 'disp1.bmp') ]

    def read_files(self):
        for im_path, disp_path in zip(self.image_list, self.disparity_list):
            name = re.search(r'[/\\]([\w\-]+)[/\\]view\d.bmp', im_path).group(1)
            im = cv2.imread(im_path)
            disp = cv2.imread(disp_path)[..., 0]
            occ = np.where(disp == 0, 0, 1)
            super().downsample_data(im.astype(np.float32), 255 * disp.astype(np.float32), occ.astype(np.float32), name, depth_dtype=np.uint16)


def read_disp_middlebury(file_name):
    if os.path.basename(file_name) == 'disp0GT.pfm':
        disp = read_pfm(file_name).astype(np.float32)
        assert len(disp.shape) == 2
        nocc_pix = file_name.replace('disp0GT.pfm', 'mask0nocc.png')
        assert os.path.exists(nocc_pix)
        nocc_pix = imageio.imread(nocc_pix) == 255
        assert np.any(nocc_pix)
        return disp, nocc_pix
    elif os.path.basename(file_name) == 'disp0.pfm':
        disp = read_pfm(file_name).astype(np.float32)
        valid = disp < 1e3
        return disp, valid


def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def read_calibration(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    result = {}
    for line in lines:
        key, value = line.split('=')
        value = value.strip()
        if '[' in value:
            value = re.sub(r'\[|\]', '', value)
            value = np.array([[float(num) for num in row.split()] for row in value.split(';')])
        else:
            value = float(value)
        result[key] = value
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=False, default='data/dataset_2014_raw', type=str, help='path to Middlebury 2014 folder')
    parser.add_argument('-o', '--path_old', required=False, default='data/dataset_2005_raw', type=str, help='path to the Middlebury 2005 folder')
    args = parser.parse_args()
    middlebury_dataset = MiddleburyDataset(args.path)
    middlebury_dataset.create()
    middlebury_dataset = Middlebury2005Dataset(args.path_old)
    middlebury_dataset.create()