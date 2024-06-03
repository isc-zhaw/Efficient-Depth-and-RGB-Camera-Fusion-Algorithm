import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import scipy.io as sio
import skimage


def read_png_color_image(path):
    return read_color_cv_to_numpy(path)

def read_rgb_color_image(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Image is None: {path}'
    return img

def read_bgr_color_image(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Image is None: {path}'
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_png_depth(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return img_mm

def read_png_depth_meters(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_meter = img_mm / 1000
    return img_meter

def read_exr_depth(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def read_color_cv_to_numpy(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Image is None: {path}'
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_binary_mask(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img = img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.where(img >= 1, 1, 0)
    return img

def read_uint8_disparity(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def read_uint16_disparity(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    return img

def read_npy_depth(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    depth = np.load(path)
    return depth

def read_npy_segmentation(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    segmentation = np.load(path)['segmentation']
    return segmentation

class LiveDownsamplingReader:
    def __init__(self, downscaling_factor: int, reader=read_rgb_color_image):
        self.downscaling_factor = int(downscaling_factor)
        self.reader = reader

    def __call__(self, path):
        img = self.reader(path)
        dtype = img.dtype
        img = img.astype(np.float32)

        h, w = img.shape[0], img.shape[1]
        h = h - int(np.mod(h, self.downscaling_factor))
        w = w - int(np.mod(w, self.downscaling_factor))
        img = img[:h, :w]
        if len(img.shape) == 3:
            factor = (1/self.downscaling_factor, 1/self.downscaling_factor, 1)
        else:
            factor = 1/self.downscaling_factor
        img = skimage.transform.rescale(img, factor, anti_aliasing=True, order=3).astype(dtype)
        return img

def read_mat_depth(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    depth = sio.loadmat(path)['depth']
    return depth

def read_tofmark_gt_depth(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    depth = sio.loadmat(path)['gt_depth']
    return depth

def read_tofmark_gray(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    gray = sio.loadmat(path)['intensity_img']
    rgb = (np.stack((gray, gray, gray), axis=-1) * 255).astype(np.uint8)
    return rgb

def read_tofmark_tof_depth(path):
    assert os.path.exists(path), f'File does not exist: {path}'
    depth = sio.loadmat(path)['tof_depth']
    return depth