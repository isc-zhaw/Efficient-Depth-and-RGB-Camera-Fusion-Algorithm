import cv2

def bicubic_interpolation(rgb, depth):
    upscaled_depth = cv2.resize(depth, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_CUBIC)
    return upscaled_depth, rgb


def lanczos4_interpolation(rgb, depth):
    upscaled_depth = cv2.resize(depth, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_LANCZOS4)
    return upscaled_depth, rgb


def nn_interpolation(rgb, depth):
    upscaled_depth = cv2.resize(depth, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_NEAREST_EXACT)
    return upscaled_depth, rgb

def smoothed_nn_interpolation(rgb, depth, sigma_factor=0.22):
    upscaled_depth = cv2.resize(depth, rgb.shape[1::-1], None, 0, 0, interpolation=cv2.INTER_NEAREST_EXACT)
    kernel_size = 2 * int(rgb.shape[0] / depth.shape[0]) + 1
    upscaled_depth = cv2.GaussianBlur(upscaled_depth, sigmaX=sigma_factor * kernel_size, ksize=(kernel_size, kernel_size))
    return upscaled_depth, rgb

class GuidedFilter:
    def __init__ (self, radius=3, eps=0.1):
        self.radius = radius
        self.eps = eps

    def __call__(self, rgb, depth):
        upscaled_depth, _ = bicubic_interpolation(rgb, depth)
        upscaled_depth = cv2.ximgproc.guidedFilter(rgb, upscaled_depth, self.radius, self.eps)
        return upscaled_depth, rgb