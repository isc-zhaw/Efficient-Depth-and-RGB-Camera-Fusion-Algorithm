import numpy as np

from evaluation.image_metrics import mean_deviation_image


def mae(expected, actual, **kwargs):
    return np.nanmean(np.abs(actual - expected))

def mse(expected, actual, **kwargs):
    return np.nanmean(np.square(actual - expected))

def rmse(expected, actual, **kwargs):
    return np.sqrt(mse(expected, actual))

def bad_pixels(expected, actual, threshold=5e-3, **kwargs):
    return  np.sum(np.where(np.abs(actual - expected) > threshold, 1, 0)) / np.size(expected)

def good_pixel_rmse(expected, actual, threshold=5e-3):
    indices = np.where(np.abs(actual - expected) <= threshold)
    return rmse(expected[indices], actual[indices])

def mean_deviation(expected, actual, neighborhood_size=31, max_deviation=0.5, **kwargs):
    mean_deviation = mean_deviation_image(expected, actual.astype(np.float32), neighborhood_size=neighborhood_size, max_deviation=max_deviation)
    return np.nanmean(np.abs(mean_deviation))