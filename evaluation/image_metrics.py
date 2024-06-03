import numpy as np

def pixel_wise_error(expected, actual):
    return actual - expected


def mean_deviation_image(expected, actual, neighborhood_size=5, max_deviation=0.1):
    expected_deviation, actual_deviation = center_valid_mean_difference_2(expected, actual, neighborhood_size=neighborhood_size, max_deviation=max_deviation)
    #actual_deviation = center_valid_mean_difference_2(actual, neighborhood_size=neighborhood_size, max_deviation=max_deviation)
    return actual_deviation - expected_deviation


def center_valid_mean_difference(x, max_deviation=0.1):
    center = x[int(x.shape[0] // 2)+1]
    max_difference = center * max_deviation
    mean = np.mean(x[np.abs(x - center) < max_difference])
    return np.abs(center - mean)


def center_valid_mean_difference_2(expected, actual, neighborhood_size=5, max_deviation=0.1):
    expected_output = np.zeros_like(expected)
    actual_output = np.zeros_like(actual)
    e_padded = np.pad(expected, pad_width=neighborhood_size, mode='constant', constant_values=0)
    a_padded = np.pad(actual, pad_width=neighborhood_size, mode='constant', constant_values=0)
    return center_valid_mean_difference_padded(e_padded, a_padded, expected_output, actual_output, neighborhood_size, max_deviation)


def center_valid_mean_difference_padded(e_padded, a_padded, expected_output, actual_output, neighborhood_size, max_deviation):
    for i in range(neighborhood_size, e_padded.shape[0] - neighborhood_size):
        e_line = np.squeeze(np.lib.stride_tricks.sliding_window_view(e_padded[i-neighborhood_size:i+neighborhood_size+1], (2 * neighborhood_size + 1, ) * 2))
        a_line = np.squeeze(np.lib.stride_tricks.sliding_window_view(a_padded[i-neighborhood_size:i+neighborhood_size+1], (2 * neighborhood_size + 1, ) * 2))
        e_center = e_line[:, neighborhood_size, neighborhood_size, None, None]
        a_center = a_line[:, neighborhood_size, neighborhood_size, None, None]
        e_mask = np.where(np.abs(e_line - e_center) <= e_center * max_deviation, True, False)
        a_mask = np.where(np.abs(a_line - a_center) <= a_center * max_deviation, True, False)
        mask = np.logical_and(e_mask, a_mask)
        e_mean = (np.sum(e_line * mask, axis=(1, 2)) / np.sum(mask, axis=(1, 2)))[:, None, None]
        a_mean = (np.sum(a_line * mask, axis=(1, 2)) / np.sum(mask, axis=(1, 2)))[:, None, None]
        expected_output[i-neighborhood_size] = np.squeeze(np.abs(e_center - e_mean))
        actual_output[i-neighborhood_size] = np.squeeze(np.abs(a_center - a_mean))
    return expected_output, actual_output
