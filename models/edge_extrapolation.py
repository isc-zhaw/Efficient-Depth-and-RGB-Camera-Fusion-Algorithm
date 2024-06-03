from typing import Callable, Dict
import numpy as np
import cv2
import scipy

from models.basic import lanczos4_interpolation


class EdgeExtrapolationModel:
    def __init__(self,
                 interpolation_order: int=2,
                 invalidation_size: int=11,
                 interpolation_size: int=3,
                 erosion_iterations: int=9, #9,
                 edge_intensity_threshold: int=35,#20, #30,
                 maximum_filter_size = None,
                 sobel_ksize: int=1,
                 rgb_threshold_low: int=38,#15,
                 rgb_threshold_high: int=188,#150
                 interpolation_method=lanczos4_interpolation
                 ) -> None:
        self.interpolation_order = interpolation_order
        self.invalidation_size = invalidation_size
        self.interpolation_size = interpolation_size
        self.upscaling_factor = None
        self.erosion_iterations = erosion_iterations
        self.edge_intensity_threshold = edge_intensity_threshold
        self.maximum_filter_size = maximum_filter_size
        self.sobel_ksize = 2*sobel_ksize+1
        self.rgb_threshold = (rgb_threshold_low, rgb_threshold_high)
        self.interpolation_method = interpolation_method

    def __call__(self, rgb: np.ndarray, depth: np.ndarray):
        self.upscaling_factor = int(rgb.shape[0] / depth.shape[0])
        if self.maximum_filter_size is None:
            self.maximum_filter_size = 2*self.upscaling_factor+1
        for i in range(100):
            hr_depth, _ = self.interpolation_method(rgb, depth)
            edges = self.detect_edges(
                rgb,
                hr_depth,
                erosion_iterations = self.erosion_iterations, #9,
                edge_intensity_threshold = self.edge_intensity_threshold, #30,
                maximum_filter_size = self.maximum_filter_size,#2*self.upscaling_factor+1,
                sobel_ksize = self.sobel_ksize,
                rgb_threshold = self.rgb_threshold
                )

            interpolated = self.neighborhood_based_interpolation(edges, hr_depth)

        additional_data = [
            {
                'title': 'Edges',
                'data': edges,
                'kwargs': {
                    'cmap': 'magma'
                }
             }
        ]

        return interpolated, rgb, additional_data

    def normalize(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def detect_edges(
        self,
        rgb: np.ndarray,
        hr_depth: np.ndarray,
        erosion_iterations: int = 5,
        edge_intensity_threshold: int = 200,
        maximum_filter_size: int = 17,
        sobel_ksize: int = 3,
        rgb_threshold: tuple = (110, 150)
        ):

        # Normalize Upscaled depth
        normalized_hr_depth = hr_depth
        normalized_hr_depth = (normalized_hr_depth - np.min(normalized_hr_depth)) / (np.max(normalized_hr_depth) - np.min(normalized_hr_depth))
        normalized_hr_depth = (normalized_hr_depth * 255).astype(np.uint8)

        # Apply Sobel to normalized depth
        grad_x = cv2.Sobel(normalized_hr_depth, cv2.CV_16S, 0, 1, ksize=sobel_ksize, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(normalized_hr_depth, cv2.CV_16S, 1, 0, ksize=sobel_ksize, borderType=cv2.BORDER_DEFAULT)
        depth_sobel = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)

        # Eroding accumulation on Depth
        depth_sobel_eroded = depth_sobel.copy().astype(np.float32)# + depth_sobel_large
        accumulated_depth_edges = np.zeros_like(normalized_hr_depth, dtype=np.float32)
        for _ in range(erosion_iterations):
            depth_sobel_eroded = cv2.erode(depth_sobel_eroded, kernel=np.ones((3, 3)))
            accumulated_depth_edges += depth_sobel_eroded# + 0 * depth_sobel_large_eroded
        # Threshold accumulated depth
        thresholded_accumulated_depth_edges = np.where(accumulated_depth_edges > edge_intensity_threshold, accumulated_depth_edges, 0)

        # Canny on RGB
        rgb_edges = cv2.Canny(rgb, rgb_threshold[0], rgb_threshold[1])

        masked_depth_edges = np.where(rgb_edges, thresholded_accumulated_depth_edges, 0)

        footprint = np.zeros((maximum_filter_size, maximum_filter_size))
        footprint[int(maximum_filter_size/2)] = 1
        horizontal_maximum_filtered_depth_edges = scipy.ndimage.maximum_filter(masked_depth_edges, footprint=footprint)
        vertical_maximum_filtered_depth_edges = scipy.ndimage.maximum_filter(masked_depth_edges, footprint=np.rot90(footprint, k=1))

        masked_rgb_edges = np.where(
            np.logical_and(
                masked_depth_edges > 0,
                np.logical_or(
                    horizontal_maximum_filtered_depth_edges == masked_depth_edges,
                    vertical_maximum_filtered_depth_edges == masked_depth_edges)),
            rgb_edges, 0)

        masked_depth_edges = np.clip(masked_depth_edges, a_min=0, a_max=1)
        masked_rgb_edges = np.clip(masked_rgb_edges, a_min=0, a_max=1)

        return masked_rgb_edges

    def neighborhood_based_interpolation(self, edges, depth):
        edge_y, edge_x = np.where(edges > 0)

        extrapolated_values = np.zeros_like(depth, dtype=float)
        extrapolation_value_counter = np.zeros_like(depth)

        for _, (angle_base, index_offset) in enumerate(zip([np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])], [0, 0, 1, 1])):
            steps = np.clip(self.upscaling_factor * angle_base, a_min=1, a_max=np.inf,)
            directional_coordinates  = np.stack((
                np.concatenate((
                    np.arange(index_offset, self.invalidation_size + index_offset),
                    np.arange(self.invalidation_size + index_offset, self.invalidation_size + index_offset + self.interpolation_size * steps[0], steps[0])
                )),
                np.concatenate((
                    np.arange(index_offset, self.invalidation_size + index_offset),
                    np.arange(self.invalidation_size + index_offset, self.invalidation_size + index_offset + self.interpolation_size * steps[1], steps[1])
                )),
                ))[:, :, None] * angle_base[:, None, None]
            directional_coordinates = np.transpose(directional_coordinates, (2, 1, 0)).astype(int)

            directional_coordinates = np.stack(
                (
                    edge_y[:, None] + directional_coordinates[..., 0],
                    edge_x[:, None] + directional_coordinates[..., 1]
                ),
                axis=-1)

            original_lr_coordinates_offset = ((-angle_base * np.mod(directional_coordinates[:, self.invalidation_size:], self.upscaling_factor)) + angle_base * self.upscaling_factor / 2).astype(int)
            original_lr_coordinates_offset = np.zeros_like(original_lr_coordinates_offset)
            directional_coordinates[:, self.invalidation_size:] = directional_coordinates[:, self.invalidation_size:] + original_lr_coordinates_offset

            # Remove coordinates that lie outside the image
            valid_idx = np.where(
                np.logical_and(
                    np.count_nonzero(directional_coordinates < 0, axis=(-1, -2)) == 0,
                    np.logical_and(
                        np.count_nonzero(directional_coordinates[..., 0] >= depth.shape[0], axis=-1) == 0,
                        np.count_nonzero(directional_coordinates[..., 1] >= depth.shape[1], axis=-1) == 0,
                    ))
            )
            directional_coordinates = directional_coordinates[valid_idx]
            original_lr_coordinates_offset = original_lr_coordinates_offset[valid_idx]

            values = depth[
                directional_coordinates.reshape(-1, 2)[:, 0],
                directional_coordinates.reshape(-1, 2)[:, 1]
                ].reshape(directional_coordinates.shape[:-1])

            intp_data = values[:, self.invalidation_size:]

            invalidation_coordinates = int(self.invalidation_size / self.upscaling_factor)
            x = np.tile(np.arange(invalidation_coordinates, invalidation_coordinates + intp_data.shape[1]), (intp_data.shape[0], 1))
            x = x + np.sum(original_lr_coordinates_offset * angle_base[None, :], axis=-1) / self.upscaling_factor
            X = np.tile(np.linspace(0, invalidation_coordinates, self.invalidation_size), (intp_data.shape[0], 1))
            p = np.stack((np.ones_like(x), ) + tuple([np.power(x, deg+1) for deg in range(self.interpolation_order)]), axis=-1)
            if self.interpolation_order == 4:
                p2 = np.stack((np.zeros((x.shape[0], x.shape[1] - 1)), np.ones((x.shape[0], x.shape[1] - 1)), 2 * x[:, 1:]), axis=-1)
                p = np.concatenate((p, p2), axis=1)
                intp_data = np.concatenate((intp_data, intp_data[:, 1:] - intp_data[:, :-1]), axis=1)

            m = np.einsum('ijk,ik->ij', np.linalg.pinv(p), intp_data)

            P = np.stack((np.ones_like(X), ) + tuple([np.power(X, deg+1) for deg in range(self.interpolation_order)]), axis=-1)
            intp_values = np.einsum('ij,ikj->ik', m, P)

            extrapolated_values[
                directional_coordinates[:, :self.invalidation_size].reshape(-1, 2)[:, 0],
                directional_coordinates[:, :self.invalidation_size].reshape(-1, 2)[:, 1]
            ] += intp_values.reshape(-1)

            extrapolation_value_counter[
                directional_coordinates[:, :self.invalidation_size].reshape(-1, 2)[:, 0],
                directional_coordinates[:, :self.invalidation_size].reshape(-1, 2)[:, 1]
            ] += 1

        extrapolation_value_counter = np.where(extrapolation_value_counter == 0, 1, extrapolation_value_counter)
        extrapolated_values = extrapolated_values / extrapolation_value_counter

        interpolated = np.where(extrapolated_values > 0, extrapolated_values, depth)
        return interpolated
