from evaluation.metrics import mae, mse, rmse, bad_pixels, mean_deviation
from evaluation.image_metrics import pixel_wise_error, mean_deviation_image

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import time


class Evaluator:
    def __init__(self, model, name) -> None:
        self.model = model
        self.name = name

        self._metrics = {'rmse': rmse, 'bad_pixels': bad_pixels, 'mae': mae, 'mse': mse, 'mean_deviation': mean_deviation}
        self.random_generator = np.random.default_rng(1234)


    def evaluate_dataset(self, dataset, plot_kwargs=None, metrics=None, metric_kwargs={}):
        if metrics is None:
            metrics = list(self._metrics.keys())
        results = {}
        for metric in metrics:
            results[metric] = 0

        times = []

        mod = 1
        if len(dataset) > 10:
            mod = int(len(dataset) / 10)
        for i, data in enumerate(dataset):
            start = time.perf_counter_ns()
            retval = self.model(data['rgb'], data['tof'])
            stop = time.perf_counter_ns()
            times.append(stop-start)
            if len(retval) == 3:
                predicted, _, additional_data = retval
            else:
                predicted, _ = retval
                additional_data = []
            if np.any(data['gt']):
                gt = data['gt']
                for metric_name in metrics:
                    results[metric_name] += self._metrics[metric_name](gt, predicted, **metric_kwargs) / len(dataset)
            if plot_kwargs is not None and i % mod == 0:
                self.plot_dataset_evaluation(predicted, data, dataset, i, additional_data=additional_data, **plot_kwargs)
        
        results['time_us'] = np.mean(times) / 1000
        return results

    def plot_dataset_evaluation(self, predicted, data, dataset, dataset_index, output_folder, additional_data=[], threshold=5e-3):
            vmin, vmax = np.nanmin(data['gt']), np.nanmax(data['gt'])
            if vmin == 0 and vmax == 0:
                vmin, vmax = np.nanmin(data['tof'][data['tof'] > 0]), np.nanmax(data['tof'])
            ticks = np.linspace(vmin, vmax, 4)

            suptitle_kwargs = {'fontsize': 40, 'fontweight': 'bold'}
            title_kwargs = {'fontsize': 40}

            colors = ['indigo', 'navy', 'mediumblue', 'royalblue', 'cornflowerblue', 'lightskyblue', 'paleturquoise', 'aliceblue', 'cornsilk', 'gold', 'orange', 'darkorange', 'orangered', 'crimson', 'darkred', 'sienna']
            error_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

            if data['gt'] is None:
                error = np.zeros_like(predicted)
                data['gt'] = np.zeros_like(predicted)
            elif dataset.name == 'ToF Dataset':
                error = mean_deviation_image(data['gt'], predicted, neighborhood_size=31, max_deviation=0.5)
                error_threshold = threshold / 5
                error_name = 'Mean Deviation'
            else:
                error = pixel_wise_error(data['gt'], predicted)
                error_threshold = 5 * threshold
                error_name = 'Error'

            n_rows = 3 + int((len(additional_data) + 1) / 2)

            f, axs = plt.subplots(n_rows, 2, figsize=(32, n_rows*10), layout='constrained')
            for axi in axs:
                for axj in axi:
                    axj.axis('off')

            f.suptitle(f'{self.name} - {dataset.name}', **suptitle_kwargs)
            axs[0][0].set_title('ToF Depthmap', **title_kwargs)
            im = axs[0][0].imshow(data['tof'], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[0][1].set_title('RGB Image', **title_kwargs)
            axs[0][1].imshow(data['rgb'])
            axs[1][0].set_title('Predicted Depthmap', **title_kwargs)
            im = axs[1][0].imshow(predicted, vmin=vmin, vmax=vmax, cmap=cmap)
            axs[1][1].set_title('Ground-Truth Depthmap', **title_kwargs)
            im = axs[1][1].imshow(data['gt'], vmin=vmin, vmax=vmax, cmap=cmap)
            cbar = f.colorbar(im, ax=axs[1][0], orientation='vertical', format='{x:.1f} ' + dataset.unit, ticks=ticks, aspect=15, shrink=0.75, pad=0.05)
            cbar.ax.tick_params(labelsize=25)
            axs[2][0].set_title(error_name, **title_kwargs)
            im = axs[2][0].imshow(error, vmin=-error_threshold, vmax=error_threshold, cmap=error_cmap)
            cbar = f.colorbar(im, ax=axs[2][0], orientation='vertical', format='{x:.1f} ' + dataset.unit, ticks=[-error_threshold, -0.5*error_threshold, 0, 0.5*error_threshold, error_threshold], aspect=15, shrink=0.75, pad=0.05)
            cbar.ax.tick_params(labelsize=25)
            axs[2][1].set_title(f'Bad Pixels (Threshold: {threshold} {dataset.unit})', **title_kwargs)
            axs[2][1].imshow(np.where(np.abs(data['gt'] - predicted) > threshold, 1, 0).reshape(data['gt'].shape), vmin=0, vmax=1, cmap='gray')

            for i, additional_element in enumerate(additional_data):
                idx_1 = i % 2
                idx_2 = int((i - idx_1) / 2) + 3
                axs[idx_2][idx_1].set_title(additional_element.get('title', 'Empty'), **title_kwargs)
                axs[idx_2][idx_1].imshow(additional_element.get('data', np.zeros((2, 2))), **additional_element.get('kwargs', {}))

            f.savefig(f'{output_folder}/{self.name.replace(" ", "_")}_{dataset.name.replace(" ", "_")}_{dataset_index}.png')

            plt.imsave(f'{output_folder}/{self.name.replace(" ", "_")}_{dataset.name.replace(" ", "_")}_{dataset_index}_depthmap.png', predicted, vmin=vmin, vmax=vmax, cmap='magma')
            plt.close()