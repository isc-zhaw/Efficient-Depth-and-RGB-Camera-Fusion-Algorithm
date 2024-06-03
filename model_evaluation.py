import json
import os
import argparse
import pathlib
import numpy as np

from evaluation.evaluate import Evaluator
from dataset.dataset import Middlebury2014Dataset, Middlebury2005Dataset, TartanAirEval, Diml, TofDataset
from utilities.file_readers import LiveDownsamplingReader, read_npy_depth, read_png_depth
from models.basic import bicubic_interpolation, smoothed_nn_interpolation, GuidedFilter
from models.dada import Dada
from models.dkn import Dkn
from models.ahmf import Ahmf
from models.lgr import Lgr
from models.edge_extrapolation import EdgeExtrapolationModel


def evaluate(methods: list = ['all'], scales: list = ['x32', 'x16', 'x8', 'x4']):
    metrics_json_path = 'metrics.json'
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    if not os.path.exists('evaluation_output'):
        pathlib.Path('evaluation_output').mkdir(parents=True)

    # Spefify how many frames to evaluate for TartanAir or set to None to evaluate all 190'000 frames
    TARTANAIR_LENGTH = 50
    DIML_LENGTH = 50

    for scale in scales:
        print(scale)
        print('--------------------\n')

        scale_int = int(scale.replace('x', ''))

        tartanair_depth_reader = LiveDownsamplingReader(scale_int, read_npy_depth)
        diml_depth_reader = LiveDownsamplingReader(scale_int, read_png_depth)

        if 'bicubic' in methods or 'all' in methods:
            print('Bicubic Interpolation')
            bicubic_evaluator = Evaluator(bicubic_interpolation, name=f'Bicubic {scale}')
            results[f'bicubic_{scale}_mb05'] =  bicubic_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'bicubic_{scale}_mb14'] = bicubic_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'bicubic_{scale}_tartanair'] = bicubic_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'bicubic_{scale}_diml'] = bicubic_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'bicubic_{scale}_3D_ToF'] = bicubic_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})

        if 'guided_filter' in methods or 'all' in methods:
            print('Guided Filter')
            guided_filter_evaluator = Evaluator(GuidedFilter(radius=scale_int, eps=0.1), name=f'Guided Filter {scale}')
            results[f'guided_filter_{scale}_mb05'] = guided_filter_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'guided_filter_{scale}_mb14'] = guided_filter_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'guided_filter_{scale}_tartanair'] = guided_filter_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'guided_filter_{scale}_diml'] = guided_filter_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'guided_filter_{scale}_3D_ToF'] = guided_filter_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})

        if 'edge_extrapolation' in methods or 'all' in methods:
            print('Edge Extrapolation')
            if scale == 'x32':
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=22, interpolation_size=3, interpolation_method=bicubic_interpolation)
            elif scale == 'x8':
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=5, interpolation_size=3, interpolation_method=bicubic_interpolation)
            elif scale == 'x4':
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=3, interpolation_size=3, interpolation_method=bicubic_interpolation)
            else:
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=11, interpolation_size=3, interpolation_method=bicubic_interpolation)

            edge_extrapolation_evaluator = Evaluator(edge_extrapolation_model, name=f'Edge Extrapolation {scale}')
            results[f'edge_extrapolation_{scale}_mb05'] = edge_extrapolation_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'edge_extrapolation_{scale}_mb14'] = edge_extrapolation_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})

            if scale == 'x32':
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=22, interpolation_size=3, interpolation_method=bicubic_interpolation)
            elif scale == 'x8':
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=5, interpolation_size=3, interpolation_method=bicubic_interpolation)
            elif scale == 'x4':
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=3, interpolation_size=3, interpolation_method=bicubic_interpolation)
            else:
                edge_extrapolation_model = EdgeExtrapolationModel(interpolation_order=2, invalidation_size=11, interpolation_size=3, interpolation_method=bicubic_interpolation)

            edge_extrapolation_evaluator = Evaluator(edge_extrapolation_model, name=f'Edge Extrapolation {scale}')
            results[f'edge_extrapolation_{scale}_tartanair'] = edge_extrapolation_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader, ), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'edge_extrapolation_{scale}_diml'] = edge_extrapolation_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})

            if scale == 'x4':
                invalidation_size = 8
                maximum_filter_size = 10
                edge_intensity_threshold=29
            elif scale == 'x8':
                invalidation_size = 15
                maximum_filter_size = 19
                edge_intensity_threshold=19
            elif scale == 'x16':
                invalidation_size = 30
                maximum_filter_size = 38
                edge_intensity_threshold=9
            else:
                invalidation_size = 60
                maximum_filter_size = 76
                edge_intensity_threshold=3
            print('Edge Extrapolation')
            edge_extrapolation_model = EdgeExtrapolationModel(
                invalidation_size=invalidation_size,
                rgb_threshold_low=100,
                rgb_threshold_high=120,
                sobel_ksize=5,
                interpolation_size=5,
                edge_intensity_threshold=edge_intensity_threshold,
                maximum_filter_size=maximum_filter_size,
                interpolation_method=smoothed_nn_interpolation)
            edge_extrapolation_evaluator = Evaluator(edge_extrapolation_model, name=f'Edge Extrapolation {scale}')
            results[f'edge_extrapolation_{scale}_3D_ToF'] = edge_extrapolation_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})

        if 'dada' in methods or 'all' in methods:
            print('DADA')
            dada_evaluator = Evaluator(Dada(f'model_checkpoints/DADA/Middlebury, {scale}/best_model.pth'), name=f'DADA {scale}')
            results[f'dada_{scale}_mb05'] =  dada_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'dada_{scale}_mb14'] =  dada_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'dada_{scale}_tartanair'] = dada_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})

            dada_evaluator = Evaluator(Dada(f'model_checkpoints/DADA/NYUv2, {scale}/best_model.pth'), name=f'DADA {scale}')

            results[f'dada_{scale}_diml'] = dada_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'dada_{scale}_3D_ToF'] = dada_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})
    
        if 'ahmf' in methods or 'all' in methods:
            print('AHMF')
            if scale_int >= 16:
                ahmf_model = Ahmf(f'model_checkpoints/AHMF/m_16_2.pth', scale=scale_int)
            else:
                ahmf_model = Ahmf(f'model_checkpoints/AHMF/m_{scale_int}_2.pth', scale=scale_int)
            ahmf_evaluator = Evaluator(ahmf_model, name=f'AHMF {scale}')
            results[f'ahmf_{scale}_mb05'] = ahmf_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'ahmf_{scale}_mb14'] = ahmf_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'ahmf_{scale}_tartanair'] = ahmf_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'ahmf_{scale}_diml'] = ahmf_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'ahmf_{scale}_3D_ToF'] = ahmf_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})

        if 'dkn' in methods or 'all' in methods:
            print('DKN')
            dkn = Dkn(model='DKN', kernel_size=3, filter_size=15)
            dkn_evaluator = Evaluator(dkn, name=f'DKN {scale}')
            results[f'dkn_{scale}_mb05'] = dkn_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'dkn_{scale}_mb14'] = dkn_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'dkn_{scale}_tartanair'] = dkn_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'dkn_{scale}_diml'] = dkn_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'dkn_{scale}_3D_ToF'] = dkn_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})

        if 'fdkn' in methods or 'all' in methods:
            print('FDKN')
            dkn = Dkn(model='FDKN', kernel_size=3, filter_size=15)
            dkn_evaluator = Evaluator(dkn, name=f'FDKN {scale}')
            results[f'fdkn_{scale}_mb05'] = dkn_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'fdkn_{scale}_mb14'] = dkn_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'fdkn_{scale}_tartanair'] = dkn_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'fdkn_{scale}_diml'] = dkn_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'fdkn_{scale}_3D_ToF'] = dkn_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})


        if 'lgr' in methods or 'all' in methods:
            print('LGR')
            if scale_int >= 16:
                lgr = Lgr(scale_int, crop_size=256)
            elif scale_int >= 8:
                lgr = Lgr(scale_int, crop_size=256)
            else:
                lgr = Lgr(scale_int, crop_size=128)
            lgr_evaluator = Evaluator(lgr, name=f'LGR {scale}')
            results[f'lgr_{scale}_mb14'] = lgr_evaluator.evaluate_dataset(Middlebury2014Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'lgr_{scale}_mb05'] = lgr_evaluator.evaluate_dataset(Middlebury2005Dataset(input_scale=scale, output_scale='x1'), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 1}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 1})
            results[f'lgr_{scale}_tartanair'] = lgr_evaluator.evaluate_dataset(TartanAirEval(length=TARTANAIR_LENGTH, tof_reader=tartanair_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'lgr_{scale}_diml'] = lgr_evaluator.evaluate_dataset(Diml(length=DIML_LENGTH, tof_reader=diml_depth_reader), metrics=['rmse', 'mae', 'mse', 'bad_pixels'], metric_kwargs={'threshold': 10}, plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 10})
            results[f'lgr_{scale}_3D_ToF'] = lgr_evaluator.evaluate_dataset(TofDataset(scale), metrics=[], plot_kwargs={'output_folder': 'evaluation_output', 'threshold': 50})

        store_results(results, metrics_json_path)


def store_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--methods', required=False, default=['all'], type=str, nargs='+', help='list of methods to evaluate')
    parser.add_argument('-s', '--scales', required=False, default=['x32', 'x16', 'x8', 'x4'], type=str, nargs='+', help='list of scale factors to evaluate')
    args = parser.parse_args()
    evaluate(args.methods, args.scales)