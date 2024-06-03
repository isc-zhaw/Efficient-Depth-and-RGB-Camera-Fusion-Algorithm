<h2 align="center">Efficient Depth and RGB Camera Fusion Algorithm for Depth Imaging with Improved Spatial Resolution</h2>
  <p align="center">
  <strong>Yannick Waelti</strong>,
  <strong>Matthias Ludwig</strong>,
  <strong>Josquin Rosset</strong>
  <strong>Teddy Loeliger</strong>,
  </p>
  <p align="center">
Institute of Signal Processing and Wireless Communications (ISC),<br>ZHAW Zurich University of Applied Sciences
</p>

Cameras for capturing depth images typically suffer from low spatial resolution, whereas RGB cameras offer high resolution. A common approach to address these drawbacks of depth cameras is to combine them with RGB cameras, using a fusion algorithm. We propose, implement and benchmark a new spatial upscaling algorithm based on edge extrapolation and compare it to state-of-the-art spatial upscaling methods. The edge extrapolation algorithm is very efficient and achieves state-of-the-art performance. It produces high-resolution depth maps with particularly clean edges, although it also exhibits certain specific artifacts in the output. The new algorithm first interpolates the depth image and combines it with an RGB image to extract an edge map. The blurred edges of the interpolated depth map are then replaced by an extrapolation from neighboring pixels for the final high-resolution depth map. The algorithm is benchmarked using the Middlebury and DIML datasets, as well as a number of new 3D time-of-flight depth images.

## Setup

### Dependencies

We recommend using the provided docker file to run our code. Use the below commands to build and run the container.

`docker build -t tof_rgb_fusion:1.0 --build-arg USER_ID=$(id -u)   --build-arg GROUP_ID=$(id -g) .`

`docker run --name tof_rgb_fusion --gpus all --mount type=bind,source=/path/to/repository,target=/ToF_RGB_Fusion -dt tof_rgb_fusion:1.0`

Make sure to include the submodules by either cloning the repo with the `--recursive` option or running `git submodule update --init --recursive` if the repo has been cloned already

### Datasets
#### 3D_ToF
The set of three RGB and depth images recorded with a Raspberry Pi camera v2 and an ESPROS DME 635 3D ToF camera is located under `data/3D_ToF`

#### Middlebury
From within the dataset directory run the `download_middlebury_2014.sh` script to download the hole filled Middlebury 2005 and the original Middlebury 2014 datasets. To create the downscaled images, run `dataset/create_middlebury_dataset.py` from the repository root.

#### DIML
Download the indoor testing data from the [official website](https://dimlrgbd.github.io/) and place the contents of the HR directory under `data/diml`.

#### TartanAir
Download the TartanAir dataset from [here](https://github.com/castacks/tartanair_tools) and use the `--rgb --depth --only-easy --only-left` options. Place the dataset under `data/tartanair`.

### Methods
#### DADA Checkpoints
Get the model checkpoints for the DADA approach from their [repository](https://github.com/prs-eth/Diffusion-Super-Resolution/blob/main/README.md#-checkpoints) and extract the contents of the .zip file into the `model_checkpoints/DADA` folder.

#### AHMF
Get the model checkpoints from the official [repository](https://github.com/zhwzhong/AHMF?tab=readme-ov-file) and place the files under `model_checkpoints/AHMF`.
To make all models loadable, change all `kernel_size` in the `UpSampler` and `InvUpSampler` to 5. Also, replace `from collections import Iterable` with `from collections.abc import Iterable`.

#### DKN and FDKN
To use the DKN and FDKN models, some changes need to be made to the code from the official [repository](https://github.com/cvlab-yonsei/dkn?tab=readme-ov-file). Add align_corners=True to all calls of `F.grid_sample` if you use a PyTorch version > 1.12
If you get a `CUDNN_STATUS_NOT_SUPPORTED` error, wrap the `F.grid_sample` status in a with `torch.backends.cudnn.flags(enabled=False):` statement

#### LGR
The official [repository](https://github.com/prs-eth/graph-super-resolution/blob/master/run_eval.py) is used. We only use the learning-free method and therefore do not require any models to be downloaded.

## Evaluation
Run `model_evaluation.py` to get metrics and upscaled depthmaps for different approaches. Methods can be specified with the `-m` option (default: all) and upscaling factors can be specified with `-s` (one or multiple of `x4`, `x8`, `x16` or `x32`).

## Citation
```
@software{Waelti_Efficient_Depth_and,
author = {Waelti, Yannick and Ludwig, Matthias and Rosset, Josquin and Loeliger, Teddy},
license = {MIT},
title = {{Efficient Depth and RGB Camera Fusion Algorithm for Depth Imaging with Improved Spatial Resolution}},
url = {https://github.com/isc-zhaw/Efficient-Depth-and-RGB-Camera-Fusion-Algorithm}
}
```