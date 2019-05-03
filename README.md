<img src='imgs/output6_120.gif' align="right" height="120px">

<br><br><br><br>

# Local Light Field Fusion
### [Project](https://fyusion.com/llff) | [Video](https://youtu.be/LY6MgDUzS3M) | [Paper](https://arxiv.org/abs/1905.00889) 

Tensorflow implementation for novel view synthesis from sparse input images.<br><br>
[Local Light Field Fusion: Practical View Synthesis 
with Prescriptive Sampling Guidelines](https://fyusion.com/llff)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>, 
 [Pratul Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>, 
 [Rodrigo Ortiz-Cayon](https://scholar.google.com/citations?user=yZMAlU4AAAAJ)<sup>2</sup>, 
 [Nima Khademi Kalantari](http://faculty.cs.tamu.edu/nimak/)<sup>3</sup>, 
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>4</sup>, 
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup>, 
 [Abhishek Kar](https://abhishekkar.info/)<sup>2</sup>  
 <sup>1</sup>UC Berkeley, <sup>2</sup>Fyusion Inc, <sup>3</sup>Texas A&amp;M, <sup>4</sup>UC San Diego  
  \*denotes equal contribution  
  In SIGGRAPH 2019
  

<img src='imgs/teaser.jpg'/>

## Table of Contents

  * [Installation TL;DR: Setup and render a demo scene](#installation-tldr-setup-and-render-a-demo-scene)
  * [Full Installation Details](#full-installation-details)
    * [Manual installation](#manual-installation)
    * [Docker installation](#docker-installation)
  * [Using your own input images for view synthesis](#using-your-own-input-images-for-view-synthesis)
    * [Quickstart: rendering a video from a zip file of your images](#quickstart-rendering-a-video-from-a-zip-file-of-your-images)
  * [General step-by-step usage](#general-step-by-step-usage)
    * [1. Recover camera poses](#1-recover-camera-poses)
    * [2. Generate MPIs](#2-generate-mpis)
    * [3. Render novel views](#3-render-novel-views)
  * [Citation](#citation)

## Installation TL;DR: Setup and render a demo scene

First install `docker` ([instructions](https://docs.docker.com/install/linux/docker-ce/ubuntu/)) and `nvidia-docker` ([instructions](https://github.com/NVIDIA/nvidia-docker)).

Run this in the base directory to download a pretrained checkpoint, download a Docker image, and run code to generate MPIs and a rendered output video on an example input dataset:
```
bash download_data.sh
sudo docker pull bmild/tf_colmap
sudo docker tag bmild/tf_colmap tf_colmap
sudo nvidia-docker run --rm --volume /:/host --workdir /host$PWD tf_colmap bash demo.sh
```
A video like this should be output to `data/testscene/outputs/test_vid.mp4`:  
</br>
<img src='imgs/fern.gif'/>

If this works, then you are ready to start processing your own images! Run
```
sudo nvidia-docker run -it --rm --volume /:/host --workdir /host$PWD tf_colmap
```
to enter a shell inside the Docker container, and [skip ahead](#using-your-own-input-images-for-view-synthesis) to the section on using your own input images for view synthesis.

## Full Installation Details

You can either install the prerequisites by hand or use our provided Dockerfile to make a docker image.

In either case, start by downloading this repository, then running the `download_data.sh` script to download a pretrained model and example input dataset:
```
bash download_data.sh
```
After installing dependencies, try running `bash demo.sh` from the base directory. (If using Docker, run this inside the container.) This should generate the video shown in the *Installation TL;DR* section at `data/testscene/outputs/test_vid.mp4`.

### Manual installation

- Install CUDA, Tensorflow, COLMAP, ffmpeg
- Install the required Python packages:
```
pip install -r requirements.txt
```
- Optional: run `make` in `cuda_renderer/` directory.
- Optional: run `make` in `opengl_viewer/` directory. You may need to install GLFW or some other OpenGL libraries. For GLFW:
```
sudo apt-get install libglfw3-dev
```


### Docker installation

To build the docker image on your own machine, which may take 15-30 mins:
```
sudo docker build -t tf_colmap:latest .
```
To download the image (~6GB) instead:
```
sudo docker pull bmild/tf_colmap
sudo docker tag bmild/tf_colmap tf_colmap
```

Afterwards, you can launch an interactive shell inside the container:
```
sudo nvidia-docker run -it --rm --volume /:/host --workdir /host$PWD tf_colmap
```
From this shell, all the code in the repo should work (except `opengl_viewer`).

To run any single command `<command...>` inside the docker container:
```
sudo nvidia-docker run --rm --volume /:/host --workdir /host$PWD tf_colmap <command...>
```


## Using your own input images for view synthesis

<img src='imgs/capture.gif'/>

Our method takes in a set of images of a static scene, promotes each image to a local layered representation (MPI), and blends local light fields rendered from these MPIs to render novel views. Please see our paper for more details. 

As a rule of thumb, you should use images where the maximum disparity between views is no more than about 64 pixels (watch the closest thing to the camera and don't let it move more than ~1/8 the horizontal field of view between images). Our datasets usually consist of 20-30 images captured handheld in a rough grid pattern.

#### Quickstart: rendering a video from a zip file of your images

You can quickly render novel view frames and a .mp4 video from a zip file of your captured input images with the `zip2mpis.sh` bash script. 
```
bash zip2mpis.sh <zipfile> <your_outdir> [--height HEIGHT]
```
`height` is the output height in pixels. We recommend using a height of 360 pixels for generating results quickly.

## General step-by-step usage

Begin by creating a base scene directory (e.g., `scenedir/`), and copying your images into a subdirectory called `images/` (e.g., `scenedir/images`).

#### 1. Recover camera poses

This script calls COLMAP to run structure from motion to get 6-DoF camera poses and near/far depth bounds for the scene.
```
python imgs2poses.py <your_scenedir>
```

#### 2. Generate MPIs

This script uses our pretrained Tensorflow graph (make sure it exists in `checkpoints/papermodel`) to generate MPIs from the posed images. They will be saved in `<your_mpidir>`, a directory will be created by the script.
```
python imgs2mpis.py <your_scenedir> <your_mpidir> \
    [--checkpoint CHECKPOINT] \
    [--factor FACTOR] [--width WIDTH] [--height HEIGHT] [--numplanes NUMPLANES] \
    [--disps] [--psvs] 
```
You should set at most one of `factor`, `width`, or `height` to determine the output MPI resolution (factor will scale the input image size down an integer factor, eg. 2, 4, 8, and height/width directly scale the input images to have the specified height or width). `numplanes` is 32 by default. `checkpoint` is set to the downloaded checkpoint by default.

Example usage:
```
python imgs2mpis.py scenedir scenedir/mpis --height 360
```

#### 3. Render novel views

You can either generate a list of novel view camera poses and render out a video, or you can load the saved MPIs in our interactive OpenGL viewer.

#### Generate poses for new view path
First, generate a smooth new view path by calling
```
python imgs2renderpath.py <your_scenedir> <your_posefile> \
	[--x_axis] [--y_axis] [--z_axis] [--circle][--spiral]
```
`<your_posefile>` is the path of an output .txt file that will be created by the script, and will contain camera poses for the rendered novel views. The five optional arguments specify the trajectory of the camera. The xyz-axis options are straight lines along each camera axis respectively, "circle" is a circle in the camera plane, and "spiral" is a circle combined with movement along the z-axis.  

Example usage:
```
python imgs2renderpath.py scenedir scenedir/spiral_path.txt --spiral
```
See `llff/math/pose_math.py` for the code that generates these path trajectories.

#### Render video with CUDA
You can build this in the `cuda_renderer/` directory by calling `make`.

Uses CUDA to render out a video. Specify the height of the output video in pixels (-1 for same resolution as the MPIs), the factor for cropping the edges of the video (default is 1.0 for no cropping), and the compression quality (crf) for the saved MP4 file (default is 18, lossless is 0, reasonable is 12-28).
```
./cuda_renderer mpidir <your_posefile> <your_videofile> height crop crf
```
`<your_videofile>` is the path to the video file that will be written by FFMPEG.

Example usage:
```
./cuda_renderer scenedir/mpis scenedir/spiral_path.txt scenedir/spiral_render.mp4 -1 0.8 18
```


#### Render video with Tensorflow
Use Tensorflow to render out a video (~100x slower than CUDA renderer). Optionally, specify how many MPIs are blended for each rendered output (default is 5) and what factor to crop the edges of the video (default is 1.0 for no cropping).
```
python mpis2video.py <your_mpidir> <your_posefile> videofile [--use_N USE_N] [--crop_factor CROP_FACTOR]
```
Example usage:
```
python mpis2video.py scenedir/mpis scenedir/spiral_path.txt scenedir/spiral_render.mp4 --crop_factor 0.8
```


#### Interactive OpenGL viewer

Controls:
- ESC to quit
- Move mouse to translate in camera plane
- Click and drag to rotate camera
- Scroll to change focal length (zoom)
- 'L' to animate circle render path

<img src='imgs/viewer.gif'/>

*The OpenGL viewer cannot be used in the Docker container.*

You need OpenGL installed, particularly GLFW:
```
sudo apt-get install libglfw3-dev
```

You can build the viewer in the `opengl_viewer/` directory by calling `make`.  

General usage (in `opengl_viewer/` directory) is 
```
./opengl_viewer mpidir
```






## Citation

If you find this useful for your research, please cite the following paper.

```
@article{mildenhall2019llff,
  title={Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines},
  author={Ben Mildenhall and Pratul P. Srinivasan and Rodrigo Ortiz-Cayon and Nima Khademi Kalantari and Ravi Ramamoorthi and Ren Ng and Abhishek Kar},
  journal={ACM Transactions on Graphics (TOG)},
  year={2019},
}
```
