from __future__ import print_function

import numpy as np
import os
import sys
import imageio
import time

from llff.poses.pose_utils import load_data
from llff.inference.mpi_utils import load_mpis, render_mpis


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('mpidir', type=str,
                    help='input mpi directory')
parser.add_argument('posefile', type=str,
                    help='input render poses file')
parser.add_argument('videofile', type=str,
                    help='output video file')
parser.add_argument('--use_N', type=int, default=5,
                    help='number of mpis to blend per output frame')
parser.add_argument('--crop_factor', type=float, default=1.,
                    help='field-of-view cropping factor (1 is uncropped)')
args = parser.parse_args()

def load_poses(filename):
    
    if filename.endswith('npy'):
        return np.load(filename)
    
    elif filename.endswith('txt'):
        with open(filename, 'r') as file:
            file.readline()
            x = np.loadtxt(file)
        x = np.transpose(np.reshape(x, [-1,5,3]), [0,2,1])
        x = np.concatenate([-x[...,1:2], x[...,0:1], x[...,2:]], -1)
        return x
    
    print('Incompatible pose file {}, must be .txt or .npy'.format(filename))
    return None
    

def render_video(mpidir, renderpathfile, outputfile, use_N):
    
    mpis = load_mpis(mpidir)
    print('Loaded {} mpis, each with shape {}'.format(len(mpis), mpis[0].mpi.shape))

    render_poses = load_poses(renderpathfile)
    if render_poses is None:
        return
    render_poses[:,:,-1] = mpis[0].pose[:,-1]
    render_poses[:,:2,-1] = np.round(render_poses[:,:2,-1] * args.crop_factor)
    print('Rendering {} poses, crop factor {}, res {}, mpi shape {}'.format(
        render_poses.shape[0], args.crop_factor, render_poses[0,:2,-1].astype(np.int32), mpis[0].mpi.shape))
    
    outframes = render_mpis(mpis, render_poses, use_N=use_N, weight_scale=1., alpha_blend=True)

    # Save to video
    outframes8 = (255*outframes).astype(np.uint8)
    imageio.mimwrite(outputfile, outframes8, quality=7, fps=30)

    print('Done')
    
    
if __name__=='__main__':
    render_video(args.mpidir, args.posefile, args.videofile, args.use_N)