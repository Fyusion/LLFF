import numpy as np
import sys

from llff.math.pose_math import generate_render_path
from llff.poses.pose_utils import load_data

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('scenedir', type=str,
                    help='input scene directory')
parser.add_argument('outname', type=str,
                    help='output .npy filename')

parser.add_argument('--x_axis', action='store_true')
parser.add_argument('--y_axis', action='store_true')
parser.add_argument('--z_axis', action='store_true')
parser.add_argument('--circle', action='store_true')
parser.add_argument('--spiral', action='store_true')

args = parser.parse_args()

comps = [
    args.x_axis,
    args.y_axis,
    args.z_axis,
    args.circle,
    args.spiral,
]
if any(comps) is False:
    comps = [True] * 5
print('Path components', comps)

poses, bds = load_data(args.scenedir, load_imgs=False)

render_poses = generate_render_path(poses, bds, comps, N=30)

if args.outname.endswith('txt'):
    
    render_poses = np.concatenate([render_poses[...,1:2], 
                                  -render_poses[...,0:1], 
                                   render_poses[...,2:]], -1)
    
    str_out = '{}\n'.format(render_poses.shape[0])
    for p in render_poses:
        str_out += ' '.join(['{}'.format(x) for x in p.T.ravel()]) + '\n'
    open(args.outname, 'w').write(str_out)
    
elif args.outname.endswith('npy'):
    np.save(args.outname, render_poses)
    
else:
    print('Output file {} does not end in .txt or .npy'.format(args.outname))
    sys.exit(-1)
    
print('Saved to', args.outname)