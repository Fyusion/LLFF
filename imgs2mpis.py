import numpy as np
import os
import sys
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from llff.poses.pose_utils import load_data
from llff.inference.mpi_utils import run_inference
from llff.inference.mpi_tester import DeepIBR

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('scenedir', type=str,
                    help='input scene directory')
parser.add_argument('mpidir', type=str,
                    help='output mpi directory')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/papermodel/checkpoint', 
                    help='pretrained network checkpoint')
parser.add_argument('--factor', type=int,
                    help='integer downsampling factor for input images')
parser.add_argument('--width', type=int,
                    help='output mpi width (pixels)')
parser.add_argument('--height', type=int,
                    help='output mpi height (pixels)')
parser.add_argument('--numplanes', type=int, default=32,
                    help='output mpi depth')
parser.add_argument('--psvs', action='store_true')
parser.add_argument('--no_mpis', action='store_true')
args = parser.parse_args()


    
def gen_mpis(basedir, savedir, fwh, logdir, num_planes, no_mpis, disps, psvs):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # load up images, poses, w/ scale factor
    poses, bds, imgs = load_data(basedir, *fwh)
    
    # load up model
    ibr_runner = DeepIBR()
    ibr_runner.load_graph(logdir)
    
    patched = imgs.shape[0] * imgs.shape[1] * num_planes > 640*480*32
    
    N = imgs.shape[-1]
    close_depths = [bds.min()*.9] * N
    inf_depths = [bds.max()*2.] * N
    mpi_bds = np.array([close_depths, inf_depths])
    
    mpis = run_inference(imgs, poses, mpi_bds, ibr_runner, num_planes, patched, disps=disps, psvs=psvs)
    
    for i in range(N):
        if not os.path.exists('{}/mpi{:02d}'.format(savedir, i)):
            os.makedirs('{}/mpi{:02d}'.format(savedir, i))
        if not no_mpis: 
            mpis[i].save('{}/mpi{:02d}'.format(savedir, i), False, True)  
        if disps:
            plt.imsave(os.path.join(savedir, 'mpi{:02d}/disps.png'.format(i)), mpis[i].disps)                
        if psvs:
            psv = np.moveaxis(mpis[i].psv, -2, 0)
            psv = (255*np.clip(psv,0,1)).astype(np.uint8)
            imageio.mimwrite(os.path.join(savedir, 'mpi{:02d}/psv.mp4'.format(i)), psv, fps=30, quality=5)
        
    with open(os.path.join(savedir, 'metadata.txt'), 'w') as file:
        file.write('{} {} {} {}\n'.format(N, imgs.shape[1], imgs.shape[0], num_planes))
    
    print( 'Saved to', savedir )
    return True
    


if __name__=='__main__':
    fwh = [args.factor, args.width, args.height]
    print('factor/width/height args:', fwh)
    if args.factor is None and args.width is None and args.height is None:
        fwh = [1, None, None]
    
    gen_mpis(args.scenedir, args.mpidir, fwh, args.checkpoint, 
             args.numplanes, args.no_mpis, True, args.psvs)

    print( 'Done with imgs2mpis' )