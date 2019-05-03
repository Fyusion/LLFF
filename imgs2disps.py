import numpy as np
import os
import sys
import imageio

import llff.inference.mpi_tester as mpi_tester
from llff.poses.pose_utils import *



def nearest_pose(p, poses):
    dists = np.sum(np.square(p[:3, 3:4] - poses[:3, 3, :]), 0)
    return np.argsort(dists)


def run(scenedir, factor, savedir):
    print 'Running', scenedir, factor, savedir
    basedir = scenedir
    
    if basedir.endswith('/'):
        basedir = basedir[:-1]
    basename = os.path.basename(basedir)

    # basedir = '/home/bmild/triumph_1'

    # factor = 4
    num_planes = 32
    # num_planes = 128

    poses, bds, imgs = load_data(basedir, factor)

    apply_scaling=True
    if apply_scaling:
        z0 = bds[0, :].min()*.9
        f = poses[-1,-1,0]
        scaling_factor = f/z0
        print z0, f, scaling_factor
        
        
    ibr_runner = mpi_tester.DeepIBR()
    logdir = './checkpoints/papermodel/checkpoint'
    ibr_runner.load_graph(logdir)
    
    
    ###### MAKE MPIS
    
    k_scale = scaling_factor / num_planes

    patched = imgs.shape[0] * imgs.shape[1] * num_planes > 640*480*32

    buffer = 80
    valid = 504
    print imgs.shape, num_planes
    print patched, valid, buffer
    patched=True

    disps_all = []
    disps_norm = []

    for i in range(imgs.shape[-1]):

        neighs = list(nearest_pose(poses[..., i], poses)[:5])
        inds = neighs + [i]

        close_depth, inf_depth = bds[0, i].min()*.9, bds[1, i].max()*1.5
        inf_depth = 1e10

        inputs = [[imgs[...,inds], np.ones_like(imgs[...,0,inds]), poses[..., inds], num_planes, close_depth, inf_depth]]

        print( i, '(of {})'.format(imgs.shape[-1]), '<-', neighs, 'depths', close_depth, inf_depth )
        print inds

        outputs = ibr_runner.run_inference(inputs, test_keys=['disps'], 
                                           patched=patched, valid=valid, buffer=buffer, verbose=False)


        d = np.squeeze(outputs[0]['disps'][0])
        disps_norm.append(d)
        d = (1.-d) /inf_depth + d / close_depth
        disps_all.append(d+0.)
        
    
    disps_norm = np.array(disps_norm)
    disps_all = np.array(disps_all)
    
    path = os.path.join(savedir, basename + str(factor))
    print('Saving at', path)

    if not os.path.exists(path):
        os.makedirs(path)

    save_poses = np.moveaxis(poses, -1, 0)
    print save_poses.shape

    hwf = save_poses[0,:,-1]
    s = ''
    s += str(int(save_poses.shape[0])) + '\n'
    s += str(int(hwf[0])) + '\n'
    s += str(int(hwf[1])) + '\n'
    s += str(float(hwf[2])) + '\n'

    p2save = save_poses[:,:3,:4] + 0.
    p2save = np.concatenate([p2save[:,:,1:2],-p2save[:,:,0:1],p2save[:,:,2:4]],-1)
    flat= p2save.reshape([-1,12])
    s += '\n'.join([' '.join([str(x) for x in y]) for y in flat])

    with open(os.path.join(path, 'poses.txt'), 'w') as file:
        file.write(s)
    #     print 'done'

    for i, x in enumerate(np.moveaxis(imgs, -1, 0)):
        print i

        x = (255*x).astype(np.uint8)
        imageio.imwrite(os.path.join(path, 'img{:02d}.png'.format(i)), x)
        x = (255*disps_norm[i]).astype(np.uint8)
        imageio.imwrite(os.path.join(path, 'disp{:02d}.png'.format(i)), x)

        with open(os.path.join(path, 'disp{:02d}.b'.format(i)), 'wb') as f:
            f.write(np.array(disps_all.shape[1:3]).astype(np.int32).tobytes())
            f.write(disps_all[i].tobytes())

            
            
if __name__=="__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])