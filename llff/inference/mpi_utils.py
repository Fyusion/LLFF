from __future__ import print_function

import numpy as np
import os, sys, imageio, time

from llff.inference.mpi_tester import DeepIBR
    

def savempi(mpi, pose, dvals, basedir, txt_only=False, binary=False):
    pose = pose + 0.
    pose_ = pose + 0.
    pose[:, 0] = pose_[:, 1]
    pose[:, 1] = -pose_[:, 0]
    
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    with open(os.path.join(basedir, 'metadata.txt'), 'w') as file:
        file.write('{} {} {}\n'.format(*[pose[i,-1] for i in range(3)]))
        for j in range(4):
            file.write('{} {} {}\n'.format(*[pose[i,j] for i in range(3)]))
            
        file.write('{} {}\n'.format(dvals[0], dvals[-1]))
    
    if txt_only:
        return
    
    if binary:
        x = (255 * np.clip(np.transpose(mpi,[2,0,1,3]),0,1)).astype(np.uint8)
        with open(os.path.join(basedir, 'mpi.b'), 'wb') as f:
            f.write(x.tobytes())
    else:
        for i in range(mpi.shape[-2]):
            x = (255 * np.clip(mpi[..., i, :],0,1)).astype(np.uint8)
            imageio.imwrite(os.path.join(basedir, 'mpi{:02d}.png'.format(i)), x)
        
    
    
    


class MPI:
    
    def __init__(self, imgs, poses, cdepth, idepth):
        self.imgs = imgs
        self.poses = poses
        self.pose = poses[..., 0]
        self.cdepth = cdepth
        self.idepth = idepth
        self.args = [None, self.pose, None, cdepth, idepth]
        
    def generate(self, generator, num_planes):
        inputs = [[self.imgs, np.ones_like(self.imgs[...,0]), self.poses, num_planes, self.cdepth, self.idepth]]
        
        outputs = generator(inputs)
        
        self.mpi = np.squeeze(outputs[0]['mpi0'][0]) 
        if 'disps' in outputs[0]:
            self.disps = np.squeeze(outputs[0]['disps'][0])
        if 'psv' in outputs[0]:
            self.psv = np.squeeze(outputs[0]['psv'])
        self.args[0] = self.mpi
        
    
    def render(self, pose, ibr_runner):
        self.args[2] = pose
        rendering, alpha = ibr_runner.render_mpi(*self.args)
        return np.concatenate([rendering, alpha[..., np.newaxis]], -1)


    def save(self, basedir, txt_only=False, binary=False):
        mpi = self.mpi
        pose = self.pose + 0.
        pose = np.concatenate([pose[:,1:2], -pose[:,0:1], pose[:,2:]], 1)
        
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        with open(os.path.join(basedir, 'metadata.txt'), 'w') as file:
            file.write('{} {} {} {}\n'.format(mpi.shape[0], mpi.shape[1], mpi.shape[2], pose[2,-1]))
            for j in range(4):
                file.write('{} {} {}\n'.format(*[pose[i,j] for i in range(3)]))
                
            file.write('{} {}\n'.format(self.idepth, self.cdepth))
        
        if txt_only:
            return
        
        if binary:
            x = (255 * np.clip(np.transpose(mpi,[2,0,1,3]),0,1)).astype(np.uint8)
            with open(os.path.join(basedir, 'mpi.b'), 'wb') as f:
                f.write(x.tobytes())
        else:
            for i in range(mpi.shape[-2]):
                x = (255 * np.clip(mpi[..., i, :],0,1)).astype(np.uint8)
                imageio.imwrite(os.path.join(basedir, 'mpi{:02d}.png'.format(i)), x)
        
    
def load_mpi(basedir):
    metadata = os.path.join(basedir, 'metadata.txt')
    mpibinary = os.path.join(basedir, 'mpi.b')
    lines = open(metadata, 'r').read().split('\n')
    h, w, d = [int(x) for x in lines[0].split(' ')[:3]]
    focal = float(lines[0].split(' ')[-1])
    data = np.frombuffer(open(mpibinary, 'rb').read(), dtype=np.uint8)/255.
    data = data.reshape([d,h,w,4]).transpose([1,2,0,3])
    data[...,-1] = np.minimum(1., data[...,-1]+1e-8)
    
    pose = np.array([[float(x) for x in l.split(' ')] for l in lines[1:5]]).T
    pose = np.concatenate([pose, np.array([h,w,focal]).reshape([3,1])], -1)
    pose = np.concatenate([-pose[:,1:2], pose[:,0:1], pose[:,2:]], 1)
    idepth, cdepth = [float(x) for x in lines[5].split(' ')[:2]]
    
    ret = MPI(None, pose[...,np.newaxis], cdepth, idepth)
    ret.mpi = data
    ret.args[0] = ret.mpi
    
    return ret

def load_mpis(mpidir):
    
    with open(os.path.join(mpidir, 'metadata.txt'), 'r') as file:
        line = file.readline()
        N = int(line.split(' ')[0])
    
    mpis = []
    for i in range(N):
        mpis.append(load_mpi('{}/mpi{:02d}'.format(mpidir, i)))
        
    return mpis
    

def run_inference(imgs, poses, mpi_bds, ibr_runner, num_planes, patched=False, disps=False, psvs=False):
    keys = ['mpi0', 'disps', 'psv']
        
    generator = lambda inputs : ibr_runner.run_inference(inputs, test_keys=keys, patched=patched, valid=270, buffer=80)
    mpis = []
    for i in range(imgs.shape[-1]):
        
        close_depth, inf_depth = mpi_bds[0,i], mpi_bds[1,i]
        
        # Sort inputs according to distance
        dists = np.sum(np.square(poses[:3, 3, i:i+1] - poses[:3, 3, :]), 0)
        neighs = np.argsort(dists)
        
        neighs = list(neighs[1:])
        if len(neighs) < 4:
            neighs += [i] * (4 - len(neighs))
            print( 'Had to extend to', neighs )
        
        # We always use 5 inputs now
        inds = [i] + list(neighs[:4]) + [i]
        print( i, '(of {})'.format(imgs.shape[-1]), '<-', inds, 'depths', close_depth, inf_depth )
        
        mpi = MPI(imgs[..., inds], poses[..., inds], close_depth, inf_depth)
        mpi.generate(generator, num_planes)
        mpis.append(mpi)

    return mpis
    

def exp_weight_fn(dists, k):
    return np.exp(-k * dists)
    
def render_mpis(mpis, render_poses, use_N, 
                weight_scale=1., alpha_blend=True, weight_fn=exp_weight_fn):

    poses = np.stack([mpi.pose for mpi in mpis], -1)

    z0 = np.min([mpi.cdepth for mpi in mpis])  # near distance
    f = mpis[0].pose[-1,-1]       # focal length
    scaling_factor = f/z0    # converts meters to units of disparity
    
    outframes = []
    ibr_runner = DeepIBR()
    ibr_runner.setup_renderer()
    
    num_planes = mpis[0].mpi.shape[-2]
    k_scale = scaling_factor / num_planes * weight_scale

    print('Beginning rendering ({} total)'.format(render_poses.shape[0]))
    t = time.time()
    for i, p in enumerate(render_poses):
        print(i, end=', ')
        sys.stdout.flush()
        
        # Blending weights, based on distance to 5 nearest inputs
        dists = np.sqrt(np.sum(np.square(p[:3, 3:4] - poses[:3, 3, :]), 0))
        inds = np.argsort(dists)[:use_N]
        weights = weight_fn(dists[inds], k_scale).reshape([-1,1,1,1])

        # Generate 5 MPI renderings
        rends = np.stack([mpis[ind].render(p, ibr_runner) for ind in inds], 0)
        if alpha_blend:
            outframe = (rends[...,:3]*weights).sum(0) / (1e-10+(rends[...,-1:]*weights).sum(0)) 
        else:
            outframe = ((rends[...,:3] / (1e-10+rends[...,-1:])) * weights).sum(0)
        outframes.append(outframe)
    print('Finished rendering, {} secs'.format(time.time() - t))

    outframes = np.stack(outframes, 0)
    return outframes
