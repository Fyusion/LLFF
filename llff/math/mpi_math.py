import tensorflow as tf
import numpy as np
import os


ALPHA_EPS = 1e-10


##########################
#  Homography/matrix math for MPIs and plane sweep volumes
##########################


# Don't remember why I redefined matrix multiply 
# but I'm sure I had a good reason at the time
def tfmm(A, B):
    with tf.variable_scope('tfmm'):
        return tf.reduce_sum(A[..., :, :, tf.newaxis] * B[..., tf.newaxis, :, :], axis=-2)

        
# Converts my pose format into 4x4 extrinsic and 3x3 intrinsic matrices
#   My rotations are in [down, right, backwards] orientation, 
#   hence the 'fix_yx' thing to convert from that format [-y x z] to the more conventional [x y z]
def myposes2mats(poses, fix_yx=False):
    with tf.variable_scope('myposes2mats'):
        def cat(arrs, ax):
            return tf.concat(arrs, ax)
        def stk(arrs, ax):
            return tf.stack(arrs, ax)

        c2w = poses[..., :3, :4] + 0.
        bottom0 = tf.zeros_like(c2w[..., :1, :3])
        bottom1 = tf.ones_like(c2w[..., :1, 3:4])
        bottom = tf.concat([bottom0, bottom1], -1)
        # fix the -y x thing
        if fix_yx:
            c2w = cat([c2w[..., :3, 1:2], -c2w[..., :3, 0:1], c2w[..., :3, 2:]], -1)

        R = c2w[..., :3, :3] + 0. 
        t = c2w[..., :3, 3:4] + 0.
        T_c2w = cat([c2w, bottom], -2)

        N = len(R.get_shape().as_list())
        perm = list(range(N-2)) + [N-1, N-2]
        R_inv = tf.transpose(R, perm)
        t_inv = -tfmm(R_inv, t)
        T_w2c = cat([cat([R_inv, t_inv], -1), bottom], -2)

        h, w, f = poses[..., 0, -1], poses[..., 1, -1], poses[..., 2, -1]
        m_z, m_o = tf.zeros_like(h), tf.ones_like(h)

        sh = tf.shape(poses)[:-2]
        sh = tf.concat([sh, tf.constant([3]), tf.constant([3])], -1)
        with tf.variable_scope('Kstuff'):
            K = stk([f, m_z, -w*.5, 
                     m_z, f, -h*.5,
                     m_z, m_z, -m_o], -1)
            K = tf.reshape(K, sh)
            
            K_inv = stk([1./f, m_z, -w*.5/f,
                         m_z, 1./f, -h*.5/f,
                         m_z, m_z, -m_o], -1)
            K_inv = tf.reshape(K_inv, sh)
            
        T_c2w, T_w2c, K, K_inv = map(lambda x : tf.cast(x, dtype=poses.dtype), [T_c2w, T_w2c, K, K_inv])
        return T_c2w, T_w2c, K, K_inv





# To warp a single plane, for PSV creation and MPI rendering
def plane_homogs(pose_t, pose_s, depths, planes_from_t=True, y_flip=True, fix_yx=False):
    with tf.variable_scope('plane_homogs'):
        T_t2w, _, _, K_t_inv = myposes2mats(pose_t, fix_yx=fix_yx)
        _, T_w2s, K_s, _ = myposes2mats(pose_s, fix_yx=fix_yx)

        T_t2s = tfmm(T_w2s, T_t2w)
        R = T_t2s[..., tf.newaxis, :3,:3]
        t = T_t2s[..., tf.newaxis, :3, 3:4]
        n = tf.constant(np.array([0,0,1.]), dtype=T_t2s.dtype)
        n = tf.reshape(n, [3,1])
        nT = tf.transpose(n)
        a = tf.reshape(depths, [-1, 1, 1])

        if planes_from_t:
            H = R - tfmm(t, nT) / a
        else:
            H = R - tfmm(t, tfmm(nT, R)) / (a + tfmm(nT, t))

        H = tfmm(K_s[..., tf.newaxis,:,:], tfmm(H, K_t_inv[..., tf.newaxis,:,:]))
        if y_flip:
            premat = tf.stack([1,0,0., 0,-1,pose_t[0,-1]-1, 0,0,1.],-1)
            premat = tf.cast(premat, dtype=T_t2s.dtype)
            premat = tf.reshape(premat, [3,3])
            
            postmat = tf.stack([1,0,0., 0,-1,pose_s[0,-1]-1, 0,0,1.],-1)
            postmat = tf.cast(postmat, dtype=T_t2s.dtype)
            postmat = tf.reshape(postmat, [3,3])
            
            H = tfmm(postmat, tfmm(H, premat))
        return H
    
    

    
# Clone of tf.contrib.image.transform since that fn did not work on the RTX 2080 Ti
def homog_warp(img, H, retcos=False, window=None):
    with tf.variable_scope('homog_warp'):  
        sh = tf.shape(img)
        h, w = sh[-3], sh[-2]

        H = tf.reshape(tf.concat([H, tf.ones_like(H[:, :1])], -1), [-1,3,3])
        
        bds = tf.stack([0,0,h,w], 0)
        if window is not None:
            bds = tf.cond(window[3] > 0, lambda : window, lambda : bds)
        coords = tf.meshgrid(tf.range(bds[0], bds[2]), tf.range(bds[1], bds[3]), indexing='ij')
        # coords = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
        coords = tf.cast(tf.stack([coords[1], coords[0]], 0), H.dtype) # [2, H, W]

        coords_t = tf.concat([coords, tf.ones_like(coords[:1, ...])], 0) # [3, H, W]
        coords_t = tf.reshape(H, [-1,3,3,1,1]) * coords_t # [-1, 3, 3, H, W]
        coords_t = tf.reduce_sum(coords_t, -3) # [-1, 3, H, W]
        coords_t = coords_t[...,:2,:,:] / coords_t[...,-1:,:,:] # [-1, 2, H, W]
        
        warp = tf.transpose(coords_t, [0,2,3,1]) # [-1, H, W, 2]
        rect_tf = tf.squeeze(tf.contrib.resampler.resampler(img, warp))

        rets = rect_tf 
        if retcos:
            rets = [rect_tf, coords_t]
        return rets


# homog_render_fn = lambda data, homog : tf.contrib.image.transform(data, homog, interpolation='BILINEAR') 
homog_render_fn = lambda data, homog, window=None : homog_warp(data, homog, window=window)


def render_mpi_homogs(mpi_rgba, pose, newpose, min_disp, max_disp, num_depths, debug=False):  
    with tf.variable_scope('render_mpi_homogs'):  
        
        outs = {}
        dispvals = tf.linspace(min_disp, max_disp, num_depths)

        H = plane_homogs(newpose, pose, 1./dispvals, planes_from_t=False, y_flip=True, fix_yx=True)
        H = tf.reshape(H, [-1, 9])
        H = H[:, :8] / H[:, 8:]

        data_in = tf.transpose(tf.squeeze(mpi_rgba), [2,0,1,3])
        window = tf.cast([0,0,newpose[0,-1],newpose[1,-1]], tf.int32)
        mpi_reproj = homog_render_fn(data_in, H, window=window) 
        mpi_reproj = tf.transpose(mpi_reproj, [1,2,0,3])[tf.newaxis, ...]
        
        # back to front compositing 
        mpiR_alpha = mpi_reproj[..., 3:4] # 1 H W D 1
        mpiR_color = mpi_reproj[..., 0:3] # 1 H W D 3

        # Add small ALPHA_EPS to prevent gradient explosion nans from tf.cumprod derivative
        weights = mpiR_alpha * tf.cumprod(1.-mpiR_alpha + ALPHA_EPS, -2, exclusive=True, reverse=True) # goddamnit pratul
        alpha_acc = tf.reduce_sum(weights[..., 0], -1)
        rendering = tf.reduce_sum(weights * mpiR_color, -2)
        accum = tf.cumsum(weights * mpiR_color, -2, reverse=True)

        return rendering, alpha_acc, accum
    
    
# Create a plane sweep volume ('cost volume')
# vectorized on initial axes of img and pose
def make_psv_homogs(img, pose, newpose, dispvals, num_depths, window=None): 
    with tf.variable_scope('make_cv_homogs'):  

        sh = tf.shape(pose)[:-2] # [N, 3,4]
        H = plane_homogs(newpose, pose, 1./dispvals, planes_from_t=True, y_flip=True, fix_yx=True) # [N, D, 3, 3]
        H = tf.reshape(H, [-1, 9]) # [N*D, 9]
        H = H[:, :8] / H[:, 8:]
    
        img_sh = tf.shape(tf.squeeze(img))
        img = tf.reshape(img, [-1, 1, img_sh[-3], img_sh[-2], img_sh[-1]]) # [N, 1, H, W, 3] 
        img_tiled = tf.tile(img, [1, num_depths, 1, 1, 1]) # go in as [N, D, H, W, 3]
        img_tiled = tf.reshape(img_tiled, [-1, img_sh[-3], img_sh[-2], img_sh[-1]]) # [N*D, H, W, 3] 

        cvd = homog_render_fn(img_tiled, H, window=window) # come out as [N*D, H, W, 3]
        h, w = img_sh[-3], img_sh[-2]
        if window is not None:
            h = tf.cond(window[3] > 0, lambda : window[2]-window[0], lambda : h)
            w = tf.cond(window[3] > 0, lambda : window[3]-window[1], lambda : w)
        cvd = tf.reshape(cvd, [-1, num_depths, h, w, img_sh[-1]]) # [N, D, H, W, 3]
        cvd = tf.squeeze(tf.transpose(cvd, [2,3,1,4,0])) # [H, W, D, 3, N] or [H, W, D, 3]

        return cvd
    
    

    

##########################
#
#   3d projection math for traditional depth map backwards warping
#
##########################


class Pose:
    def __init__(self, m, window=None): #, h, w, focal, mat):
        self.h = m[0,4]
        self.w = m[1,4]
        self.f = m[2,4]
        self.mat = m[:3,:4]
        
        R = self.mat[:3, :3]
        t = self.mat[:3, 3]
        RT = tf.transpose(R)
        self.invmat = tf.concat([RT, -tf.reduce_sum(RT*t, axis=-1, keep_dims=True)], axis=-1)
        
        if window is None:
            w0, h0 = 0., 0.
            w1, h1 = self.w, self.h
        else:
            window = tf.cast(window, m.dtype)
            h0 = tf.cond(window[3] > 0, lambda : window[0], lambda : 0.)
            w0 = tf.cond(window[3] > 0, lambda : window[1], lambda : 0.)
            h1 = tf.cond(window[3] > 0, lambda : window[2], lambda : self.h) - h0
            w1 = tf.cond(window[3] > 0, lambda : window[3], lambda : self.w) - w0
            # w0, h0 = window[1], window[0]
            # w1, h1 = window[3], window[2]
        self.w0, self.h0, self.w1, self.h1 = w0, h0, w1, h1
        
        self.xf = (self.w-0) / self.f * .5
        self.yf = (self.h-0) / self.f * .5
        yvals = (tf.to_float(tf.range(h0, h0+h1)) - self.h*.5) / self.f
        xvals = (tf.to_float(tf.range(w0, w0+w1)) - self.w*.5) / self.f
        co_y, co_x = tf.meshgrid(yvals, xvals, indexing='ij')
        
        self.cam_co = tf.stack([co_y, co_x, -tf.ones_like(co_y)], axis=-1)
        
    def matmul(self, M, pts):
        pts = tf.concat([pts, tf.ones_like(pts[...,:1])], axis=-1)
        pts = tf.reduce_sum(M * tf.expand_dims(pts, -2), axis=-1)
        return pts

    def world2cam(self, pts):
        return self.matmul(self.invmat, pts)
    
    def cam2world(self, pts):
        return self.matmul(self.mat, pts)
        
        
    def project_out(self, depths, single=False):
        if not single:
            depths = tf.expand_dims(depths, axis=-1)
            
        pts = self.cam_co * depths
        pts = self.cam2world(pts)
        return pts
        
    
    def project_in(self, pts, clip=False):
        
        pts = self.world2cam(pts)
        pts = pts[..., :2] / -pts[..., 2:3]
        
        pts_y = (pts[...,0]/self.yf + 1.) * self.h * .5
        pts_x = (pts[...,1]/self.xf + 1.) * self.w * .5
        if clip:
            pts_y = tf.clip_by_value(pts_y, 0., self.h-1.)
            pts_x = tf.clip_by_value(pts_x, 0., self.w-1.)
        ret = tf.stack([pts_y, pts_x], axis=-1)

        return ret


def warp(img, pose, newpose, depth, single=False, debug=False):
    pts = newpose.project_out(depth, single)
    pix = pose.project_in(pts, clip=False)
    
    pix = tf.expand_dims(pix, 0)
    pix_y, pix_x = pix[...,0], pix[...,1]
    
    mask_y = tf.logical_and(pix_y <= pose.h-1., pix_y >= 0.)
    mask_x = tf.logical_and(pix_x <= pose.w-1., pix_x >= 0.)
    mask = tf.to_float(tf.logical_and(mask_x, mask_y))
    
    pix_y = tf.clip_by_value(pix_y, 0., pose.h-1.)
    pix_x = tf.clip_by_value(pix_x, 0., pose.w-1.)
    output = tf.contrib.resampler.resampler(img, tf.stack([pix_x, pix_y], -1))
    output_masked = tf.expand_dims(mask, -1) * output
    
    if debug:
        return output_masked, mask, pix, output
    else:
        return output_masked, mask
    
    