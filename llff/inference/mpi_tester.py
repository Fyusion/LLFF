import tensorflow as tf
tf.contrib.resampler # dumb but required to get tf.contrib to load
import numpy as np
import time, sys, os
from llff.math.mpi_math import render_mpi_homogs, make_psv_homogs


class DeepIBR():
    
    def __init__(self):
        self.loaded = None
        self.sess = None
        self.graph = tf.Graph()
        
        
        
    def setup_renderer(self):
        
        with self.graph.as_default():
            
            self.Sess()
            
            # mpi_rgba, pose, newpose, min_disp, max_disp, num_depths
            mpi_rgba = tf.placeholder(tf.float32, [None, None, None, 4], name='mpi_rgba')
            pose = tf.placeholder(tf.float32, [3, 5], name='pose')
            newpose = tf.placeholder(tf.float32, [3, 5], name='newpose')
            close_depth = tf.placeholder(tf.float32, [], name='close_depth')
            inf_depth = tf.placeholder(tf.float32, [], name='inf_depth')

            self.run_args = [mpi_rgba, pose, newpose, close_depth, inf_depth]

            self.render_args = [tf.expand_dims(mpi_rgba, 0), pose, newpose, 1./inf_depth, 1./close_depth, tf.shape(mpi_rgba)[-2]]
            self.rendered_result, self.rendered_alpha, _ = render_mpi_homogs(*self.render_args)
        
        
    def render_mpi(self, *args):
        fdict = {a : b for a, b in zip(self.run_args, args)}
        rgb, alpha = self.sess.run([self.rendered_result, self.rendered_alpha], feed_dict=fdict)
        rgb, alpha = np.squeeze(rgb), np.squeeze(alpha)
        return rgb, alpha
    
    
    
    
    def setup_lf_filter(self):
        
        with self.graph.as_default():
            
            self.Sess()
        
            # img, pose, newpose, dispvals, num_depths
            img = tf.placeholder(tf.float32, [None, None, None, None], name='lfi_img')
            pose = tf.placeholder(tf.float32, [None, 3, 5], name='lfi_pose')
            newpose = tf.placeholder(tf.float32, [3, 5], name='lfi_newpose')
            dispval = tf.placeholder(tf.float32, [], name='lfi_dispval')

            self.run_args_lf_filter = [img, pose, newpose, dispval]

            cvd = tf.squeeze(make_psv_homogs(img, pose, newpose, tf.reshape(dispval, [1]), 1))
            # cvd = tf.squeeze(tf.cond(tf.shape(img)[0] > 1, lambda : tf.reduce_mean(cvd, -1), lambda : cvd))
            self.lf_filter = cvd
            
            
    def render_lf_filter(self, *args):
        fdict = {a : b for a, b in zip(self.run_args_lf_filter, args)}
        return self.sess.run(self.lf_filter, feed_dict=fdict)
            
        
        
        
        
    def clearmem(self):
        self.sess.close()
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.setup_renderer()
            
    def Sess(self):
        if self.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement=True
            print( 'Creating session' )
            sess = tf.Session(config=config)
            self.sess = sess
        return self.sess
    
    def load_graph(self, dirpath, meta_only=False):
        
        ckpt_path = dirpath
        if os.path.isdir(ckpt_path):
            ckpt_path = tf.train.latest_checkpoint(ckpt_path)
        if ckpt_path is None:
            print( 'No ckpts found at ', dirpath )
            return None
        if self.loaded and self.loaded==ckpt_path:
            print( 'Already loaded', ckpt_path )
            return
        elif self.loaded and self.loaded[:self.loaded.rfind('/')]==ckpt_path[:ckpt_path.rfind('/')]:
            print( 'Already have graph for {}, loading weights {}', self.loaded, ckpt_path )
            self.load_weights(ckpt_path)
            return

        with self.graph.as_default():
            
            sess = self.Sess()

            if ckpt_path is not None:
                print( 'Restoring from',ckpt_path )
                self.saver = tf.train.import_meta_graph(ckpt_path + '.meta')
                t_vars = tf.trainable_variables()
                print( 'Meta restored' )
            else:
                print( 'No checkpoint found in {}'.format(ckpt_path) )
                return None

            var_col = tf.get_collection('inputs')
            var_col = [v for v in var_col if 'seq_to_use' not in v.name]
            print( 'Found inputs:' )
            print( [v.name for v in var_col] )
            
            self.fixvars = var_col

            output_ = tf.get_collection('outputs')
            clip_name = lambda n, s : n[:n.rfind(s)] if s in n else n
            outputs = {clip_name(clip_name(out.name, '_1'), ':0') : out for out in output_}
            print( 'Found outputs:' )
            print([k for k in sorted(outputs)])
            self.outputs = outputs
            
            self.setup_renderer()
            print( 'Setup renderer' )
            
            if meta_only:
                return
            
            self.saver.restore(sess, ckpt_path)
            print( 'Weights restored' )
            self.loaded = ckpt_path
            
            
            
    def load_weights(self, ckpt_path):
        if '.ckpt' not in ckpt_path:
            ckpt_path = tf.train.latest_checkpoint(ckpt_path)
        
        if self.loaded==ckpt_path:
            print( 'Already loaded', ckpt_path )
            return
        
        self.saver.restore(self.sess, ckpt_path)
        print( 'Weights restored from', ckpt_path )
        self.loaded = ckpt_path
        
            
    def run_inference(self,
        test_data,
        test_keys,
        true_fdicts=False,
        patched=False,
        valid=192, 
        buffer=32,
        verbose=True):
        
        sess = self.sess
        outputs = self.outputs
        
        def vprint(*args):
            if verbose:
                print(args)
        
        # vprint( 'Running test instead of train' )
        all_outputs = []
        if test_keys:
            outputs = {d : outputs[d] for d in test_keys if d in outputs}
            vprint( 'abdriged outputs to', outputs.keys() )
            
        for i, curr_np_inputs in enumerate(test_data):
            vprint( '{} of {}'.format(i, len(test_data)) )
            if true_fdicts:
                cni = curr_np_inputs
            else:
                cni = [x[np.newaxis, ...] for x in curr_np_inputs[:3]] + curr_np_inputs[3:]
                
            fdict = {a : b for a, b in zip(self.fixvars, cni)}
            if not patched:
                out = sess.run(outputs, feed_dict=fdict)
            else:
                # Need more complicated code for dealing with the 'patched' case
                sh = cni[0].shape[1:3]
                diam = valid+buffer*2
                windows = []
                N = 0
                for i in range(0, sh[0], valid):
                    N += 1
                    for j in range(0, sh[1], valid):
                        window = []
                        window += [max(0, i-buffer), max(0, j-buffer)]
                        window += [min(sh[0], window[0]+diam), min(sh[1], window[1]+diam)]
                        window += [i-window[0], j-window[1]]
                        window += [min(window[2],window[4]+valid), min(window[3],window[5]+valid)]
                        windows.append(window)
                M = len(windows)//N
                vprint( '{} gridded into {} x {}'.format(sh, N,M) )
                
                out = None
                for i in range(0, sh[0], valid):
                    out_row = None
                    for j in range(0, sh[1], valid):
                        vprint( '.' ) # rudimentary progress bar
                        window, windows = windows[0], windows[1:]
                        
                        fdict[self.fixvars[-1]] = window[0:4]
                        out_ = sess.run(outputs, feed_dict=fdict)

                        # crop patch to size
                        w = window[4:8]
                        out_ = {k : out_[k][:, w[0]:w[2], w[1]:w[3], ...] for k in out_}
                        out_row = out_ if out_row is None else {k : np.concatenate([out_row[k], out_[k]], 2) for k in out_row}
                    out = out_row if out is None else {k : np.concatenate([out[k], out_row[k]], 1) for k in out}
                    
            all_outputs.append(out)
        return all_outputs
        
        