#include <stdio.h>
#include <iostream>
#include <math.h>

struct MPI;

struct MPIMeta {
    
    // Host mem
    uint8_t* data;
    float *c2ws, *cif;
    unsigned int width, height, N_nodes, planes;
    int N_blend;
    
    // Device mem
    uint8_t* d_data;
    float *d_c2ws, *d_cif;
    MPI* d_mpis;
    
    // Load from disk
    void load_all_mpis(const char* basedir);
    
    // Send to gpu
    void mpis2gpu();
    
    // Render
    MPI **mpi_list, **d_mpi_list;
    float *weights, *d_weights;
    
    void set_blend_weights(const float* pose_arr);
    void render_pose(const float* pose_arr, uint8_t* d_out_arr, 
                     int imgH, int imgW, float imgF, int N_blend_);
    
    void render_pose_lf(const float* pose_arr, uint8_t* d_out_arr, 
                 int imgH, int imgW, float imgF, int N_blend_);
    
};
