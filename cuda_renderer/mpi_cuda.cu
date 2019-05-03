#include "mpi_cuda.h"


// includes, project
#include <cuda.h>
// #include <cuda_runtime.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>


#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define GLM_FORCE_SWIZZLE
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp> 

#include <algorithm>
#include <vector>
#include <fstream>


using namespace std;
using namespace glm;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}





struct MPI {
    __host__ __device__
    glm::vec4 bilerp(int z, float y, float x) {
        uint8_t* img = data + z * W * H * 4;
        
//         float yt_ = y-floorf(y);
//         float xt_ = x-floorf(x);
//         float yt[] = {1.f-yt_, yt_};
//         float xt[] = {1.f-xt_, xt_};
        
//         glm::vec4 ret(0);
        // for (int i = 0; i < 2; ++i) {
        //     for (int j = 0; j < 2; ++j) {
        //         int yy=floorf(y)+i, xx=floorf(x)+j;
        //         glm::vec4 rgba(0);
        //         if (x >= 0 && x < W && y >= 0 && y < H)
        //             for (int c = 0; c < 4; ++c)
        //                 rgba[c] = *(img + 4 * (yy*W + xx) + c)/255.f;
        //         ret += rgba * yt[i] * xt[j];
        //     }
        // }
        // return ret;
        
        glm::vec4 rgba(0);
        int yy=floorf(y), xx=floorf(x);
        if (x >= 0 && x < W && y >= 0 && y < H)
            for (int c = 0; c < 4; ++c)
                rgba[c] = *(img + 4 * (yy*W + xx) + c)/255.f;
        return rgba;
        
    }
    __host__ __device__
    glm::vec4 isect_camspace(int z, glm::vec3 origin, glm::vec3 dir) {
        float zt = z / (float)(D-1.f);
        float z_plane = -1./((1.f-zt)/idepth + zt/cdepth);
        float t = (z_plane - origin[2]) / dir[2];
        glm::vec3 pt = origin + t * dir;
        pt = pt / -pt[2] * focal ;
        float y = -pt[1] + H*.5;
        float x = pt[0] + W*.5;
        
        return bilerp(z, y, x);
    }
    __host__ __device__
    glm::vec4 alpha_trace(glm::vec3 origin, glm::vec3 dir) {
        
        origin = vec3(w2c * vec4(origin, 1.));
        dir = vec3(w2c * vec4(dir, 0.));
        
        glm::vec4 accum(0);
        for (int z = 0; z < D; ++z) {
            glm::vec4 curr = isect_camspace(z, origin, dir);
            curr[3] = max((float)1e-5, curr[3]);
            glm::vec4 premult = vec4(
                curr[0]*curr[3], curr[1]*curr[3], curr[2]*curr[3], curr[3]);
            accum = premult + (1.f-premult[3]) * accum;
        }
        return accum;
    }
    
    __host__ __device__
    void init(uint8_t* data_, int D_, int H_, int W_,
             float cdepth_, float idepth_, float focal_,
             float* c2w_raw) {
        data = data_;
        D = D_;
        H = H_;
        W = W_;
        focal = focal_;
        cdepth = cdepth_;
        idepth = idepth_;
        c2w = glm::make_mat4(c2w_raw);
        w2c = glm::inverse(c2w);
    }
    
    
    uint8_t* data;
    int D, H, W;
    float cdepth, idepth, focal;
    
    glm::mat4 c2w, w2c;
    
};


__host__ __device__
glm::vec4 render_newpixel(glm::mat4 c2w, MPI* mpi, int y, int x, 
                          int H, int W, float focal) {
    glm::vec3 origin(0);
    glm::vec3 dir((x - W * .5f) / focal, -(y - H * .5f) / focal, -1.f);
    
    origin = vec3(c2w * vec4(origin, 1.f));
    dir = vec3(c2w * vec4(dir, 0.f));
    
    return mpi->alpha_trace(origin, dir);
}




__global__
void render_mpi(glm::mat4 new_c2w, int H, int W, float focal, 
                MPI* mpi, uint8_t* out_arr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= H || j >= W)
        return;
    
    glm::vec4 rgba = render_newpixel(new_c2w, mpi, i, j, H, W, focal);
    for (int c = 0; c < 3; ++c) {
        float x = rgba[c] / (.1f/255.f + rgba[3]);
        x = clamp(x, 0.f, 1.f);
        out_arr[3 * (W * i + j) + c] = 255 * x;
    }
}


__global__
void blend_mpis(glm::mat4 new_c2w, int H, int W, float focal, 
                MPI** mpis, float* weights, int N, uint8_t* out_arr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= H || j >= W)
        return;
    
    glm::vec4 rgba(0.);
    for (int k = 0; k < N; ++k) {
        rgba += render_newpixel(new_c2w, mpis[k], i, j, H, W, focal) * weights[k];
    }
    for (int c = 0; c < 3; ++c) {
        float x = rgba[c] / rgba[3];
        x = clamp(x, 0.f, 1.f);
        out_arr[3 * (W * i + j) + c] = 255 * x;
    }
}

__device__
void update_N_best(float* scores, int* inds, float score, int ind, int N) {
    int i = 0;
    while (i < N && (score < scores[i] || inds[i] == -1)) {
        ++i;
    }
    --i;
    if (i >= 0) {
        scores[i] = score;
        inds[i] = ind;
    }
}


__global__
void trace_mpis(glm::mat4 new_c2w, int H, int W, float focal, 
                MPI* mpis, int N_mpis, uint8_t* out_arr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= H || j >= W)
        return;
    
    const unsigned int N_blend = 3;
    float bdist[N_blend];
    int bind[N_blend];
    for (int k = 0; k < N_blend; ++k)
        bind[k] = -1;
    
    glm::vec3 p0 = glm::vec3(column(new_c2w, 3));
    
    for (int m = 0; m < N_mpis; ++m) {
        
        glm::vec3 origin(0);
        glm::vec3 dir((j - W * .5f) / focal, -(i - H * .5f) / focal, -1.f);

        origin = vec3(mpis[m].c2w * vec4(origin, 1.f));
        dir = vec3(mpis[m].c2w * vec4(dir, 0.f));
        
        glm::vec3 diff = p0 - origin;
        float t = dot(dir, diff) / dot(dir, dir);
        float dist = length(dir * t - diff);
        
        update_N_best(bdist, bind, dist, m, N_blend);
    }
    
    glm::vec4 rgba(0.);
    
    for (int k = 0; k < N_blend; ++k) {
        float w = exp(-bdist[k]);
        glm::vec4 curr = render_newpixel(new_c2w, &mpis[bind[k]], i, j, H, W, focal);
        // if (curr[3]==0.f)
        //     curr = vec4(0,0,0,1);
        rgba += curr * w;
    }
    
    float d = max(0.f, bdist[N_blend-1] - 1.f);
    float smooth_out = exp(-d);
    smooth_out = 1.;
    
    for (int c = 0; c < 3; ++c) {
        float x = rgba[c] / rgba[3] * smooth_out;
        x = clamp(x, 0.f, 1.f);
        out_arr[3 * (W * i + j) + c] = 255 * x;
    }
}



void load_mpi(const char* basedir, unsigned int width, unsigned int height, unsigned int num_planes,
              uint8_t* data, float* cam2world, float& close_depth, float& inf_depth, float& focal) {

    char filestr[1024];
    sprintf(filestr, "%s/metadata.txt", basedir);
    std::ifstream file(filestr);

    float hf, wf, df;
    file >> hf >> wf >> df >> focal;
    std::cout << "hf " << hf << " wf " << wf << " df " << df << " focal " << focal << std::endl;
    for (int i = 0; i < 16; ++i) {
        if (i%4 != 3) 
            file >> cam2world[i];
        else
            cam2world[i] = i < 15 ? 0. : 1.;
    }

    file >> inf_depth >> close_depth;
    std::cout << "depths " << inf_depth << " " << close_depth << std::endl;

    unsigned long BUFFERSIZE = width*height*4*num_planes;
    sprintf(filestr, "%s/mpi.b", basedir);
    FILE * filp = fopen(filestr, "rb"); 
    int bytes_read = fread(data, sizeof(uint8_t), BUFFERSIZE, filp);
    if (bytes_read != BUFFERSIZE) 
        cout << "Weeoooeeooo didn't read whole mpi" << endl;
}


__global__
void init_mpis(MPIMeta d_meta, MPI* d_mpis);


std::vector<std::pair<float, int>> pose_dists(glm::vec3 p, float* c2ws, int N) {
    std::vector<std::pair<float, int>> dists;
    for (int i = 0; i < N; ++i) {
        glm::vec3 q = glm::make_vec3(c2ws + i*16 + 12);
        float dist = glm::length(p-q);
        dists.push_back(std::pair<float, int>(dist, i));
    }
    std::sort(dists.begin(), dists.end());
    return dists;
}


void MPIMeta::load_all_mpis(const char* basedir) {
    char filestr[1024];
    sprintf(filestr, "%s/metadata.txt", basedir);
    std::ifstream file(filestr);
    int N_nodes;
    unsigned int width, height, num_planes;
    file >> N_nodes >> width >> height >> num_planes;
    std::cout << "loading " << N_nodes << " mpis" << std::endl;
    std::cout << width << " x " << height << std::endl;

    unsigned int IMGSIZE = width * height * 4;
    unsigned long BUFFERSIZE = IMGSIZE * num_planes;
    printf("%d %d %d %d\n", num_planes, height, width, N_nodes);
    cout << "Big request (host) " << BUFFERSIZE * N_nodes / (1 << 20) << " MB" << endl;
    uint8_t *mpi_data = new uint8_t [BUFFERSIZE * N_nodes];
    float *c2ws = new float [N_nodes * 16];
    float *cif = new float [N_nodes * 3];

    for (int i = 0; i < N_nodes; ++i) {
        char filestr[1024];
        sprintf(filestr, "%s/mpi%02d", basedir, i);
        std::cout << filestr << std::endl;

        load_mpi(filestr, width, height, num_planes,
                 mpi_data + BUFFERSIZE * i, 
                 c2ws + 16 * i, 
                *(cif + 3 * i), *(cif + 3 * i + 1), *(cif + 3 * i + 2));

    }
    // need to return
    // width, height, N_nodes, mpi_data, c2ws, cif
    this->data = mpi_data;
    this->c2ws = c2ws;
    this->cif = cif;
    this->width = width;
    this->height = height;
    this->N_nodes = N_nodes;
    this->planes = num_planes;
    this->N_blend = 0;
    cout << "Successfully loaded " << N_nodes << " mpis" << endl;

}
    
__device__
void init_gpu(MPIMeta meta, int i) {
    unsigned long vox = meta.width * meta.height * meta.planes;
    meta.d_mpis[i].init(meta.d_data + i*4*vox, 
                   meta.planes, meta.height, meta.width,
             meta.d_cif[i*3 + 0], 
             meta.d_cif[i*3 + 1], 
             meta.d_cif[i*3 + 2],
             meta.d_c2ws + i*16);
}

void MPIMeta::mpis2gpu()
{
    cout << "Mallocing" << endl;
    unsigned long vox = height * width * planes;
    cout << "Big request " << vox * 4 * N_nodes / (1 << 20) << " MB" << endl;
    gpuErrchk(cudaMalloc(&d_data, vox * 4 * N_nodes));
    gpuErrchk(cudaMalloc(&d_c2ws, N_nodes * 16 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_cif, N_nodes * 3 * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_mpis, sizeof(MPI) * N_nodes));

    cout << "Memcpying" << endl;
    gpuErrchk(cudaMemcpy(d_data, data, vox * 4 * N_nodes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c2ws, c2ws, N_nodes * 16 * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cif, cif, N_nodes * 3 * sizeof(float), cudaMemcpyHostToDevice));

    cout << "Initializing on device" << endl;
    init_mpis<<<1,N_nodes>>>(*this, d_mpis);
    gpuErrchk(cudaDeviceSynchronize());

    cout << "Done" << endl;

}
    
    
void MPIMeta::set_blend_weights(const float* pose_arr) {
    glm::mat4 pose = glm::make_mat4(pose_arr);

    vector<pair<float, int>> dists = pose_dists(vec3(column(pose, 3)), c2ws, N_nodes);
    cout << "Using: ";
    for (int j = 0; j < N_blend; ++j) {
        cout << dists[j].second << ", ";
        mpi_list[j] = d_mpis + (int)dists[j].second;
        weights[j] = exp(-dists[j].first);
    }
    cout << endl;

    gpuErrchk(cudaMemcpy(d_mpi_list, mpi_list, N_blend * sizeof(MPI*), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_weights, weights, N_blend * sizeof(float), cudaMemcpyHostToDevice));
}

    
void MPIMeta::render_pose(const float* pose_arr, uint8_t* d_out_arr, 
                 int imgH, int imgW, float imgF, int N_blend_) {
    glm::mat4 pose = glm::make_mat4(pose_arr);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(imgH/threadsPerBlock.x+1, 
              imgW/threadsPerBlock.y+1); 

    if (N_blend != N_blend_) {
        N_blend = N_blend_;

        mpi_list = (MPI**)malloc(N_blend * sizeof(MPI*));
        weights = (float*)malloc(N_blend * sizeof(float));
        gpuErrchk(cudaMalloc(&d_mpi_list, sizeof(MPI*) * N_blend));
        gpuErrchk(cudaMalloc(&d_weights, sizeof(float) * N_blend));
    }

    set_blend_weights(pose_arr);

    blend_mpis<<<numBlocks, threadsPerBlock>>>(
           pose, imgH, imgW, imgF, 
            d_mpi_list, d_weights, N_blend,
            d_out_arr);
}

    
void MPIMeta::render_pose_lf(const float* pose_arr, uint8_t* d_out_arr, 
                 int imgH, int imgW, float imgF, int N_blend_) {
    glm::mat4 pose = glm::make_mat4(pose_arr);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(imgH/threadsPerBlock.x+1, 
              imgW/threadsPerBlock.y+1); 

    trace_mpis<<<numBlocks, threadsPerBlock>>>(
           pose, imgH, imgW, imgF, 
            d_mpis, N_nodes,
            d_out_arr);
}



__global__
void init_mpis(MPIMeta d_meta, MPI* d_mpis)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= d_meta.N_nodes)
        return;
    
    init_gpu(d_meta, i);
}
