#include <stdio.h>
#include <iostream>
#include <math.h>

// includes, project
#include <cuda.h>
// #include <cuda_runtime.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>

#define GLM_FORCE_CUDA
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp> 

#include "mpi_cuda.h"

#include <iostream>
#include <iomanip>

#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>

#include <chrono> 
using namespace std::chrono; 

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



void gpu2ffmpeg(const char* filename, uint8_t* d_out_arr_vid,
                  int imageWidth, int imageHeight, int N_frames, int crf) {
    
    FILE *pPipe;
    long lSize;
    int imgcols = imageWidth, imgrows = imageHeight, elemSize = 3;
    
    stringstream sstm;
    sstm << "/usr/bin/ffmpeg -y -framerate 30 -f rawvideo -vcodec rawvideo -s " << imgcols << "x" << imgrows  
        <<" -pix_fmt rgb24 -i - -pix_fmt yuv420p -r 30 -crf " << crf << " -c:v libx264 -shortest " << filename;

    cout << "ffmpeg, calling:" << endl;
    cout << sstm.str() << endl;
    // open a pipe to FFmpeg
    if ( !(pPipe = popen(sstm.str().c_str(), "w")) ) {
        cout << "popen error" << endl;
        exit(1);
    }

    // write to pipe
    lSize = imgrows * imgcols * elemSize;
    uint8_t* out_arr = new uint8_t[lSize];
    
    for (int i = 0; i < N_frames; ++i) {
        cudaMemcpy(out_arr, d_out_arr_vid + lSize * i, lSize, cudaMemcpyDeviceToHost);
        gpuErrchk(cudaDeviceSynchronize());
        fwrite(out_arr, 1, lSize, pPipe);        
    }
    fflush(pPipe);
    fclose(pPipe);
    
    delete [] out_arr;
}


void load_render_poses(const char* posefile, vector<mat4>& poses, int& w, int& h, float& f) {
    float buffer[16];
    
    std::ifstream file(posefile);
    float wf, hf;
    int N_poses=0;
    string line;
    
    int l = 0;
    while (getline(file, line)) {
        // cout << l++ << ": " << line << endl;
        auto sline = istringstream(line);
        if (N_poses==0){
            sline >> N_poses;
            cout << "Loading render poses: " << posefile << ", " << N_poses << endl;
        } else {
            
            for (int j = 0; j < 16; ++j) {
                if (j%4==3) {
                    buffer[j] = j < 15 ? 0.f : 1.f;
                } else {
                    sline >> buffer[j];
                    // cout << buffer[j] << ", ";
                }
            }
            // cout << endl;
            sline >> hf >> wf >> f;
            mat4 p;
            memcpy(value_ptr(p), buffer, 16 * sizeof(float));
            poses.push_back(p);
            // cout << l << ": " << hf << " " << wf << " " << f << endl;
        }
        l++;
    }
    w = (int)wf;
    h = (int)hf;
    cout << w << " x " << h << ", " << f << endl;
    
}

void render_poses(MPIMeta meta, const char* mpidir, const char* posefile, const char* videofile, 
                  int height=-1, float scale=1., int crf=18) {
    cout << "Begin render poses" << endl;
    
    int imageWidth, imageHeight;
    float focal;
    vector<mat4> poses;
    
    load_render_poses(posefile, poses, imageWidth, imageHeight, focal);
    
    if (height <= 0) {
        height = meta.height;
    }
    
    float factor = height / (float)imageHeight;
    imageWidth  = int(imageWidth  * factor * scale * .5) * 2;
    imageHeight = int(imageHeight * factor * scale * .5) * 2;
    focal *= factor;
    
    int N_blend = 5;
    
    int N_frames = poses.size();
    
    // GPU buffer for frames
    uint8_t* d_out_arr_vid;
    unsigned long N_bytes = 3 * imageWidth * imageHeight;
    cout << "Alloc video buffer on GPU " << N_frames * N_bytes / (1<<20) << " MB" << endl;
    gpuErrchk(cudaMalloc(&d_out_arr_vid, N_frames * N_bytes));
    
    auto start = high_resolution_clock::now(); 
    
    cout << "Render frames" << endl;
    for (int i = 0; i < N_frames; ++i) {
        
        glm::mat4 pose = poses[i];
        
        meta.render_pose(glm::value_ptr(pose), d_out_arr_vid + N_bytes * i, 
                     imageHeight, imageWidth, focal, N_blend);
        
    }
    gpuErrchk(cudaDeviceSynchronize());
    
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<milliseconds>(stop - start); 

    cout << "End render vid " << duration.count() << " ms" << endl;
    
    
    std::cout << "Saving " << videofile << std::endl;
    gpu2ffmpeg(videofile, d_out_arr_vid,
                  imageWidth, imageHeight, N_frames, crf);
    
    cout << "finished" << endl;
    
}



void run(const char* mpidir, const char* posefile, const char* videofile,
                    int height, float scale, int crf) {
    
    MPIMeta meta;
    
    std::cout << "Loading " << mpidir << std::endl;
    meta.load_all_mpis(mpidir);
    
    meta.mpis2gpu();
    
    render_poses(meta, mpidir, posefile, videofile, height, scale, crf);
    
}



int main(int argc, const char* argv[]) {
    
    if (argc < 7) {
        std::cout << "Usage: demo <mpidir> <posefile> <videofile> <height> <scale> <crf>" << std::endl;
        return -1;
    }
    run(argv[1], argv[2], argv[3], stoi(argv[4]), stof(argv[5]), stoi(argv[6]));
    
    cout << "Done!" << endl;
}