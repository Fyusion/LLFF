cuda_renderer: main.cu mpi_cuda.cu
	nvcc -ccbin g++ -std=c++11 -m64 -I. \
	main.cu mpi_cuda.cu -o cuda_renderer 

clean: 
	rm cuda_renderer
    