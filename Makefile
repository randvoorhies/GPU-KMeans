
all: GPUKMeans.o 
	g++ ${CPPFLAGS} -L/usr/X11R6/lib -lX11 -L/usr/local/cuda/lib -lcuda -lcudart GPUKMeans.o test-GPUKMeans.cpp -o main

GPUKMeans.o: GPUKMeans.cu GPUKMeans.h 
	nvcc -DTHRUST_DEBUG GPUKMeans.cu -m64 -c -o GPUKMeans.o

