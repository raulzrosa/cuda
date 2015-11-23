#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]){

	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("  Device:                                        \"%s\"\n", deviceProp.name);
    printf("  Compute Capability:                            %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Multiprocessors count:                         %d\n", deviceProp.multiProcessorCount);
    
    
	printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Warp size:                                     %d\n", deviceProp.warpSize);
    printf("  Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z):  (%d, %d, %d)\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    
    printf("  Total global mem:                              %0.f MBytes\n", deviceProp.totalGlobalMem/1048576.0f);
    
    return 0;
}