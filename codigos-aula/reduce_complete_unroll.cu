#include <stdio.h>	

__global__ void reduce (int *a, int *b, int N){
	extern __shared__ int s_a[];

	int tid = threadIdx.x;
	int id = threadIdx.x + (blockDim.x*2) * blockIdx.x;

	s_a[tid] = 0;
	if (id >= N) return;

	if (id+blockDim.x >= N)
		s_a[tid] = a[id];
	else
		s_a[tid] = a[id] + a[id+blockDim.x];

	__syncthreads();
	
	
	if (tid < 512){
		s_a[tid] += s_a[tid + 512];
		__syncthreads();
	}
	if (tid < 256){
		s_a[tid] += s_a[tid + 256];
		__syncthreads();
	}
	if (tid < 128){
		s_a[tid] += s_a[tid + 128];
		__syncthreads();
	}
	if (tid < 64){
		s_a[tid] += s_a[tid + 64];
		__syncthreads();
	}
	if (tid < 32){
		volatile int* s_b = s_a;
		s_b[tid] += s_b[tid + 32];
		s_b[tid] += s_b[tid + 16];
		s_b[tid] += s_b[tid + 8];
		s_b[tid] += s_b[tid + 4];
		s_b[tid] += s_b[tid + 2];
		s_b[tid] += s_b[tid + 1];
	}
	

	if (tid == 0)
		b[blockIdx.x] = s_a[0];
	
}


int main (int argc, char *argv[]){
	int N = atoi(argv[1]);

	int num_thread = 1024;
	int num_block = ceil((float)N/1024);
	num_block = ceil((float)num_block/2);

	size_t size = N * sizeof(int);
	size_t size_result = num_block * sizeof(int);

	int *h_a, *h_b;

	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size_result);

	int *d_a, *d_b;
	cudaMalloc(&d_a,size);
	cudaMalloc(&d_b,size_result);

	for (int i = 0; i < N; i++)
		h_a[i] = 1;

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

	printf("Blocks: %d Threads: %d \n", num_block, num_thread);
	reduce<<<num_block, num_thread, num_thread * sizeof(int)>>>(d_a, d_b, N);

	cudaMemcpy(h_b, d_b, size_result, cudaMemcpyDeviceToHost);

	int result = 0;
	for (int i = 0; i < num_block; i++){
		result += h_b[i];
		printf("%d ", h_b[i]);
	}

	printf("\nResultado: %d\n", result);

	return 0;
}