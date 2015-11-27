#include <stdio.h>

#define N 16


void print (int *a){
	for (int i = 0; i < N; i++){
		printf("%d ", a[i]);
	}
	printf("\n\n");
}

__global__ void shuffle (int *a, int *b, int srcLane){
	int value = a[threadIdx.x];
    value = __shfl(value, srcLane, N);
    b[threadIdx.x] = value;
}


__global__ void shuffle_half (int *a, int *b, int srcLane){
	int value = a[threadIdx.x];
    value = __shfl(value, srcLane, N/2);
    b[threadIdx.x] = value;
}


__global__ void shuffle_up (int *a, int *b, int delta){
	int value = a[threadIdx.x];
    value = __shfl_up(value, delta, N);
    b[threadIdx.x] = value;
}

__global__ void shuffle_down (int *a, int *b, int delta){
	int value = a[threadIdx.x];
    value = __shfl_down(value, delta, N);
    b[threadIdx.x] = value;
}

__global__ void shuffle_xor (int *a, int *b, int lanemask){
	int value = a[threadIdx.x];
    value = __shfl_xor(value, lanemask, N);
    b[threadIdx.x] = value;
}

int main (int argc, char *argv[]){

	size_t size = N * sizeof(int);

	int *h_a, *h_b;
	int *d_a, *d_b;

	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);

	printf("Vetor inicial: \n");
	for (int i = 0; i < N; i++){
		h_a[i] = i;
		printf("%d ", h_a[i]);
	}
	printf("\n\n");

	cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);

	shuffle<<<1,N>>>(d_a, d_b,2);
	cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);

	printf("__shfl: \n");
	print(h_b);



	shuffle_half<<<1,N>>>(d_a, d_b,2);
	cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);

	printf("__shfl halfwarp: \n");
	print(h_b);



	shuffle_up<<<1,N>>>(d_a, d_b,2);
	cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);

	printf("__shfl up: \n");
	print(h_b);



	shuffle_xor<<<1,N>>>(d_a, d_b,1);
	cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);

	printf("__shfl xor 1: \n");
	print(h_b);



	shuffle_xor<<<1,N>>>(d_a, d_b,2);
	cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);

	printf("__shfl xor 2: \n");
	print(h_b);


	shuffle_xor<<<1,N>>>(d_a, d_b,4);
	cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost);

	printf("__shfl xor 4: \n");
	print(h_b);

	free(h_a);
	free(h_b);

	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}