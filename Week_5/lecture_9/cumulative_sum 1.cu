#include <stdio.h>
#include "timerc.h"

__global__ void cumulative_sum(int *a, int *b) {
	int size_per_block = 2 * blockDim.x;
	int start_address = size_per_block * blockIdx.x;

	for(int s = 1; s <= size_per_block / 2; s *= 2) {
		if(threadIdx.x < blockDim.x / s) {
			a[start_address + 2 * s - 1 + threadIdx.x * s * 2] += a[start_address + s - 1 + threadIdx.x * s * 2];
		}
		__syncthreads();
	}
    
    int mult = 1;
    for(int s = size_per_block / 2; s >= 2; s /= 2) {
        if(threadIdx.x < 2 * mult - 1) {
            a[start_address + s - 1 + (s / 2) + threadIdx.x * s] += a[start_address + s - 1 + threadIdx.x * s];
        }
        __syncthreads();
        mult *= 2;
    }
    
    if(threadIdx.x == 0) {
        b[blockIdx.x] = a[start_address + 2*blockDim.x-1];
    }

}

__global__ void fix_sum(int *a, int *b, int size_per_block) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id > size_per_block/2 - 1) {
      a[id*2] += b[blockIdx.x - 1];
			a[id*2+1] += b[blockIdx.x - 1];
		}			
}

__global__ void sum_subtotals(int *b, int num_blocks) {
	for(int i = 1; i < num_blocks; i++)
	{
		b[i] += b[i-1];
	}
}

//printf("%d ", a[id]);
  

int main() {
    
    int SIZE = 128;
	int *device_input;
	int *device_output;
	int *host_input = (int *) malloc(SIZE * sizeof(int));
    int *host_output = (int *) malloc(SIZE * sizeof(int));
    int *host_output_b = (int *) malloc(SIZE * sizeof(int));
    int threads_per_block = 16;
    int size_per_block = 2 * threads_per_block;
    int num_blocks = (SIZE + size_per_block - 1) / size_per_block;
    
	float time;


	cudaMalloc(&device_input, SIZE * sizeof(int));
	cudaMalloc(&device_output, num_blocks * sizeof(int));

	for(int i = 0; i < SIZE; i++) {
		host_input[i] = 1;
	}

    cudaMemcpy(device_input, host_input, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	gstart();
	cumulative_sum<<<num_blocks, threads_per_block>>>(device_input, device_output);
	sum_subtotals<<<1,1>>>(device_output, num_blocks);
	fix_sum<<<num_blocks, threads_per_block>>>(device_input, device_output, size_per_block);

	cudaMemcpy(host_output, device_input, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_output_b, device_output, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
	gend(&time);

	printf("it took %f seconds\n", time);


	
	for(int i = 0; i < SIZE; i++) {
        printf("%d ", host_output[i]);
	/*	int correct = (i * (i + 1)) / 2;
		if(a_host[i] != correct) 
			printf("Error at pos: %d expected: %d actual: %d\n", i, correct, a_host[i]); */
	}
    printf("\n");
	
	for(int i = 0; i < num_blocks; i++)
	{
		printf("%d ", host_output_b[i]);
	}
    printf("\n");
    
    free(host_output);
    free(host_input);
    cudaFree(device_input);
    
    return 0;
}
