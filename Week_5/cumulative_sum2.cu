/* CUMULATIVE SUMS OF EACH BLOCK ARE STORED IN THE
** ARRAY b. WE THEN NEED THE CUMULATIVE SUM OF THESE
** ELEMENTS TO GO BACK AND FIX a. SO WE MUST CALL THE 
** FUNCTION AGAIN ON b, STORE ITS SUMS IN A NEW ARRAY,
** AND SO ON UNTIL WE ARE WORKING WITH ONLY ONE BLOCK. THEN 
** WE CAN GO BACK AND 'FIX' EACH ARRAY IN TURN. THIS
** CODE EXECUTES THAT PROCESS. */

#include<stdio.h>
#include<stdlib.h>
#include "timerc.h"

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define NL printf("\n");

#define MAX_ITERATIONS 10

__global__ void cumulative_sum2(int* a, int* b)
{
	int starting_address = 2*blockDim.x*blockIdx.x;
	int elem_per_block = 2*blockDim.x;
	
	int step = 1;
	for(step = 1; step <= blockDim.x; step *= 2)
	{
		int first = step;
		
		int idx = first + (threadIdx.x / step)*2*step + threadIdx.x%step;
		a[starting_address+idx] += a[starting_address+idx - (threadIdx.x%step) - 1];
		__syncthreads();
		
		if(threadIdx.x == 0)
		{
			b[blockIdx.x] = a[elem_per_block*(blockIdx.x+1)-1];
		}
	}
}

/* SAME AS ABOVE EXCEPT WE ASSUME THIS KERNEL HAS
** BEEN CALLED WITH ONLY ONE BLOCK SO NO NEED
** FOR b ARRAY. */
__global__ void cumulative_sum2_no_b(int* a, int size)
{
	if(threadIdx.x*2 < size )
	{
					int step = 1;
					for(step = 1; step <= blockDim.x; step *= 2)
					{
						int first = step;
						
						int idx = first + (threadIdx.x / step)*2*step + threadIdx.x%step;
						a[idx] += a[idx - (threadIdx.x%step) - 1];
						__syncthreads();		
					}
	}
}

/* PRINT THE CONTENTS OF INTEGER ARRAY STORED
** ON DEVICE FROM CPU. */
__global__ void print_array(int* a, int size)
{
	for( int i = 0; i < size; i++)
	{
		printf("%d ", a[i]);
	}
	printf("\n");
}

__global__ void finish_sum(int* a, int* b)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(blockIdx.x>0)
	{
		a[id*2] += b[blockIdx.x-1];
		a[id*2+1] += b[blockIdx.x-1];
	}
}
		
void CPU_cumulative_sum(int* a, int size)
{
	for(int i = 1; i < size; i++)
	{
		a[i] += a[i-1];
	}
}


int main(void)
{
	int n = 1024*1024*3;
	int threads_per_block = 256;
	int size_per_block = threads_per_block*2;
	int num_blocks = (n + size_per_block-1) / size_per_block;
	
	float cpu_time;
	float gpu_time;
	
	float gpu_setup;
	float gpu_setdown;
	
printf("%d elements, %d elements per block, %d blocks on first iteration\n", n, size_per_block, num_blocks);
	
	int* host_input = (int*) malloc(sizeof(int)*n);
	for(int i = 0; i < n; i++)
	{
		host_input[i] = 1;
	}

	gstart();	
	int* dev_ptr1;
	cudaMalloc(&dev_ptr1, n*sizeof(int));
	cudaMemcpy(dev_ptr1, host_input, n*sizeof(int), H2D);
	
	
	int* subtotal_arrays[MAX_ITERATIONS];
	subtotal_arrays[0] = dev_ptr1;

	int lastarray_size;	
	int i = 1;
	gend(&gpu_setup);
	gstart();
	while(num_blocks > 1)
	{
//printf("Num blocks = %d\n", num_blocks);
//print_array<<<1,1>>>(subtotal_arrays[i-1], num_blocks*size_per_block);
		
		int* b;
		cudaMalloc(&b, num_blocks*sizeof(int));
		subtotal_arrays[i] = b;	
		cumulative_sum2<<<num_blocks, threads_per_block>>>(subtotal_arrays[i-1], subtotal_arrays[i]);
//print_array<<<1,1>>>(subtotal_arrays[i], num_blocks);
	
		i++;
		lastarray_size = num_blocks;
		num_blocks = (num_blocks+size_per_block-1)/size_per_block;
	}

	cumulative_sum2_no_b<<<1, threads_per_block>>>(subtotal_arrays[i-1], lastarray_size);
//print_array<<<1, 1>>>(subtotal_arrays[i-1], lastarray_size);

	num_blocks = lastarray_size;	
	for(int rev = i-1; rev > 0; rev--)
	{
		finish_sum<<<num_blocks, threads_per_block>>>(subtotal_arrays[rev-1], subtotal_arrays[rev]);
//print_array<<<1,1>>>(subtotal_arrays[rev-1], num_blocks*size_per_block);

		cudaFree(subtotal_arrays[rev]);
		num_blocks *= size_per_block;
	}
	gend(&gpu_time);

	int* host_output = (int*) malloc(n*sizeof(int));
	gstart();
	cudaMemcpy(host_output, dev_ptr1, n*sizeof(int), D2H);
	gend(&gpu_setdown);

	cstart();
	CPU_cumulative_sum(host_input, n);
	cend(&cpu_time);
		
	for(i = 0; i < n; i++)
	{
		//printf("%d ", host_output[i]);
		if(host_input[i] != host_output[i])
		{
			
			printf("GPU error! @ element %d\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("GPU setup time: %f, GPU compute time: %f, GPU setdown time: %f, CPU time: %f, ", gpu_setup, gpu_time, cpu_time, gpu_setdown); 	
	NL	
}

	
