#include <stdio.h> 
/* Prints some properties of the GPU. */

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf(" Total Global Memory: %d\n", prop.totalGlobalMem);
		printf(" Shared Memory Per Block: %d\n", prop.sharedMemPerBlock);
		printf(" Regs Per Block: %d\n", prop.regsPerBlock);
		printf(" Warp Size: %d\n", prop.warpSize);
		printf(" maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
		printf(" Max Block Dimensons: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf(" Max Grid Dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],prop.maxGridSize[1], prop.maxGridSize[1], prop.maxGridSize[2]);
		
		
 } 
}	
