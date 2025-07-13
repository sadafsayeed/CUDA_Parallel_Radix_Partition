// Sadaf Sayeed Chowdhury
// U29205993

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
void dataGenerator(int* data, int count, int first, int step)
{
	assert(data != NULL);

	for(int i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

//define the histogram kernel here
__global__ void histogram(int* keys, int* histogram, int count, int bits, int partitions)
{
    // Making a shared memory array to store the temporary histogram counts
    extern __shared__ int shared_hist[];

    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Initializing the shared memory array to 0
    for(int i=threadIdx.x; i<partitions; i+=blockDim.x){
        shared_hist[i] = 0;
    }
    
    __syncthreads(); 

    if (k < count) {
        // This gets the histogram bin for the current key and atomically increments it in the right bin of the shared memory
        int h = bfe(keys[k], 0, bits);
        atomicAdd(&shared_hist[h], 1);
        
    }
    __syncthreads(); 

    // This writes the shared memory values in the global memory
    for (int i = threadIdx.x; i < partitions; i += blockDim.x) {
        atomicAdd(&histogram[i], shared_hist[i]);
    }
}

//define the prefix scan kernel here
__global__ void prefixScan(int* histogram, int* sum, int partitions)
{   
    // Making shared memory for the scan operation
    extern __shared__ int shared_mem[]; 

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize the shared memory with data from the histogram; the first element of the shared memory is initialized with 0 for the calculation
    if (i < partitions) {
        shared_mem[threadIdx.x] = (threadIdx.x > 0) ? histogram[i-1]: 0; 
    }

    __syncthreads();

    // Performing the prefix sum with step sizes doubling every iteration
    for (int j = 1; j < partitions; j *= 2) {
        int sum = 0;
        if (threadIdx.x >= j) {
            sum = shared_mem[threadIdx.x - j];
        }
        shared_mem[threadIdx.x] += sum;  
    
    }
    // Writing the prefix sum from the shared memory to the global memory
    if (i < partitions) {
        sum[i] = shared_mem[threadIdx.x]; 
    }
}


//define the reorder kernel here
__global__ void Reorder(int* keys, int* sum, int* output, int rSize, int nbits)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Reordering the array of keys using the prefix sums
    if (k < rSize) {
        int h = bfe(keys[k], 0, nbits);        
        int offset = atomicAdd(&sum[h], 1);
        output[offset] = keys[k];
    }
}

int main(int argc, char const *argv[]) {
    int rSize = atoi(argv[1]); 
    int partitions = atoi(argv[2]);
    int nbits = ceil(log2(partitions));
    
    // Declaring host and device pointers
    int* r_h;
    int *r_d, *histogram_d, *sum_d, *output_d;

    // Allocating host memory for histogram_h and sum_h
    int *histogram_h = (int*)malloc(partitions * sizeof(int));
    int *sum_h = (int*)malloc(partitions * sizeof(int));

    cudaMallocHost((void**)&r_h, sizeof(int) * rSize); // use pinned memory 
    
    dataGenerator(r_h, rSize, 0, 1);
    
    // Allocating memory for r_d, histogram_d, sum_d, and output_d
    cudaMalloc((void**)&r_d, sizeof(int) * rSize);
    cudaMalloc((void**)&histogram_d, sizeof(int) * partitions);
    cudaMalloc((void**)&sum_d, sizeof(int) * partitions);
    cudaMalloc((void**)&output_d, sizeof(int) * rSize);

    // Copying the input keys from host to device
    cudaMemcpy(r_d, r_h, sizeof(int) * rSize, cudaMemcpyHostToDevice);
    // Initializing histogram_d with 0s
    cudaMemset(histogram_d, 0, sizeof(int) * partitions);

    // Creating CUDA events for measuring the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Histogram kernel
    int numThreadsPerBlock = partitions;
    int numBlocks = (rSize + numThreadsPerBlock - 1) / numThreadsPerBlock;
    cudaEventRecord(start);
    histogram<<<numBlocks, numThreadsPerBlock, partitions * sizeof(int)>>>(r_d, histogram_d, rSize, nbits, partitions);
    cudaMemcpy(histogram_h, histogram_d, sizeof(int) * partitions, cudaMemcpyDeviceToHost);

    // Prefix scan kernel
    numBlocks = (partitions + numThreadsPerBlock - 1) / (numThreadsPerBlock);
    prefixScan<<<numBlocks, numThreadsPerBlock, sizeof(int) * partitions>>>(histogram_d, sum_d, partitions);
    cudaMemcpy(sum_h, sum_d, sizeof(int) * partitions, cudaMemcpyDeviceToHost);

    // Reorder kernel 
    numBlocks = (rSize + numThreadsPerBlock - 1) / (numThreadsPerBlock);
    Reorder<<<numBlocks, numThreadsPerBlock>>>(r_d, sum_d, output_d, rSize, nbits);
    cudaMemcpy(r_h, output_d, sizeof(int) * rSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    // Displays the partition information
    for (int i = 0; i < partitions; ++i) {
        int offset = (i == 0) ? 0 : sum_h[i];
        int keys = histogram_h[i];
        printf("partition %d: offset %d, number of keys %d\n", i, offset, keys);
    }

    // Displays the first 10 sorted output after the Reorder kernel runs
    printf("First 10 sorted output after reorder: \n");
    for(int i=0; i<10 && i<rSize; ++i){
        printf("%d\n", r_h[i]);
    }

    printf("****** Total Running Time of All Kernels = %.5f s ******\n", time/1000.0);

    /* Free memory */
    cudaFree(r_d);
    cudaFree(histogram_d);
    cudaFree(sum_d);
    cudaFree(output_d);
    cudaFreeHost(r_h);
    free(histogram_h);
    free(sum_h);

    return 0;
}
