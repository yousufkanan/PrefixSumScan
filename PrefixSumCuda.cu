/*
     * This program calculates the prefix sum or prefix multiplication of an array of random numbers.
     * This approach uses cuda.
     * The program takes the following command line arguments:
     * -s <size> : size of the array (default: 4194304)
     * -r <seed> : seed for the random number generator (default: current time)
     * -f <function> : function to execute (prefixSum or prefixMult, default: prefixSum)
     * -o <outputFile> : output file to write the results (default: /dev/null)
     * -i <inputFile> : input file to read the array from (default: /dev/null)
     * Example: ./PrefixSumSerial -s 100 -r 1 -f prefixSum -o output.txt -i input.txt
     * Instructor: Dr. Jeffery Bush
     * Authors: Yousuf Kanan and Derek Allmon
     * 
     * To compile:
 *     nvcc -arch=sm_86 -O3 --compiler-options -march=native PrefixSumCuda.cu -o PrefixSumCuda
 * 
 * 
 * 
 *     To run: srun --pty --wait=0 --export=ALL -p gpu-shared --exclusive /bin/bash
 * 
 *      ./PrefixSumCuda -s 1000000 -r 1 -f sum -o output.txt
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Cuda kernal for up-sweep phase of the prefix sum
// each thread calculates the sum of the elements in its block stored in shared memory
// reduction: performs a binary tree reduction on the shared memory array so that the last element of the array contains the sum of all elements in the block
// Cuda kernel for computing the prefix sum using the up-sweep and down-sweep phases
template <typename T>
__global__ void prefixSumKernel(T *data, int n) {
    extern __shared__ T shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n) shared[tid] = data[idx];
    else shared[tid] = 0;

    __syncthreads();

    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x)
            shared[index] += shared[index - stride];
        __syncthreads();
    }

    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            shared[index + stride] += shared[index];
        }
    }
    __syncthreads();

    // Write results back to global memory
    if (idx < n) data[idx] = shared[tid];
}
double *initializeArray(int size, unsigned int seed) {
    double *arr = new double[size];
    srand(seed);
    for (int i = 0; i < size; i++) {
        arr[i] = static_cast<double>(rand()) / RAND_MAX * 2;
    }
    return arr;
}

// argument parsing function: -s <size> -r <seed> -f <function> -o <outputFile> -i <inputFile> 
void parseArguments(int argc, char **argv, long long *size, unsigned int *seed, char **outputFile) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            *size = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            *outputFile = argv[++i];
        } else {
            std::cerr << "Unknown option or missing argument: " << argv[i] << std::endl;
            exit(1);
        }
    }
}



int main(int argc, char **argv) {
     long long size = 4194304;  // default size
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    char *outputFile = nullptr;

    parseArguments(argc, argv, &size, &seed, &outputFile);

    double *arr = initializeArray(size, seed);
    int *d_arr;
    cudaMalloc((void **)&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    int numThreadsPerBlock = 256;
    int numBlocks = (size + numThreadsPerBlock - 1) / numThreadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    prefixSumKernel<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(int)>>>(d_arr, size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

    // Allocate memory to store the total sum on the host
    int totalSum;
    
}

