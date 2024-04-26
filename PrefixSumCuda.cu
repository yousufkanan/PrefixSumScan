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
__global__ void prefixSumKernel(int *data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) temp[tid] = data[idx];
    else temp[tid] = 0;

    __syncthreads();

    // Upsweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x)
            temp[index] += temp[index - stride];
        __syncthreads();
    }

    // Downsweep (scan) phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
        __syncthreads();
    }

    if (idx < n) data[idx] = temp[tid];
}


__global__ void prefixMultKernel(int *data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) temp[tid] = data[idx];
    else temp[tid] = 1;

    __syncthreads();

    // Upsweep (reduce) phase for multiplication
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x)
            temp[index] *= temp[index - stride];
        __syncthreads();
    }

    // Downsweep (scan) phase for multiplication
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] *= temp[index];
        }
        __syncthreads();
    }

    if (idx < n) data[idx] = temp[tid];
}

double *initializeArray(int size, unsigned int seed) {
    double *arr = (double *)malloc(size * sizeof(double));
    if (!arr) {
        std::cerr << "Memory allocation failed\n";
        exit(1);
    }
    srand(seed);
    for (int i = 0; i < size; i++) {
        arr[i] = (double)rand() / (double)RAND_MAX * 2;
    }
    return arr;
}

// argument parsing function: -s <size> -r <seed> -f <function> -o <outputFile> -i <inputFile> 
void parseArguments(int argc, char **argv, long long *size, unsigned int *seed, bool *doSum, char **outputFile, char **inputFile) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            *size = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            *doSum = (strcmp(argv[i + 1], "sum") == 0);
            i++;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            *outputFile = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            *inputFile = argv[++i];
        } else {
            std::cerr << "Unknown option or missing argument: " << argv[i] << std::endl;
            exit(1);
        }
    }
}


int main(int argc, char **argv) {
    long long size = 4194304;  // default size
    unsigned int seed = (unsigned int)time(NULL);
    bool doSum = true;
    char *outputFile = NULL;
    char *inputFile = NULL;

    parseArguments(argc, argv, &size, &seed, &doSum, &outputFile, &inputFile);

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

    if (doSum) {
        prefixSumKernel<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(int)>>>(d_arr, size);
    } else {
        prefixMultKernel<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(int)>>>(d_arr, size);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (outputFile) {
        std::ofstream out(outputFile);
        for (int i = 0; i < size; i++) {
            out << arr[i] << std::endl;
        }
        out.close();
    }

    free(arr);

    return 0;
}

