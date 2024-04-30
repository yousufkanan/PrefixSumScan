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
#define CHECK(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
      if (abort) exit(code);
   }
}

template <typename T>
__global__ void prefixOperationKernel(T *data, long long n, bool isMultiplication) {
    extern __shared__ char shared_base[];
    T* shared = reinterpret_cast<T*>(shared_base);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) shared[tid] = data[idx];
    else shared[tid] = isMultiplication ? T(1) : T(0);

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            if (isMultiplication) {
                shared[index] *= shared[index - stride];
            } else {
                shared[index] += shared[index - stride];
            }
        }
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            if (isMultiplication) {
                shared[index + stride] *= shared[index];
            } else {
                shared[index + stride] += shared[index];
            }
        }
    }
    __syncthreads();

    if (idx < n) data[idx] = shared[tid];
}





template <typename T>
T *initializeArray(long long size, unsigned int seed) {
    T *arr = new T[size];
    srand(seed);
    for (long long i = 0; i < size; i++) {
        arr[i] = T(rand()) / T(RAND_MAX) * 2.0;
    }
    return arr;
}

void parseArguments(int argc, char **argv, long long *size, unsigned int *seed, std::string *dataType, bool *isMultiplication) {
    *dataType = "float";  // Default to float if not specified
    *isMultiplication = false; // Default to sum

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            *size = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            *dataType = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0) {
            *isMultiplication = true;
        } else {
            std::cerr << "Unknown option or missing argument: " << argv[i] << std::endl;
            exit(1);
        }
    }
}


int main(int argc, char **argv) {
    long long size = 4194304;  // Default size
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    std::string dataType = "float";  // Default data type
    bool isMultiplication = false; // Default to sum

    
    parseArguments(argc, argv, &size, &seed, &dataType, &isMultiplication);

    int numThreadsPerBlock = 256;
    int numBlocks = (size + numThreadsPerBlock - 1) / numThreadsPerBlock;

    cudaEvent_t start, stop;
    float milliseconds;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    if (dataType == "float") {
        float *arr_float = initializeArray<float>(size, seed);
        float *d_arr_float;
        CHECK(cudaMalloc((void **)&d_arr_float, size * sizeof(float)));
        CHECK(cudaMemcpy(d_arr_float, arr_float, size * sizeof(float), cudaMemcpyHostToDevice));

        CHECK(cudaEventRecord(start));
        prefixOperationKernel<float><<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(float)>>>(d_arr_float, size, isMultiplication);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        float sum_float;
        CHECK(cudaMemcpy(&sum_float, d_arr_float + size - 1, sizeof(float), cudaMemcpyDeviceToHost));

        // std::cout << "Float results:\nArray Size: " << size << "\nSum: " << sum_float << "\nExecution time: " << milliseconds << " ms\n";
        // prints only the size,and the execution time on one line
        std::cout << size << " " << milliseconds << std::endl;
        delete[] arr_float;
        CHECK(cudaFree(d_arr_float));
    } else if (dataType == "double") {
        double *arr_double = initializeArray<double>(size, seed);
        double *d_arr_double;
        CHECK(cudaMalloc((void **)&d_arr_double, size * sizeof(double)));
        CHECK(cudaMemcpy(d_arr_double, arr_double, size * sizeof(double), cudaMemcpyHostToDevice));

        CHECK(cudaEventRecord(start));
        prefixOperationKernel<double><<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(double)>>>(d_arr_double, size, isMultiplication);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        double sum_double;
        CHECK(cudaMemcpy(&sum_double, d_arr_double + size - 1, sizeof(double), cudaMemcpyDeviceToHost));

        //std::cout << "Double results:\nArray Size: " << size << "\nSum: " << sum_double << "\nExecution time: " << milliseconds << " ms\n";
        // prints only the size,and the execution time on one line
        std::cout << size << " " << milliseconds << std::endl;
        delete[] arr_double;
        CHECK(cudaFree(d_arr_double));
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}
