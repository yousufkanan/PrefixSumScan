#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

// nvcc -arch=sm_86 -O3 --compiler-options -march=native PrefixSumCudaBuiltIn.cu -o PrefixSumCudaBuiltIn


#define CHECK(ans) { checkCudaError((ans), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

template <typename T>
void initializeArray(thrust::host_vector<T> &h_data, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < h_data.size(); i++) {
        h_data[i] = T(rand()) / T(RAND_MAX) * 2.0;  // Initialize with values between 0 and 2
    }
}

void parseArguments(int argc, char **argv, int *size, unsigned int *seed, std::string *dataType) {
    *dataType = "float";  // Default to float if not specified

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            *size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            *dataType = argv[++i];
        } else {
            std::cerr << "Unknown option or missing argument: " << argv[i] << std::endl;
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    int size = 4194304;  // Default size
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    std::string dataType = "float";  // Default data type

    parseArguments(argc, argv, &size, &seed, &dataType);

    cudaEvent_t start, stop;
    float milliseconds;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    if (dataType == "float") {
        thrust::host_vector<float> h_data(size);
        initializeArray<float>(h_data, seed);

        thrust::device_vector<float> d_data = h_data;
        thrust::device_vector<float> d_output(size);

        CHECK(cudaEventRecord(start));
        thrust::exclusive_scan(d_data.begin(), d_data.end(), d_output.begin());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        std::cout << size << " " << milliseconds << " ms\n";
    } else if (dataType == "double") {
        thrust::host_vector<double> h_data(size);
        initializeArray<double>(h_data, seed);

        thrust::device_vector<double> d_data = h_data;
        thrust::device_vector<double> d_output(size);

        CHECK(cudaEventRecord(start));
        thrust::exclusive_scan(d_data.begin(), d_data.end(), d_output.begin());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        std::cout << size << " " << milliseconds << " ms\n";
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}
