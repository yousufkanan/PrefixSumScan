/*
 * This program calculates the prefix sum or prefix multiplication of an array of random numbers.
 * This approach is serial and uses a single thread.
 * The program takes the following command line arguments:
 * -s <size> : size of the array (default: 4194304)
 * -r <seed> : seed for the random number generator (default: current time)
 * -f <function> : function to execute (prefixSum or prefixMult, default: prefixSum)
 * -o <outputFile> : output file to write the results (default: /dev/null)
 * -i <inputFile> : input file to read the array from (default: /dev/null)
 * -t <threads> : number of threads to use (default: 8)
 * Example: ./PrefixSumSerial -s 100 -r 1 -f prefixSum -o output.txt -i input.txt
 * Instructor: Dr. Jeffery Bush
 * compile with: gcc-13 -Wall -O3 -fopenmp -march=native PrefixSumSharedAlgorithm2.c -o PrefixSumSharedAlgorithm2 -lm
 * Exameple: ./PrefixSumSharedAlgorithm2 -s 100 -r 1 -f prefixSum -o output.txt -i input.txt -t 8
 * Authors: Yousuf Kanan and Derek Allmon
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void prefixSumShared(double *arr, int size, int numThreads) {
    if (size < 2) return;  // No operation needed for size < 2

    double *new_arr = (double *)malloc(size * sizeof(double));
    if (new_arr == NULL) return; // Allocation failed

    #pragma omp parallel num_threads(numThreads)
    {
        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
        int start_index = thread_id * (size / thread_count);
        int end_index = (thread_id + 1) * (size / thread_count);

        // Handle remainder
        if (thread_id == thread_count - 1) {
            end_index = size;
        }

        // Copy to new_arr
        for (int j = start_index; j < end_index; j++) {
            new_arr[j] = arr[j];
        }

        #pragma omp barrier

        // Compute prefix sums in each thread's partition
        for (int step = 1; step < thread_count; step <<= 1) {
            if (thread_id % (2 * step) == 0) {
                int add_index = start_index + step - 1;
                if (add_index < size) {
                    new_arr[start_index + 2 * step - 1] += new_arr[add_index];
                }
            }
            #pragma omp barrier
        }
    }

    // Copy results back to original array
    for (int i = 0; i < size; i++) {
        arr[i] = new_arr[i];
    }

    free(new_arr); // Free the allocated memory
}

void prefixMultShared(double *arr, int size, int numThreads) {
    int max_depth = (int)ceil(log2(size));  // Calculate once and cast to integer

    #pragma omp parallel for num_threads(numThreads)
    for (int d = 0; d < max_depth; d++) {
        int step = 1 << (d + 1);
        for (int i = step - 1; i < size; i += step) {
            arr[i] *= arr[i - (step >> 1)];
        }
    }

    // Reverse phase of the Blelloch scan
    arr[size - 1] = 1; // Set last element to one before reverse phase
    for (int d = max_depth - 1; d >= 0; d--) {
        int step = 1 << (d + 1);
        #pragma omp parallel for num_threads(numThreads)
        for (int i = step - 1; i < size; i += step) {
            double temp = arr[i - (step >> 1)];
            arr[i - (step >> 1)] = arr[i];
            arr[i] *= temp;
        }
    }
}

// -s <size> : size of the array (default: 4194304) -r <seed> : seed for the random number generator (default: current time) -f <function> : function to execute (prefixSum or prefixMult, default: prefixSum) -o <outputFile> : output file to write the results (default: /dev/null) -i <inputFile> : input file to read the array from (default: /dev/null) -t <threads> : number of threads to use (default: 8)
void parseArguments(int argc, char **argv, long long *size, unsigned int *seed, void (**function)(double *, int, int), char **outputFile, char **inputFile, int *threads)
{
     for (int i = 1; i < argc; i++){
          if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
               *size = atoll(argv[++i]);
          }
          if (strcmp(argv[i], "-r") == 0 && i + 1 < argc){
               *seed = (unsigned int)atoi(argv[++i]);
          }
          if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
               if (strcmp(argv[i + 1], "prefixMult") == 0) {
                    *function = prefixMultShared;
               }
               else if (strcmp(argv[i + 1], "prefixSum") == 0) {
                    *function = prefixSumShared;
               }
               i++;
          }
          if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
               *outputFile = argv[++i];
          }
          if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
               *inputFile = argv[++i];
          }
          if (strcmp(argv[i], "-t") == 0 && i + 1 < argc){
               *threads = atoi(argv[++i]);
          }
     }
}

double *initializeArray(int size, unsigned int seed) {
    double *arr = (double *)malloc(size * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    unsigned int rand_state = seed;

    srand(seed);
    for (int i = 0; i < size; i++) {
         arr[i] = (double)rand_r(&rand_state) / (double)RAND_MAX * 2;
    }
    return arr;
}

void writeOutputFile(double *arr, long long size, char *outputFile)
{
     if (strcmp(outputFile, "/dev/null") == 0)
     {
          return;
     }
     FILE *file = fopen(outputFile, "w");
     if (!file)
     {
          fprintf(stderr, "Failed to open output file\n");
          exit(1);
     }
     for (int i = 0; i < size; i++)
     {
          fprintf(file, "%lf\n", arr[i]);
     }
     fclose(file);
}

int main(int argc, char **argv)
{
     long long size = 4194304;  
    unsigned int seed = (unsigned int)time(NULL);
    void (*function)(double *, int, int) = prefixSumShared;
    char *outputFile = "/dev/null";
    char *inputFile = "/dev/null";
    int threads = 8;
    parseArguments(argc, argv, &size, &seed, &function, &outputFile, &inputFile, &threads);
    
    omp_set_num_threads(threads);
    double *arr = initializeArray(size, seed);
     writeOutputFile(arr, size, inputFile);
    double start = omp_get_wtime();
    function(arr, size, threads);
    double end = omp_get_wtime();
    printf("%llu, %f ms\n", size, (end - start) * 1000.0);

    writeOutputFile(arr, size, outputFile);
    free(arr);
    return 0;
}



