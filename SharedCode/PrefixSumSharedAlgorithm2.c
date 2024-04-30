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
 * compile with: gcc-13 -Wall -O3 -fopenmp -march=native PrefixSumSharedAlgorithm1.c -o PrefixSumSharedAlgorithm1
 * Exameple: ./PrefixSumSharedAlgorithm1 -s 100 -r 1 -f prefixSum -o output.txt -i input.txt -t 8
 * Authors: Yousuf Kanan and Derek Allmon
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void prefixSumShared(double *arr, long long size, int numThreads)
{
        int *output = (int *)malloc(size * sizeof(int));
    if (output == NULL)
    {
        // Handle memory allocation failure if needed
        return;
    }
    int nthreads = 1;
    int thread_id = 0;
    nthreads = numThreads;
    thread_id = 8;
    int tbegin = size * thread_id / nthreads;
    int tend = size * (thread_id + 1) / nthreads;
    if (tbegin < tend)
    {
        output[tbegin] = 0;
        for (long long j = tbegin + 1; j < tend; j++)
        {
            output[j] = arr[j-1] + output[j - 1];
        }
    }
    if (nthreads == 1)
    {
        
        return;
    }
  
#pragma omp barrier

    if (thread_id == 0)
    {
        for (int i = 1; i < nthreads; i++)
        {
            int ibegin = size * (i - 1) / nthreads;
            int iend = size * i / nthreads;

            if (ibegin < iend)
            {
                output[iend] = arr[iend - 1] + output[ibegin];
            }
            if (i == nthreads - 1)
            {
                output[iend] += output[iend - 1];
            }
        }
#pragma omp barrier

#pragma omp simd
        for (int i = tbegin + 1; i < tend; i++)
        {
            output[i] += output[tbegin];
        }
    }
    
        for (long long i = 0; i < size; i++)
        {
            printf("%d, ", output[i]);
            arr[i] = output[i];
        }
        free(output);

}

// copy the output array to the arr array}

void prefixMultShared(double *arr, long long size, int numThreads)
{
    double *output = (double *)malloc(size * sizeof(double));
    if (output == NULL)
    {
        // Handle memory allocation failure if needed
        return;
    }

    int steps = (int)ceil(log2(size)); // Use ceil to handle non-power of two sizes

    for (long long i = 0; i < steps; i++)
    {
        int powerOfTwo = 1 << i; // Compute 2^i once for use in the loop

#pragma omp parallel for num_threads(numThreads) default(none) shared(arr, output, size, powerOfTwo)
        for (long long j = 0; j < size; j++)
        {
            if (j < powerOfTwo)
            {
                output[j] = arr[j];
            }
            else
            {
                output[j] = arr[j] * arr[j - powerOfTwo];
            }
        }

        // Swap pointers
        double *temp = arr;
        arr = output;
        output = temp;
    }

    // Ensure final output is in arr, if the total number of steps is odd, we need one last swap
    if (steps % 2 != 0)
    {
        memcpy(arr, output, size * sizeof(double));
    }
}

void parseArguments(int argc, char **argv, long long *size, unsigned int *seed, void (**function)(double *, long long, int), char **outputFile, char **inputFile, int *threads)

{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
        {
            *size = atoll(argv[++i]);
        }
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
        {
            *seed = (unsigned int)atoi(argv[++i]);
        }
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc)
        {
            if (strcmp(argv[i + 1], "prefixMult") == 0)
            {
                *function = prefixMultShared;
            }
            else if (strcmp(argv[i + 1], "prefixSum") == 0)
            {
                *function = prefixSumShared;
            }
            i++;
        }
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            *outputFile = argv[++i];
        }
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
        {
            *inputFile = argv[++i];
        }
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
        {
            *threads = atoi(argv[++i]);
        }
    }
}

double *initializeArray(long long size, unsigned int seed)
{
    double *arr = (double *)malloc(size * sizeof(double));
    if (!arr)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    srand(seed);
    for (int i = 0; i < size; i++)
    {
        arr[i] = (double)rand() / (double)RAND_MAX * 2;
        //    arr[i] = i;
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
    for (long long i = 0; i < size; i++)
    {
        fprintf(file, "%lf\n", arr[i]);
    }
    fclose(file);
}

int main(int argc, char **argv)
{
    long long size = 4194304;
    unsigned int seed = (unsigned int)time(NULL);
    void (*function)(double *, long long, int) = prefixSumShared;
    char *outputFile = "/dev/null";
    char *inputFile = "/dev/null";
    int threads = 8;
    parseArguments(argc, argv, &size, &seed, &function, &outputFile, &inputFile, &threads);

    //  printf("Size: %lld\n", size);
    //    printf("Function: %s\n", function == prefixSumShared ? "prefixSum" : "prefixMult");
    // print the number of threads
    omp_set_num_threads(threads);
#define num_threads threads
    printf("Threads: %d\n", threads);
    double *arr = initializeArray(size, seed);
    //   printf("Input file: %s\n", inputFile);
    writeOutputFile(arr, size, inputFile);

    double start = omp_get_wtime();
    function(arr, size, num_threads);
    double end = omp_get_wtime();
    // do time and size on the same line
    // printf("Time: %f milliseconds \n", (end - start) * 1000.0);
    printf("%llu , %f \n", size, (end - start) * 1000.0);
    //   printf("Output file: %s\n", outputFile);
    writeOutputFile(arr, size, outputFile);
    free(arr);
    return 0;
}
