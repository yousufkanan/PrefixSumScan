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
 * compile with: gcc-13 -Wall -O3 -fopenmp -march=native PrefixSumShared.c -o PrefixSumShared -lm
 * Exameple: ./PrefixSumShared -s 100 -r 1 -f prefixSum -o output.txt -i input.txt -t 8
 * Authors: Yousuf Kanan and Derek Allmon
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

void prefixSumShared(double *arr, int size)
{
#pragma omp parallel for ordered
     for (int i = 1; i < size; i++)
     {
#pragma omp ordered
          {
               arr[i] += arr[i - 1];
          }
     }
}

void prefixMultShared(double *arr, int size)
{
#pragma omp parallel for ordered
     for (int i = 1; i < size; i++)
     {
#pragma omp ordered
          {
               arr[i] *= arr[i - 1];
          }
     }
}

void parseArguments(int argc, char **argv, long long *size, unsigned int *seed, void (**function)(double *, int), char **outputFile, char **inputFile, int *threads)
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
    srand(seed);
    for (int i = 0; i < size; i++) {
        arr[i] = (double)rand() / (double)RAND_MAX * 2;
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
     void (*function)(double *, int) = prefixSumShared;
     char *outputFile = "/dev/null";
     char *inputFile = "/dev/null";
     int threads = 8;
     parseArguments(argc, argv, &size, &seed, &function, &outputFile, &inputFile, &threads);
     printf("Size: %lld\n", size);
     printf("Function: %s\n", function == prefixSumShared ? "prefixSum" : "prefixMult");

     double *arr = initializeArray(size, seed);
     printf("Input file: %s\n", inputFile);
     writeOutputFile(arr, size, inputFile);

     double start = omp_get_wtime();
     function(arr, size);
     double end = omp_get_wtime();
     printf("Time: %f milliseconds \n", (end - start) * 1000.0);
     printf("Output file: %s\n", outputFile);
     //print size and time on one line without text
     printf("%lld %f\n", size, (end - start) * 1000.0);

     writeOutputFile(arr, size, outputFile);
     free(arr);
     return 0;
}
