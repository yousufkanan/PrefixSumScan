/**
 * gcc-13 -O3 -fopenmp -march=native PrefixSumSharedOpenMPBuiltin.c -o PrefixSumSharedOpenMPBuiltin -lm
 * ./PrefixSumSharedOpenMPBuiltin -s 100 -r 1 -f prefixSum -o output.txt -i input.txt -t 8
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void prefixSumShared(double *arr, int size, int numThreads) {
/*
    #pragma omp simd reduction (inscan, +: a)
    for (i = 0; i < 64; i++) {
    int t = a;

    d[i] = t;
    #pragma omp scan inclusive (a)
    int u = c[i];
    a += u;
    }
    */
    double* output = (double*)malloc(size * sizeof(double));
    if (output == NULL) {
        // Handle memory allocation failure if needed
        return;
    }


}


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
    srand(seed);
    for (int i = 0; i < size; i++) {
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
     
   //  printf("Size: %lld\n", size);
 //    printf("Function: %s\n", function == prefixSumShared ? "prefixSum" : "prefixMult");
     //print the number of threads
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
