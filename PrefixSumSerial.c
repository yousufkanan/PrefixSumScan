/*
     * This program calculates the prefix sum or prefix multiplication of an array of random numbers.
     * This approach is serial and uses a single thread.
     * The program takes the following command line arguments:
     * -s <size> : size of the array (default: 4194304)
     * -r <seed> : seed for the random number generator (default: current time)
     * -f <function> : function to execute (prefixSum or prefixMult, default: prefixSum)
     * -o <outputFile> : output file to write the results (default: /dev/null)
     * -i <inputFile> : input file to read the array from (default: /dev/null)
     * Example: ./PrefixSumSerial -s 100 -r 1 -f prefixSum -o output.txt -i input.txt
     * Instructor: Dr. Jeffery Bush
     * compile with: gcc-13 -Wall -O3 -march=native PrefixSumSerial.c -o PrefixSumSerial -lm
     * Authors: Yousuf Kanan and Derek Allmon
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void prefixSum(double *arr, int size) {
    for (int i = 1; i < size; i++) {
        arr[i] += arr[i - 1];
    }
}

void prefixMult(double *arr, int size) {
    for (int i = 1; i < size; i++) {
        arr[i] *= arr[i - 1];
    }
}

/**
 * Parses command line arguments and assigns values to corresponding variables.
 *
 * @param argc The number of command line arguments.
 * @param argv An array of strings containing the command line arguments.
 * @param size A pointer to a long long variable to store the size argument.
 * @param seed A pointer to an unsigned int variable to store the seed argument.
 * @param function A pointer to a function pointer to store the function argument.
 * @param outputFile A pointer to a string variable to store the outputFile argument.
 * @param inputFile A pointer to a string variable to store the inputFile argument.
 * chat gpt was used to generate this function
 */
void parseArguments(int argc, char **argv, unsigned long long *size, unsigned int *seed, void (**function)(double *, int), char **outputFile, char **inputFile) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            *size = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            *seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "prefixMult") == 0) {
                *function = prefixMult;
            } else if (strcmp(argv[i + 1], "prefixSum") == 0) {
                *function = prefixSum;
            }
            i++;
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            *outputFile = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            *inputFile = argv[++i];
        } else {
            fprintf(stderr, "Unknown option or missing argument: %s\n", argv[i]);
            exit(1);
        }
    }
}

/**
 * Initializes an array of random numbers.
 *
 * @param size The size of the array.
 * @param seed The seed for the random number generator.
 * @return A pointer to the array of random numbers.
 */
double *initializeArray(int size, unsigned int seed) {
    double *arr = (double *)malloc(size * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    srand(seed);
    for (int i = 0; i < size; i++) {
        arr[i] = (double)rand() / (double)RAND_MAX * 2;
       // arr[i] = i;
    }
    return arr;
}

/**
 * Executes a function on an array and measures the time taken.
 * The function is executed on the array in place.
 * The time taken is measured using clock_gettime.
 * The time taken is returned in milliseconds.
 * @param arr The array to operate on.
 * @param size The size of the array.
 * @param function The function to execute on the array.
 * @return The time taken to execute the function in milliseconds.
 * found information on clock_gettime at stackoverflow.com
*/
double executeFunction(double *arr, int size, void (*function)(double *, int)) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    function(arr, size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    //cpu time used in milliseconds
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

/**
 * Writes the results of the array to a file.
 *
 * @param filename The name of the file to write the results to.
 * @param arr The array of results.
 * @param size The size of the array.
 * chat gpt was used to generate this function
 */
void writeResultsToFile(const char *filename, double *arr, int size) {
    if (strcmp(filename, "/dev/null") == 0) {
        return;
    }
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fprintf(fp, "%f\n", arr[i]);
    }
    fclose(fp);
}

/**
 * Main function that parses command line arguments, 
 * Initializes an array, executes a function on the array, and writes the results to a file.
 */
int main(int argc, char **argv) {
    unsigned long long size = 4194304;
    unsigned int seed = (unsigned int)time(NULL);
    void (*function)(double *, int) = prefixSum;
    char *outputFile = "/dev/null";
    char *inputFile = "/dev/null";

    parseArguments(argc, argv, &size, &seed, &function, &outputFile, &inputFile);

    printf("Size: %lld\n", size);
    printf("Function: %s\n", function == prefixSum ? "prefixSum" : "prefixMult");
  
    double *arr = initializeArray(size, seed);

      
    printf("Input file: %s\n", inputFile);
    writeResultsToFile(inputFile, arr, size);

    double cpu_time_used = executeFunction(arr, size, function);
    printf("Time taken: %f milliseconds\n", cpu_time_used);
    
    printf("Output file: %s\n", outputFile);
    writeResultsToFile(outputFile, arr, size);

    free(arr);
    
    return 0;
}
