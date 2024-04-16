/*
Authors: Yousuf Kanan and Derek Almon
Compile on Mucluster: gcc -Wall PrefixSumSeral.c -o PrefixSumSeral -lm 
Compile on Expert: gcc -Wall PrefixSumSeral.c -o PrefixSumSeral
comile locally : gcc-13 -Wall PrefixSumSeral.c -o PrefixSumSeral -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <util.h>
#include <time.h>
// Function to prfix sum the array
void prefixSum(double *arr)
{
    for (int i = 1; i < sizeof(arr); i++)
    {
        arr[i] = arr[i] + arr[i - 1];
    }
}

void prefixMult(double *arr)
{
    for (int i = 1; i < sizeof(arr); i++)
    {
        arr[i] = arr[i] * arr[i - 1];
    }
}
int main(int argc, char **argv)
{
    // Check if the user has entered the correct number of arguments
    // Arguments are the size of the npy array, the seed for the random number generator
    // and the type of function prefixSum or PrefixMult to be used
    // if their is no argument for size of npy array, the default size is 2048 elements
    // if their is no argument for seed, the default seed is random 
    // if their is no argument for function type, the default function is prefixSum
    int size = 2048;
    unsigned int seed = (unsigned int)time(NULL); // Default seed based on current time
    void (*function)(int *, int) = prefixSum; // Default function

    // Check for the first argument: size of the array
    if (argc > 1) {
        size = atoi(argv[1]); 
    }

    // Check for the second argument: seed for the random number generator
    if (argc > 2) {
        seed = (unsigned int)atoi(argv[2]);
    }
    srand(seed);
    double *arr = (double *)malloc(size * sizeof(double));

    for (int i = 0; i < size; i++)
    {
        // Generate random numbers between 0 and 1
        arr[i] = (double)rand() / (double)RAND_MAX;
    }

    printf("\n");
    printf("\n ");
    // Check for the third argument: function type
    //start timer 
    double start = get_time();
    if (argc > 3) {
        if (strcmp(argv[3], "prefixMult") == 0) {
            

            prefixMult(arr);
        } 
        else if (strcmp(argv[3], "prefixSum") == 0) {
            prefixSum(arr);
        } 
        else {
            // ivalid continuing iwth prefix sum
            printf("Invalid function type. Continuing with prefixSum.\n");
            prefixSum(arr);
            
        }
    }
    else {
        prefixSum(arr);
    }

    //end timer
    double end = get_time();
    printf("Time: %f\n", end - start);
    // Print the array
    for (int i = 0; i < size; i++)
    {
        printf("%f ", arr[i]);
    }
    return 0;

}   
