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

#include <stdio.h>
#define N 100


int main(void)
{
     int a[N], b[N];
     int x = 0;
     // initialization
     for (int k = 0; k < N; k++)
          a[k] = k + 1;

     // a[k] is included in the computation of producing results in b[k]
     #pragma omp parallel for simd reduction(inscan,+: x)
     for (int k = 0; k < N; k++) {
     x += a[k];
     #pragma omp scan inclusive(x)
     b[k] = x;
     }

     for (int k = 0; k < N; k++)
          printf("%d, ", b[k]);
 // 5050, 1 3 6

 return 0;
 }
 //COMPILE with gcc-13 -O3 -fopenmp -march=native PrefixSumSharedOpenMPBuiltin.c -o PrefixSumSharedOpenMPBuiltin -lm