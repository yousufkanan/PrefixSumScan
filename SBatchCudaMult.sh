#!/bin/bash
#SBATCH  -p gpu-shared --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

nvcc -arch=sm_86 -O3 --compiler-options -march=native PrefixSumCuda.cu -o PrefixSumCuda

for i in {1..50}
do
    
     ./PrefixSumCuda -s 2  -m  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 2 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 4  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 4 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 8  -m    >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 8 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 16  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 16 -t double   -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 32  -m    >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 32 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 64  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 64 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 128  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 128 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 256  -m    >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 256 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 512  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 512 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 1024  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 1024 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 2048  -m    >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 2048 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 4096  -m    >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 4096 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 8192  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 8192 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 16384  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 16384 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 32768  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 32768 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 65536   -m  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 65536 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 131072   -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 131072 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 262144  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 262144 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 524288 -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 524288 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 1048576  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 1048576 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 2097152  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 2097152 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 4194304  -m    >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 4194304 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 8388608  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 8388608 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 16777216  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 16777216 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 33554432  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 33554432 -t double  -m    >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 67108864  -m   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 67108864 -t double  -m   >>Cuda/CudaSize_mult_double.txt 

    done 






























