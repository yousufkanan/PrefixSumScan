#!/bin/bash
#SBATCH  -p gpu-shared --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

nvcc -arch=sm_86 -O3 --compiler-options -march=native PrefixSumCuda.cu -o PrefixSumCuda


for i in {1..10}
do
    
    ./PrefixSumCuda -s 2   >>Cuda/Cudafloat.txt 
    ./PrefixSumCuda -s 2 -t double  >>Cuda/CudaSize_double.txt 

    ./PrefixSumCuda -s 4  >>Cuda/Cudafloat.txt 
    ./PrefixSumCuda -s 4 -t double  >>Cuda/CudaSize_double.txt 

    ./PrefixSumCuda -s 8  >>Cuda/Cudafloat.txt 
     ./PrefixSumCuda -s 8 -t double  >>Cuda/CudaSize_double.txt 

     ./PrefixSumCuda -s 16  >>Cuda/Cudafloat.txt 
     ./PrefixSumCuda -s 16 -t double  >>Cuda/CudaSize_double.txt 

     ./PrefixSumCuda -s 32  >>Cuda/Cudafloat.txt 
     ./PrefixSumCuda -s 32 -t double  >>Cuda/CudaSize_double.txt 
    
done