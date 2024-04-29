#!/bin/bash
#SBATCH  -p gpu-shared --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

nvcc -arch=sm_86 -O3 --compiler-options -march=native PrefixSumCuda.cu -o PrefixSumCuda


for i in {1..50}
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

    ./PrefixSumCuda -s 64  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 64 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 128  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 128 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 256  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 256 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 512  >>Cuda/Cudafloat.txt    
    ./PrefixSumCuda -s 512 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 1024  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 1024 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 2048  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 2048 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 4096  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 4096 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 8192  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 8192 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 16384  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 16384 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 32768  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 32768 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 65536  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 65536 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 131072  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 131072 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 262144  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 262144 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 524288  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 524288 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 1048576  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 1048576 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 2097152  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 2097152 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 4194304  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 4194304 -t double  >>Cuda/CudaSize_double.txt
    
    ./PrefixSumCuda -s 8388608  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 8388608 -t double  >>Cuda/CudaSize_double.txt


    ./PrefixSumCuda -s 16777216  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 16777216 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 33554432  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 33554432 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 67108864  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 67108864 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 134217728  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 134217728 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 268435456  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 268435456 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 536870912  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 536870912 -t double  >>Cuda/CudaSize_double.txt

    ./PrefixSumCuda -s 1073741824  >>Cuda/Cudafloat.txt
    ./PrefixSumCuda -s 1073741824 -t double  >>Cuda/CudaSize_double.txt


    
done