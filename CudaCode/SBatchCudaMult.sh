#!/bin/bash
#SBATCH  -p gpu-shared --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

nvcc -arch=sm_86 -O3 --compiler-options -march=native PrefixSumCuda.cu -o PrefixSumCuda

for i in {1..50}
do
    
     ./PrefixSumCuda -s 2  -m true >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 2 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 4  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 4 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 8  -m true   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 8 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 16  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 16 -t double   -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 32  -m true   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 32 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 64  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 64 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 128  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 128 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 256  -m true   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 256 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 512  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 512 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 1024  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 1024 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 2048  -m true   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 2048 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 4096  -m true   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 4096 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 8192  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 8192 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 16384  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 16384 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 32768  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 32768 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 65536   -m true >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 65536 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 131072   -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 131072 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 262144  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 262144 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 524288 -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 524288 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 1048576  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 1048576 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 2097152  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 2097152 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 4194304  -m true   >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 4194304 -t double  -m true  >>Cuda/CudaSize_mult_double.txt

     ./PrefixSumCuda -s 8388608  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 8388608 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 16777216  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 16777216 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 33554432  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 33554432 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 67108864  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 67108864 -t double  -m true  >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 134217728  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 134217728 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 268435456  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 268435456 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 536870912  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 536870912 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

     ./PrefixSumCuda -s 1073741824  -m true  >>Cuda/Cudafloat_mult.txt 
     ./PrefixSumCuda -s 1073741824 -t double  -m true   >>Cuda/CudaSize_mult_double.txt 

    done 





























