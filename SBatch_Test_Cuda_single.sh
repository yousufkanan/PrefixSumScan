#!/bin/bash
#SBATCH  -p gpu-shared --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL

nvcc -arch=sm_86 -O3 --compiler-options -march=native PrefixSumCuda.cu -o PrefixSumCuda


echo "Running Cuda Prefix Sum with float" >>Cuda/Cudafloat.txt
./PrefixSumCuda -s 2   >>Cuda/Cudafloat.txt 
   
    
