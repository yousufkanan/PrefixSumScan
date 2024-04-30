#!/bin/bash
#SBATCH --p high mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL


gcc -Wall -O3 -fopenmp -march=native PrefixSumSharedAlgorirthm1.c -o PrefixSumSharedAlgorirthm1 -lm

for i in {1..50}
do 

./PrefixSumSharedAlgorirthm1 -s 2 -t 32  >>Shared/SharedSizeAlgo1.txt