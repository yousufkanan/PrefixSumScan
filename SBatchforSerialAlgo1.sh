#!/bin/bash
#SBATCH -p high-mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --export=ALL


gcc -Wall -O3 -fopenmp -march=native PrefixSumSharedAlgorirthm1.c -o PrefixSumSharedAlgorirthm1 -lm

for i in {1..50}
do 

./PrefixSumSharedAlgorirthm1 -s 2 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 2 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 4 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 4 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 8 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 8 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 16 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 16 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 32 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 32 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 64 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 64 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 128 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 128 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 256 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 256 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 512 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 512 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 1024 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 1024 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 2048 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 2048 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 4096 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 4096 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 8192 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 8192 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 16384 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 16384 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 32768 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 32768 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 65536 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 65536 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 131072 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 131072 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 262144 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 262144 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 524288 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 524288 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 1048576 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 1048576 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 2097152 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 2097152 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 4194304 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 4194304 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 8388608 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 8388608 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 16777216 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 16777216 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 33554432 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 33554432 >> Shared/SharedAlgo1_Mult.txt

./PrefixSumSharedAlgorirthm1 -s 67108864 >> Shared/SharedAlgo1.txt
./PrefixSumSharedAlgorirthm1 -f prefixMult -s 67108864 >> Shared/SharedAlgo1_Mult.txt

done


