#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /home/kanany/PrefixSumScan/Output/SerialO3MarchNative/SerialO3MarchNativeSize%j.out
#SBATCH -e /home/kanany/PrefixSumScan/SerialO3MarchNative/SerialO3MarchNativeSize%j.err

#SBATCH --export=ALL

gcc -Wall -O3 -fopenmp -march=native PrefixSumShared.c -o PrefixSumShared -lm

for i in {1..50}
do 
 ./PrefixSumShared -s 10  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 100  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 1000  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 10000  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 100000  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 1000000  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 10000000  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 100000000  >>Shared/Shared.txt & disown

 ./PrefixSumShared -s 1000000000  >>Shared/Shared.txt & disown


done

for i in {1..1000}
do
# with random size between 100 and 1000000000
 ./PrefixSumShared -s $((10 + RANDOM % 1000000000))  >>Shared/Shared.txt & disown
done