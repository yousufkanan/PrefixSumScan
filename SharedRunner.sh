gcc -Wall -O3 -fopenmp -march=native PrefixSumShared.c -o PrefixSumShared -lm

for i in {1..50}
do 
srun ./PrefixSumShared -s 10  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 100  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 1000  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 10000  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 100000  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 1000000  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 10000000  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 100000000  >>Shared/Shared.txt & disown

srun ./PrefixSumShared -s 1000000000  >>Shared/Shared.txt & disown
done

for i in {1..1000}
do
#srun with random size between 100 and 1000000000
srun ./PrefixSumShared -s $((10 + RANDOM % 1000000000))  >>Shared/Shared.txt & disown
done