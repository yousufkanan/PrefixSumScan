#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /home/kanany/PrefixSumScan/Output/SerialO3MarchNative/SerialO3MarchNativeSize%j.out
#SBATCH -e /home/kanany/PrefixSumScan/SerialO3MarchNative/SerialO3MarchNativeSize%j.err

#SBATCH --export=ALL

gcc -Wall -O3 -march=native PrefixSumSerial.c -o PrefixSumSerialO3MarchNative -lm
#make direectory if it doesn't exist
mkdir -p Serial/O3MarchNative
for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 10 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize10.txt & disown 
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 100 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize100.txt & disown 
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 1000 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize1000.txt
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 10000 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize10000.txt & disown
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 100000 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativesSize100000.txt & disown 
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 1000000 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize1000000.txt & disown 
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 10000000 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize10000000.txt & disown 
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 100000000 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize100000000.txt & disown 
done

for i in {1..50}
    sun ./PrefixSumSerialO3MarchNative -s 1000000000 -r 0 >>Serial/O3MarchNative/SerialO3MarchNativeSize1000000000.txt & disown 
done


