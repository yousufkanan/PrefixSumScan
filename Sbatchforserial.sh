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
gcc -Wall -O3 -march=native PrefixSumSerial.c -o PrefixSumSerial -lm


for i in {1..50}
do
    
   # echo "Running Serial with size 2" 
 ./PrefixSumSerial -s 2  >>Serial/SerialSize.txt

    #echo "Running Serial with size 4"
 ./PrefixSumSerial -s 4  >>Serial/SerialSize.txt

    
 ./PrefixSumSerial -s 8  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 16  >>Serial/SerialSize.txt

   
 ./PrefixSumSerial -s 32  >>Serial/SerialSize.txt

    
 ./PrefixSumSerial -s 64  >>Serial/SerialSize.txt

    
 ./PrefixSumSerial -s 128  >>Serial/SerialSize.txt

   
 ./PrefixSumSerial -s 256  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 512  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 1024  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 2048  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 4096  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 8192  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 16384  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 32768  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 65536  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 131072  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 262144  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 524288  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 1048576  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 2097152  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 4194304  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 8388608  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 16777216  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 33554432  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 67108864  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 134217728  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 268435456  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 536870912  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 1073741824  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 2147483648  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 4294967296  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 8589934592  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 17179869184  >>Serial/SerialSize.txt
    
 ./PrefixSumSerial -s 34359738368  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 68719476736  >>Serial/SerialSize.txt

 ./PrefixSumSerial -s 137438953472  >>Serial/SerialSize.txt

    #echo 

done
