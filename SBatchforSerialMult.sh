#!/bin/bash
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
    
   # echo "Running Serial Mult"  
     ./PrefixSumSerial -s 2 -f prefixMult >>Serial/SerialSize_Mult.txt 

    
     ./PrefixSumSerial -s 4 -f prefixMult >>Serial/SerialSize_Mult.txt 

    
     ./PrefixSumSerial -s 8 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 16 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 32  -f prefixMult >>Serial/SerialSize_Mult.txt 

    
     ./PrefixSumSerial -s 64 -f prefixMult >>Serial/SerialSize_Mult.txt 

    
     ./PrefixSumSerial -s 128 -f prefixMult  >>Serial/SerialSize_Mult.txt 

   
     ./PrefixSumSerial -s 256 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 512 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 1024 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 2048 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 4096 -f prefixMult  >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 8192 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 16384 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 32768 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 65536 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 131072 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 262144 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 524288  -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 1048576 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 2097152 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 4194304 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 8388608  -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 16777216 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 33554432 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 67108864 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 134217728 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 268435456 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 536870912 -f prefixMult  >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 1073741824 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 2147483648 -f prefixMult  >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 4294967296 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 8589934592 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 17179869184 -f prefixMult >>Serial/SerialSize_Mult.txt 
    
     ./PrefixSumSerial -s 34359738368 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 68719476736 -f prefixMult >>Serial/SerialSize_Mult.txt 

     ./PrefixSumSerial -s 137438953472 -f prefixMult >>Serial/SerialSize_Mult.txt 

    #echo 

done
