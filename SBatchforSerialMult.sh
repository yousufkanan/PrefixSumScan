
gcc -Wall -O3 -march=native PrefixSumSerial.c -o PrefixSumSerial -lm


for i in {1..50}
do
    
   # echo "Running Serial with size 2" 
    srun ./PrefixSumSerial -s 2 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    #echo "Running Serial with size 4"
    srun ./PrefixSumSerial -s 4 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    
    srun ./PrefixSumSerial -s 8 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 16 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 32  -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    
    srun ./PrefixSumSerial -s 64 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    
    srun ./PrefixSumSerial -s 128 -f prefixMult  >>Serial/SerialSize_Mult.txt & disown

   
    srun ./PrefixSumSerial -s 256 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 512 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 1024 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 2048 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 4096 -f prefixMult  >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 8192 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 16384 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 32768 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 65536 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 131072 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 262144 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 524288  -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 1048576 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 2097152 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 4194304 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 8388608  -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 16777216 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 33554432 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    srun ./PrefixSumSerial -s 67108864 -f prefixMult >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 134217728  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 268435456  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 536870912  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 1073741824  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 2147483648  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 4294967296  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 8589934592  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 17179869184  >>Serial/SerialSize_Mult.txt & disown
    
    # srun ./PrefixSumSerial -s 34359738368  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 68719476736  >>Serial/SerialSize_Mult.txt & disown

    # srun ./PrefixSumSerial -s 137438953472  >>Serial/SerialSize_Mult.txt & disown

    #echo 

done
