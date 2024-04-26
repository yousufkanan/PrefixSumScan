#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /home/kanany/PrefixSumScan/Output/SerialO3MarchNative/SerialO3MarchNativeSize%j.out
#SBATCH -e /home/kanany/PrefixSumScan/SerialO3MarchNative/SerialO3MarchNativeSize%j.err

#SBATCH --export=ALL

for i in {1..50}
    #SerialRunO3MarchNative.sh will be called here with the appropriate arguments

done
