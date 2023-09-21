#!/bin/bash

#SBATCH --cpus-per-task=1   # number of processor cores (i.e., tasks)
#SBATCH --mem=7G   # memory per CPU core
#SBATCH -J "Flor-Batch"   # job name
#SBATCH -e ./flor_out/%j-err.txt
#SBATCH -o ./flor_out/%j-out.txt
#SBATCH -p htc

source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

delete_finished=$4
start_point=$5
batch_size=$6

arguments=" --source $1 --weights $2 --csv $3 --append"

if [ $delete_finished == "True" ]; then
  arguments="${arguments} --delete_finished "
fi

if [ $start_point != 0 ]; then
  arguments="${arguments} --start_point ${5} --batch_size ${6}"
fi

echo $arguments
python3 -u main.py $arguments
