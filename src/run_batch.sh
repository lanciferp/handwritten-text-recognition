#!/bin/bash

#SBATCH --cpus-per-task=1   # number of processor cores (i.e., tasks)
#SBATCH --mem=7G   # memory per CPU core
#SBATCH -J "Flor-Batch"   # job name
#SBATCH -e ./flor_out/%j-err.txt
#SBATCH -o ./flor_out/%j-out.txt
#SBATCH -p htc
#export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -qq ffmpeg libsm6 libxext6 -y

delete_finished=$4

arguments=" --source $1 --weights $2 --csv $3 --append"

if [ $delete_finished == "True" ]; then
  arguments="${arguments} --delete_finished "
fi


echo $arguments
python3 -u main.py $arguments
