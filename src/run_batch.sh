#!/bin/bash

#SBATCH --cpus-per-task=1   # number of processor cores (i.e., tasks)
#SBATCH --mem=7G   # memory per CPU core
#SBATCH -J "Flor-Batch"   # job name
#SBATCH -e ./flor_out/%j-err.txt
#SBATCH -o ./flor_out/%j-out.txt
#SBATCH -p htc


source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

delete_finished = $5
start_point = $6
batch_size = $7

arguments = "--source $1 --weights $2 --csv $3 --append --finished $4"

if [$delete_finished]; then
  arguments += "--delete_finished"
  exit
fi

if [start_point]; then
  arguments += " --start_point "
  arguments += $6
  arguments += " --batch_size "
  arguments += $7

python3 -u main.py "$arguments"
