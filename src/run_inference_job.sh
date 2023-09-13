#!/bin/bash
#SBATCH --cpus-per-task=1   # number of processor cores (i.e., tasks)
#SBATCH --mem=7G   # memory per CPU core
#SBATCH -J "Flor-Main"   # job name
#SBATCH -e ./flor_out/%j-err.txt
#SBATCH -o ./flor_out/%j-out.txt
#SBATCH -p htc

#Usage sbatch run_inference_job.sh Directory ColumnName WeightsName

source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

python3 OneLine.py "--job_config" "$1"
