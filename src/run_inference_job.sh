#!/bin/bash
#SBATCH --cpus-per-task=1   # number of processor cores (i.e., tasks)
#SBATCH --mem=7   # memory per CPU core
#SBATCH -J "Flor-Main"   # job name
#SBATCH -e /shared/home/cyclemgmt/handwritten-text-recognition/src/flor_out/%j-err.txt
#SBATCH -o /shared/home/cyclemgmt/handwritten-text-recognition/src/flor_out/%j-out.txt
#SBATCH -p htc

#Usage sbatch run_inference_job.sh Directory ColumnName WeightsName

module load python/3.8
source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

python3 OneLine.py "--job_config" "$1"
