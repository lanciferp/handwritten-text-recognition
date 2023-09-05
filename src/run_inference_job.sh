#!/bin/bash
#SBATCH --cpus-per-task=8   # number of processor cores (i.e., tasks)
#SBATCH --mem=50G   # memory per CPU core
#SBATCH -J "Flor-Main"   # job name
#SBATCH -e /shared/home/cyclemgmt/handwritten-text-recognition-master/src/flor_out/%j-err.txt
#SBATCH -o /shared/home/cyclemgmt/handwritten-text-recognition-master/src/flor_out/%j-out.txt
#SBATCH -p htc

#Usage sbatch run_inference_job.sh Directory ColumnName WeightsName

module load python/3.8
source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

python3 main.py "--job_config" "$1"
