#!/bin/bash

#SBATCH --cpus-per-task=1   # number of processor cores (i.e., tasks)
#SBATCH --mem=7G   # memory per CPU core
#SBATCH -J "NA CDown"   # job name
#SBATCH -e ./flor_out/%j-err.txt
#SBATCH -o ./flor_out/%j-out.txt
#SBATCH -p htc

export DEBIAN_FRONTEND=noninteractive
sudo apt-get install -qq ffmpeg libsm6 libxext6  -y < /dev/null > /dev/null
source /shared/home/cyclemgmt/FlorHTR_env/bin/activate

python3 -u ./Copydown.py "--name" $1 "--last_name" $2 "--relation" $3 "--output" $4