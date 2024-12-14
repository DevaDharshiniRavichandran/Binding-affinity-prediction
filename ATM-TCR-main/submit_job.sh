#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 16            # number of cores 
#SBATCH -t 0-08:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="dravich6@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment


source activate myenv

conda activate py3_env

#Run the software/python script
python main.py --infile data_new/BAP/epi_split/train.csv --split_type epitope --lr 0.01
