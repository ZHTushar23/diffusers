#!/bin/bash 
#SBATCH --job-name=v1test      # Job name
#SBATCH --mail-user=ztushar1@umbc.edu       # Where to send mail
#SBATCH --mem=30000                       # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=72:00:00                   # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000            # Specific hardware constraint
#SBATCH --error=v1test.err                # Error file name
#SBATCH --output=v1test.out               # Output file name

python v1_test.py  