#!/bin/bash 
#SBATCH --job-name=DFtrain       # Job name
#SBATCH --mail-user=ztushar1@umbc.edu       # Where to send mail
#SBATCH --mem=40000                       # Job memory request
#SBATCH --gres=gpu:4                     # Number of requested GPU(s) 
#SBATCH --time=72:00:00                   # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_8000            # Specific hardware constraint
#SBATCH --error=ddpm1.err                # Error file name
#SBATCH --output=ddpm1.out               # Output file name


export CUDA_LAUNCH_BLOCKING=1
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py