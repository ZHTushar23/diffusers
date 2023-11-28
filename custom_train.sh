#!/bin/bash 
#SBATCH --job-name=CustomD        # Job name
#SBATCH --mail-user=ztushar1@umbc.edu       # Where to send mail
#SBATCH --mem=20000                       # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=72:00:00                   # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_8000            # Specific hardware constraint
#SBATCH --error=v1.err                # Error file name
#SBATCH --output=v1.out               # Output file name

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="saved-model"
export INSTANCE_DIR="./data/cat"
export rel=True

python examples/custom_diffusion/train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_cat/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="cat" --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=5 \
  --scale_lr --hflip  \
  --no_safe_serialization\
  --modifier_token "<new1>" 