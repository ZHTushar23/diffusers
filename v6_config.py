'''
create a TrainingConfig class containing the training hyperparameters 
'''

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 72  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 10  # how many images to sample during evaluation
    num_epochs = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-rad-72-cond-e2d-no-rescale"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42

if __name__=="__main__":
    config = TrainingConfig()