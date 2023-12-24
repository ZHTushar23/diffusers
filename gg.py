import torch
def get_time_embedding(timesteps):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # print(freqs.shape)
    # Shape: (batch_size, 1, 160)
    # timesteps = timesteps.unsqueeze(-1)
    # timesteps = timesteps.unsqueeze(-1)
    # print(timesteps.shape)
    x = timesteps * freqs[None, :]
    # print(x.shape)
    # Shape: (batch_size, 1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


batch_size = 1
timesteps = torch.randint(0, 100, (batch_size,))
time_embeddings = get_time_embedding(timesteps)
print("Time Embeddings shape:", time_embeddings.shape)

