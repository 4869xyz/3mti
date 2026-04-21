import random
import torch

def random_zero_along_dim(tensor, dim=1):

    n = tensor.shape[dim]

    keep_idx = random.randint(0, n - 1)

    mask = torch.ones(n, dtype=tensor.dtype, device=tensor.device)
    for i in range(n):
        if i != keep_idx and random.random() < 0.5: 
            mask[i] = 0

    mask[keep_idx] = 1

    shape = [1] * tensor.ndim
    shape[dim] = n
    mask = mask.view(*shape)

    return tensor * mask 