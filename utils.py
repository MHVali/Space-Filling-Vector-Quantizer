import numpy as np
import torch

def codebook_extension(entries, eps):
    final_indices = torch.zeros(entries * 2)
    temp1 = torch.arange(0, entries * 2)
    left = temp1[0:entries]
    right = temp1[entries:]
    left_even = left[::2]
    right_even = right[::2]

    temp2 = torch.arange(entries)
    left_fractional_indices = temp2[0:int(entries / 2)].to(torch.float32)
    right_fractional_indices = temp2[int(entries / 2):].to(torch.float32)

    final_indices[left_even] = left_fractional_indices
    final_indices[left_even + 1] = left_fractional_indices + eps
    final_indices[right_even] = right_fractional_indices - eps
    final_indices[right_even + 1] = right_fractional_indices

    return final_indices

def codebook_initialization(vectors):

    num_samples = vectors.shape[0]
    embedding_dim = vectors.shape[1]

    if type(vectors) is np.ndarray:
        vectors = torch.from_numpy(vectors)

    norm_vectors = torch.linalg.norm(vectors, dim=1)
    _,indices = torch.sort(norm_vectors)
    sorted_vectors = vectors[indices]

    initial_codebook = torch.zeros((4, embedding_dim))
    hop_size = int(num_samples / 4)
    for i in range(4):
        initial_codebook[i] = torch.mean(sorted_vectors[i * hop_size:(i + 1) * hop_size], dim=0)

    return initial_codebook