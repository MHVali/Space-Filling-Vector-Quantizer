"""
Code to plot the training logs saved during training the code "train.py". The plots will be saved as a pdf file.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# hyper-parameters you used for training. Now they are needed to load your saved arrays.
batch_size = 64
desired_vq_bitrate = 6
learning_rate = 1e-3

# create pdf file
pdf_file = PdfPages(f'log_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.pdf')

# loading the training logs
total_vq_loss = np.load(f'total_vq_loss_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy')
total_perplexity = np.load(f'total_perplexity_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy')
with open(f"used_codebook_indices_list_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}", "rb") as fp:
    used_codebook_indices_list = pickle.load(fp)

num_epochs = total_vq_loss.shape[1]

# plotting used codebook indices during training for each bitrate
for i in range(len(used_codebook_indices_list)):
    num_bars = np.size(used_codebook_indices_list[i])
    histogram = np.log10(used_codebook_indices_list[i] + 1)
    fig = plt.figure(figsize=(10,6))
    plt.bar(np.arange(1, num_bars + 1), height=histogram, width=1)
    plt.title(f'SpaceFillingVQ Codebook Usage Histogram | VQ Bitrate = {i + 2}')
    pdf_file.savefig(fig, bbox_inches='tight')

# plotting VQ loss during training
fig = plt.figure(figsize=(15, 5))
total_vq_loss = total_vq_loss.reshape(-1,1)
scatter_index = (np.arange(desired_vq_bitrate - 1) * num_epochs).astype(np.int64)
scatter_index[1:] = scatter_index[1:] - 1
plt.plot(total_vq_loss)
plt.scatter(scatter_index, total_vq_loss[scatter_index], s=50, marker='X', c='red')
for i in range(desired_vq_bitrate - 1):
    plt.annotate(f'{i + 2}Bits', (scatter_index[i], total_vq_loss[scatter_index[i]]))
plt.title(f'VQ Loss')
pdf_file.savefig(fig, bbox_inches='tight')

# plotting perplexity (average codebook usage) during training
fig = plt.figure(figsize=(15, 5))
total_perplexity = total_perplexity.reshape(-1,1)
plt.plot(total_perplexity)
plt.scatter(scatter_index, total_perplexity[scatter_index], s=50, marker='X', c='red')
for i in range(desired_vq_bitrate - 1):
    plt.annotate(f'{i + 2}Bits', (scatter_index[i], total_perplexity[scatter_index[i]]))
plt.title('Perplexity')
pdf_file.savefig(fig, bbox_inches='tight')

pdf_file.close()