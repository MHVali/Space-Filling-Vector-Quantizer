"""
An example code to show how to train the Space-FillingVQ module on a Normal distribution. Notice that the bitrate
for Space-FillingVQ has to be increased step by step during training, starting from 2 bits (4 corner points) to
desired bitrate (2**desired_bits corner points).
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np

from spacefilling_vq import SpaceFillingVQ
from utils import codebook_initialization, codebook_extension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
desired_vq_bitrate = 5
codebook_extension_eps = 0.01
embedding_dim = 128
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

# Data distribution which we apply SpaceFillingVQ on it
data = np.random.randn(int(2**18), embedding_dim).astype(np.float32)
data = torch.from_numpy(data).to(device)

num_batches = int(data.shape[0] / batch_size)
milestones = [int(num_epochs*0.6), int(num_epochs*0.8)]

# Arrays to save the logs of training
total_vq_loss = np.zeros((desired_vq_bitrate - 1, num_epochs)) # tracks VQ loss
total_perplexity = np.zeros((desired_vq_bitrate - 1, num_epochs)) # tracks perplexity
used_codebook_indices_list = [] # tracks indices of used codebook entries

initial_codebook = codebook_initialization(torch.randn(int(1e4),embedding_dim)).to(device)
vector_quantizer = SpaceFillingVQ(desired_vq_bitrate, embedding_dim, device=device, initial_codebook=initial_codebook)
vector_quantizer.to(device)

for bitrate in range(2, desired_vq_bitrate+1):

    optimizer = optim.Adam(vector_quantizer.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    entries = int(2 ** bitrate) # Current bitrate for Space-FillingVQ (current number of corner points)
    used_codebook_indices = np.zeros((entries,))

    if bitrate > 2: # Codebook extension phase (increasing sapce-FillingVQ bitrate/corner points)
        final_indices = codebook_extension(vector_quantizer.entries, codebook_extension_eps).to(device)
        codebook = vector_quantizer.decode(final_indices)
        vector_quantizer.codebook.data[0:int(2**bitrate)] = codebook


    for epoch in range(num_epochs):

        vq_loss_accumulator = perplexity_accumulator = 0

        print(f'<<<<<<<<<<########## VQ Bitrate = {bitrate} | Epoch = {epoch + 1} ##########>>>>>>>>>>')

        for i in range(num_batches):
            data_batch = data[i*batch_size : (i+1)*batch_size]

            optimizer.zero_grad()

            quantized, perplexity, selected_indices = vector_quantizer(data_batch, entries)

            vq_loss = F.mse_loss(data_batch, quantized) # use this loss if you are exclusively training only the
                                                        # Space-FillingVQ module.

            vq_loss.backward()
            optimizer.step()

            used_codebook_indices[selected_indices] += 1
            used_codebook_indices[selected_indices+1] += 1

            vq_loss_accumulator += vq_loss.item()
            perplexity_accumulator += perplexity.item()

            vq_loss_average = vq_loss_accumulator / (i+1)
            perplexity_average = perplexity_accumulator / (i+1)

        total_vq_loss[bitrate-2, epoch] = vq_loss_average
        total_perplexity[bitrate-2, epoch] = perplexity_average

        scheduler.step()

        # printing the training logs for each epoch
        print("epoch:{}, vq loss:{:.6f}, perpexlity:{:.4f}".format(epoch+1, vq_loss_average, perplexity_average))

    used_codebook_indices_list.append(used_codebook_indices)

# saving the training logs and Space-FillingVQ trained model
np.save(f'total_sfvq_loss_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', total_vq_loss)
np.save(f'total_perplexity_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', total_perplexity)
np.save(f'codebook_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', vector_quantizer.codebook.cpu().detach().numpy())

with open(f"used_codebook_indices_list_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}", "wb") as fp:
    pickle.dump(used_codebook_indices_list, fp)

checkpoint_state = {"vector_quantizer": vector_quantizer.state_dict()}
torch.save(checkpoint_state, f"vector_quantizer_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.pt")

print("\nTraining Finished >>> Logs and Checkpoints Saved!!!")

######################## Evaluation (Inference) of Space-FillingVQ #############################

quantized_data = torch.zeros_like(data)

eval_batch_size = 64
num_batches = int(data.shape[0]/eval_batch_size)
with torch.no_grad():
    for i in range(num_batches):
        data_batch = data[(i*eval_batch_size):((i+1)*eval_batch_size)]
        quantized_data[(i*eval_batch_size):((i+1)*eval_batch_size)] = vector_quantizer.evaluation(data_batch)

mse = F.mse_loss(data, quantized_data).item()
print("Mean Squared Error = {:.4f}".format(mse))

