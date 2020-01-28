#%%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import DCTModel
from dataset import TextMotionDataset
from torch.utils.data import Dataset, DataLoader, random_split

from utils import now_time
from utils import AverageMeter
from utils import load_encode

import os

seed = 42
torch.manual_seed(seed)
#%%
batch_size = 60
data_path = './data/extro_dct_dataset.npz'
word2idx = load_encode('vocab_to_int.txt')

# load and split dataset
tdhm_dataset = TextMotionDataset(data_path)
train_size = int(0.8 * len(tdhm_dataset))
valid_size = int(0.1 * len(tdhm_dataset))
test_size = len(tdhm_dataset) - train_size - valid_size
train_set, valid_set, test_set = random_split(
    tdhm_dataset, [train_size, valid_size, test_size])

train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=1)

#%%
# building model
dof_num = 3
top_freq_num = 10
output_dim = dof_num * top_freq_num
embed_dim = 100
learning_rate = 1e-3
l1_penalty_coef = 1e-5
hidden_dim = 200
model_save_folder = './saved_models/'

num_epochs = 100
best_loss = 1000 # can also use math.inf
epochs_since_improvement = 0

# load pretrained word embeddings
embedding = np.load('glove_pretrained_weights.npy')
embedding = torch.from_numpy(embedding).float()

model = DCTModel(embed_dim=embed_dim,
                              hidden_dim=hidden_dim,
                              out_dim=output_dim,
                              vocab_size=len(embedding),
                              pretrain_weight=None)#embedding)

# setting optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%
training_loss = []
validate_loss = []
losses = AverageMeter()
training_date = now_time()
for epoch in range(num_epochs):
    # start training
    losses.reset()
    model.train()
    for idx, (input, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(input)

        # apply L1 regularisation and L2 loss
        l1_penalty = 0
        for param in model.parameters():
            l1_penalty += torch.norm(param, p=1)
        loss = F.mse_loss(output.squeeze(),
                          target.float()) + l1_penalty_coef * l1_penalty

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), len(input))
        #  print training progress
        if idx % 100 == 0:
            print('Train Epoch:\t{} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, idx, len(train_dataloader), losses.avg))
    training_loss.append(losses.avg)

    # start validation
    losses.reset()
    model.eval()
    for idx, (input, target) in enumerate(valid_dataloader):
        output = model(input)

        # apply L1 regularisation and L2 loss
        l1_penalty = 0
        for param in model.parameters():
            l1_penalty += torch.norm(param, p=1)
        loss = F.mse_loss(output.squeeze(),
                          target.float()) + l1_penalty_coef * l1_penalty

        losses.update(loss.item(), len(input))
        #  print training progress
        if idx % 100 == 0:
            print('Validate Epoch:\t{} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, idx, len(valid_dataloader), losses.avg))
    validate_loss.append(losses.avg)

    # save the best and last model checkpoints
    is_best = losses.avg < best_loss
    if is_best:
        best_loss = losses.avg
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" %
              epochs_since_improvement)

    save_state_dict = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'valid_loss': losses.avg,
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }
    torch.save(save_state_dict,
               model_save_folder + 'last_checkpoint_DCT_' + training_date)
    if is_best:
        torch.save(save_state_dict,
                   model_save_folder + 'best_checkpoint_DCT_' + training_date)
#%%
# save the loss curve
np.savetxt('./result/dct_loss_' + training_date+ '.txt', (training_loss, validate_loss))
