#%%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from scipy.fftpack import dct

from models import DCTModel
from dataset import TextMotionDataset
from utils import now_time, AverageMeter, load_encode

import os

seed = 42
torch.manual_seed(seed)
#%%
batch_size = 60
data_path = './data/extro_dct_dataset.npz'
word2idx = load_encode('encode_dict.txt')
idx2word = {v: k for k, v in word2idx.items()}

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
dof_num = 3
top_frequency_num = 10
output_dim = dof_num * top_frequency_num
embed_dim = 100
learning_rate = 1e-3
l1_penalty_coef = 1e-5
hidden_dim = 200
model_save_folder = './saved_models/'

num_epochs = 100
best_loss = 1000
epochs_since_improvement = 0

# load pretrained word embeddings
embedding = np.load('glove_pretrained_weights.npy')
embedding = torch.from_numpy(embedding).float()

model = DCTModel(embed_dim=embed_dim,
                              hidden_dim=hidden_dim,
                              out_dim=output_dim,
                              vocab_size=len(embedding),
                              pretrain_weight=embedding)

# setting optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%%
# load saved models
save_state_dict = torch.load(model_save_folder+'best_checkpoint_DCT_best')
model.load_state_dict(save_state_dict['model'])
optimizer.load_state_dict(save_state_dict['optim'])
epoch = save_state_dict['epoch']
valid_loss = save_state_dict['valid_loss']
epochs_since_improvement = save_state_dict['epochs_since_improvement']
model.eval()

losses = AverageMeter()
training_date = now_time()
total_diff = []

#%%
prediction_list = []
target_list = []
input_list = []
print('Evaluating on test set...')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

for idx, (input, target) in enumerate(test_dataloader):
    with torch.no_grad():
        prediction = model(input)

    # calculate loss
    loss = F.mse_loss(prediction.squeeze(), target.float())
    losses.update(loss.item(), len(input))


    prediction_list.append(prediction.view(3, 10).numpy())
    target_list.append(target.float().view(3, 10).numpy())
    input_list.append(input.view(-1).numpy())

print('\nThe average loss on test set is: ', losses.avg)

input_arr = np.array(input_list)
target_arr = np.array(target_list)
prediction_arr = np.array(prediction_list)


#%%
# convert back to time domain
target_traj = np.empty_like(target_arr)
prediction_traj = np.empty_like(prediction_arr)

for i in range(len(target_arr)):
    for j in range(3):
        target_traj[i, j] = dct(target_arr[i, j], type=3, norm='ortho')
        prediction_traj[i, j] = dct(prediction_arr[i, j], type=3, norm='ortho')

# evalulate STD and corr
std_pred = np.mean(np.std(prediction_traj, axis=-1), axis=0)
std_target = np.mean(np.std(target_traj, axis=-1), axis=0)

# calculate pearson correlation
pearson_corr = np.empty((len(target_traj), 3))
for i in range(len(target_traj)):
    for j in range(3):
        pearson_corr[i, j] = np.corrcoef(target_traj[i, j],
                                         prediction_traj[i, j])[0, 1]

pearson_corr = np.mean(pearson_corr,axis=0)

# calculate MSE on time domain
mse_traj_loss = np.mean(np.mean((target_traj - prediction_traj)**2, axis=0), axis=-1)
