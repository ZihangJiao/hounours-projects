#%%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from scipy.fftpack import dct, idct

from models import Seq2SeqModel, nopad_mse_loss
from dataset import Seq2SeqDataset, seq2seq_collate_fn
from utils import now_time
from utils import AverageMeter
from utils import load_encode

import pylab
import time
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns

import math

import os
import json


seed = 42
torch.manual_seed(seed)
#%%
# dataset setting
batch_size = 11


# personality = 'e'
#
# train_data_path = './train_valid_test_data/'+ personality +'_seq2seq_dataset_train.npz'
# valid_data_path = './train_valid_test_data/'+ personality +'_seq2seq_dataset_valid.npz'
# test_data_path = './train_valid_test_data/'+ personality +'_seq2seq_dataset_test.npz'


extro_data_test_path = './data/extro_seq2seq_dataset_test.npz'

intro_data_test_path = './data/intro_seq2seq_dataset_test.npz'

natural_data_test_path = './data/natural_seq2seq_dataset_test.npz'

dic = json.load(open("vocab_to_int.txt"))
word2idx = dic  # load word map

idx2word = {v: k for k, v in word2idx.items()}


test_set = Seq2SeqDataset(extro_data_test_path, word2idx)

test_dataloader = DataLoader(test_set,
                             batch_size=1,
                             collate_fn=seq2seq_collate_fn)


#%%
# buliding model

dof_num = 4
embed_dim = 100
learning_rate = 1e-4
encoder_hidden_dim = 128
decoder_hidden_dim = 128
model_save_folder = './saved_models/'


num_epochs = 5
epoch_init = 0
best_loss = 1000
epochs_since_improvement = 0

# load pretrained word embeddings
embedding = np.load('glove_pretrained_weights.npy', allow_pickle=True)
embedding = torch.from_numpy(embedding).float()

model = Seq2SeqModel(embed_dim=100,
                     vocab_size=3933,
                     dof=dof_num,
                     enc_dim=encoder_hidden_dim,
                     dec_dim=decoder_hidden_dim,
                     enc_layers=1,
                     dec_layers=1,
                     bidirectional=True,
                    #  dropout_prob=0.5,
                     pretrain_weight=None,
                     teacher_forcing_ratio=0)

# setting optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nopad_mse_loss(reduction='mean')

# load saved models
save_state_dict = torch.load(model_save_folder+'last_checkpoint_Seq2Seq_0121_232045')
# print(save_state_dict['model'])
model.load_state_dict(save_state_dict['model'])
optimizer.load_state_dict(save_state_dict['optim'])
epoch = save_state_dict['epoch']
valid_loss = save_state_dict['valid_loss']
epochs_since_improvement = save_state_dict['epochs_since_improvement']

# if use GPU, uncomment this
device = torch.device('cuda')
            # if torch.cuda.is_available()
            # else torch.device('cpu')

# if use GPU, comment this
# device = torch.device('cpu')

model = model.to(device)
criterion = criterion.to(device)

#%%
print('Evaluating on test set')
# os.environ['KMP_DUPLICATE_LIB_OK']='True's
output_list = []
attn_weights_list = []
in_seq_list = []
tgt_len_list = []
target_list = []
losses = AverageMeter()
model.eval()
# print(test_dataloader)
for idx, (in_seq, tgt_seq, in_len, tgt_len) in enumerate(test_dataloader):
    # print(len(tgt_seq[0]))

    with torch.no_grad():
        # calculate separately
        unpacked_out_enc, enc_mask = model.encoder(in_seq, in_len)
        # print(unpacked_out_enc)
        # print(enc_mask)
        model.decoder.set_teacher_forcing_ratio(0.0)
        # output, attn_weights = model.decoder(tgt_seq, unpacked_out_enc, enc_mask)
        output, attn_weights = model(in_seq, tgt_seq, in_len)
        # print(output)

    # save the results for subsequent analysis
    # print("out: "+str(output))
    # print("attn "+str(attn_weights))
    # print("outs: " + str(output.squeeze()))
    output_list.append(output.squeeze())
    attn_weights_list.append(attn_weights.squeeze())
    in_seq_list.append(in_seq.squeeze())
    # print(in_seq_list)
    tgt_len_list.append(tgt_len.squeeze())
    target_list.append(tgt_seq.squeeze())
    # print('.', end='')
# print(in_seq_list)
output_arr = np.array([i.numpy() for i in output_list])
target_arr = np.array([i.numpy() for i in target_list])
# print(target_arr)
# print(output_arr[0])


# the_sum = 0.0
#
# for a in range(len(output_arr)):
#     for b in range(len(output_arr[a])):
#         for x in output_arr[a][b]:
#             the_sum += x**2
#
#         for num in range(len(output_arr[a][b])):
#             if(output_arr[a][b][num] > 0):
#                 output_arr[a][b][num] = math.sqrt(1.0/the_sum * output_arr[a][b][num]**2)
#             else:
#                 output_arr[a][b][num] = -math.sqrt(1.0/the_sum * output_arr[a][b][num]**2)
#         the_sum = 0.0
        # print("here" + str(output_arr[a][b]))



# imaginary = []
# real1 = []
# real2 = []
# real3 = []
#
# target_imaginary = []
# target_real1 = []
# target_real2 = []
# target_real3 = []
# # for i in range(len(output_arr)):print(dec_outputs)
# i = 4
# for k in range(int(len(output_arr[i]))):
#     imaginary.append(output_arr[i][k][0])
#     target_imaginary.append(target_arr[i][k][0])
#
#     real1.append(output_arr[i][k][1])
#     target_real1.append(target_arr[i][k][1])
#
#     real2.append(output_arr[i][k][2])
#     target_real2.append(target_arr[i][k][2])
#
#     real3.append(output_arr[i][k][3])
#     target_real3.append(target_arr[i][k][3])
# fig, axs = plt.subplots(2,2)
# axs[0,0].plot(imaginary)
# axs[0,0].plot(target_imaginary)
# plt.setp(axs[0,0],ylim = (0.97,1.01))
# # axs[0,0].legend('real_imaginary','target_imaginary')
#
#
# axs[0,1].plot(real1)
# axs[0,1].plot(target_real1)
# plt.setp(axs[0,1],ylim = (-0.1,0.1))
# # axs[0,1].legend('real1','target_real1')
#
#
# axs[1,0].plot(real2)
# axs[1,0].plot(target_real2)
# plt.setp(axs[1,0],ylim = (-0.2,0.2))
# # axs[1,0].legend('real2','target_real2')
#
# axs[1,1].plot(real3)
# axs[1,1].plot(target_real3)
# plt.setp(axs[1,1],ylim = (-0.1,0.1))
# # axs[1,1].legend('real3','target_real3')
#
#
# plt.savefig('./performance')

# print(output_arr)
# print("target"+str(output_arr))

# %%
# calculate loss on each dim


err = 0.0
for i in range(len(output_list)):
    err += criterion(output_list[i].unsqueeze(0),
                    target_list[i].unsqueeze(0),
                    tgt_len_list[i].unsqueeze(0),
                    keep_dof=True)
err /= len(output_list)
#%%
std_out = 0.0
std_tgt = 0.0

for i in range(len(output_arr)):
    std_out += np.std(output_arr[i],axis=0)
    std_tgt += np.std(target_arr[i],axis=0)


std_out /= len(output_arr)
std_tgt /= len(output_arr)
# print(std_out)
#%%
pearson_corr = np.zeros(4)
for i in range(len(output_arr)):
    for j in range(4):
        pearson_corr[j] += np.corrcoef(output_arr[i][:,j], target_arr[i][:,j])[0, 1]

pearson_corr /= len(output_arr)

# %%
print('mse:', err.numpy())
print('corr:', pearson_corr)
print('std output:', std_out)
print('std target:', std_tgt)




#%%
#  visualize embedding weights




# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2)
# embed_weights = model.encoder.embedding.weight.data.numpy()
# #%%
# # fit_origin = tsne.fit_transform(embedding)  # [vocab_size,2]
# fit_result = tsne.fit_transform(embed_weights)  # [vocab_size,2]

#%%
# choose words randomly






# id_list = np.arange(3801)
# np.random.shuffle(id_list)
# id_list = id_list[:500]
# #%%
# plt.figure(figsize=(12,4))
# label = [idx2word[i] for i in range(len(idx2word))]
# # normalize to 0~1
# normalize_func = lambda x: (x - np.min(x, 0)) / (np.max(x, 0) - np.min(x, 0))
# temp_result = normalize_func(fit_result)
#
# search_words = ['milk', 'juice']
# for i in id_list:
#     clr = 'r' if label[i] in search_words else 'k'
#     texts = plt.annotate(label[i], xy=(temp_result[i, 0], temp_result[i, 1]),color = clr)
#     texts.set_alpha(0.5)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title('The 2D projection of orginal word embedding',fontsize=16)
#     plt.savefig('tsne_origin.jpg', bbox_inches = 'tight', dpi=150)
