#%%
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from models import Seq2SeqModel, nopad_mse_loss
from dataset import Seq2SeqDataset, seq2seq_collate_fn, BySequenceLengthSampler
from utils import now_time
from utils import AverageMeter
from utils import load_encode


import os
import json




seed = 42
torch.manual_seed(seed) # fix a seed for reproduce

#%%
# dataset setting
batch_size = 50

extro_data_train_path = './data/extro_seq2seq_dataset_train.npz'
extro_data_valid_path = './data/extro_seq2seq_dataset_valid.npz'
extro_data_test_path = './data/extro_seq2seq_dataset_test.npz'

intro_data_train_path = './data/intro_seq2seq_dataset_train.npz'
intro_data_valid_path = './data/intro_seq2seq_dataset_valid.npz'
intro_data_test_path = './data/intro_seq2seq_dataset_test.npz'

natural_data_train_path = './data/natural_seq2seq_dataset_train.npz'
natural_data_valid_path = './data/natural_seq2seq_dataset_valid.npz'
natural_data_test_path = './data/natural_seq2seq_dataset_test.npz'
# data_path = './data/extro_seq2seq_dataset.npz'

dic = json.load(open("vocab_to_int.txt"))
word2idx = dic  # load word map

# torch.cuda.empty_cache()

train_set = Seq2SeqDataset(extro_data_train_path, word2idx)
valid_set = Seq2SeqDataset(extro_data_valid_path, word2idx)
test_set = Seq2SeqDataset(extro_data_test_path, word2idx)



# print(len(valid_set))
# print(len(test_set))

# input_test_set = []
# target_test_set = []
# for i in range(0, len(test_set)):
#     input_test_set.append(test_set[i][0])
#     target_test_set.append(test_set[i][1])
#
# np.savez("test_data.npz", input=input_test_set, target=target_test_set)

#
# bucket_boundaries = [10,20,30,40,50,100,150]
# batch_sizes=32
# sampler = BySequenceLengthSampler(train_set,bucket_boundaries, batch_sizes)
# # print(sampler)
# train_dataloader = DataLoader(train_set, batch_size=1,
#                         batch_sampler=sampler,
#                         num_workers=0,
#                         drop_last=False, pin_memory=False,
#                         collate_fn=seq2seq_collate_fn)



train_dataloader = DataLoader(train_set,
                              batch_size,
                              shuffle=True,
                              collate_fn=seq2seq_collate_fn)
valid_dataloader = DataLoader(valid_set,
                              batch_size,
                              shuffle=True,
                              collate_fn=seq2seq_collate_fn)
test_dataloader = DataLoader(test_set,
                             batch_size=1,
                             collate_fn=seq2seq_collate_fn)


del train_set
del valid_set
del test_set
#%%
# building model
dof_num = 4
embed_dim = 100
learning_rate = 1e-4
encoder_hidden_dim = 128
decoder_hidden_dim = 128
model_save_folder = './saved_models/'
load_checkpoint_name = None

num_epochs = 50
epoch_init = 0
best_loss = 1000
epochs_since_improvement = 0
clip_norm = 500

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
                     pretrain_weight=None, # embedding,
                     teacher_forcing_ratio=0.1)

# setting optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nopad_mse_loss(reduction='mean')

if load_checkpoint_name is not None:
    # load saved models
    save_state_dict = torch.load(model_save_folder + load_checkpoint_name)
    model.load_state_dict(save_state_dict['model'])
    optimizer.load_state_dict(save_state_dict['optim'])
    epoch_init = save_state_dict['epoch']
    best_loss = save_state_dict['valid_loss']
    epochs_since_improvement = save_state_dict['epochs_since_improvement']

# if use GPU, uncomment this
# print(torch.cuda.is_available())
device = torch.device('cuda')
#             if torch.cuda.is_available()
#             else torch.device('cpu')

# if use GPU, comment this
# device = torch.device('cpu')

model = model.to(device)
criterion = criterion.to(device)

#%%
validate_loss = []
training_loss = []
losses = AverageMeter()
training_date = now_time()
print('training time: ', training_date)

for epoch in range(num_epochs):
    # start training
    losses.reset()
    # print(losses)
    model.train()
    model.decoder.set_teacher_forcing_ratio(0)

    for idx, (in_seq, tgt_seq, in_len, tgt_len) in enumerate(train_dataloader):
        # print(in_seq)
        # print(tgt_seq)
        in_seq = in_seq.to(device)
        tgt_seq = tgt_seq.to(device)
        # target = target.to(device)
        in_len = in_len.to(device)
        tgt_len = tgt_len.to(device)

        # if error happens when backprop, uncomment this to debug
        # with torch.autograd.set_detect_anomaly(True):
        output, attn_weights = model(in_seq, tgt_seq, in_len)
        # print(attn_weights)

        # calc loss and back-prop
        loss = criterion(output, tgt_seq, tgt_len)
        # print(tgt_seq)
        # print('otput::')
        # print(in_seq)
        # print(output)
        # print(target.size())
        loss.backward()

        # if need to avoid gradient explosion, uncomment this
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        optimizer.zero_grad()

        #  print training progress
        losses.update(loss.item(), len(in_seq))
        if idx % 5 == 0:
            print('Train Epoch:\t{} [{}/{}]\tLoss: {:.6f}'.format(
                epoch + epoch_init, idx, len(train_dataloader), losses.avg))

    training_loss.append(losses.avg)

    # # start validation
    losses.reset()
    model.train()
    model.decoder.set_teacher_forcing_ratio(0)
    for idx, (in_seq, tgt_seq, in_len, tgt_len) in enumerate(valid_dataloader):

        with torch.no_grad():
            output, attn_weights = model(in_seq, tgt_seq, in_len)
                # print(attn_weights)
            # print(in_seq)
            print(output)
                # print('!!!!!'+str(target))
            loss = criterion(output, tgt_seq, tgt_len)

        #  print training progress
        del output
        del attn_weights

        losses.update(loss.item(), len(in_seq))
        del loss

        if idx % 5 == 0:
            print('Validate Epoch:\t{} [{}/{}]\tLoss: {:.6f}'.format(
                epoch + epoch_init, idx, len(valid_dataloader), losses.avg))

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
        'epoch': epoch + epoch_init,
        'epochs_since_improvement': epochs_since_improvement,
        'valid_loss': losses.avg,
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }
    torch.save(save_state_dict,
               model_save_folder + 'last_checkpoint_Seq2Seq_' + training_date)
    if is_best:
        torch.save(
            save_state_dict,
            model_save_folder + 'best_checkpoint_Seq2Seq_' + training_date)

#%%
# save the loss curve
# np.savetxt('./result/Seq2Seq_loss_' + training_date + '.txt',
#            (training_loss, validate_loss))

np.savetxt('./result/Seq2Seq_loss_' + training_date + '.txt',
           (training_loss, validate_loss))
print('training time', training_date)
