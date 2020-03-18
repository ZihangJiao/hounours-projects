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
from utils import token_regularizer
from utils import word_split_rule
from big_phoney import BigPhoney

from collections import defaultdict

import pylab
import time
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns

import math

import os
import json

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        # print("Here")
        return False


seed = 42
torch.manual_seed(seed)
#%%
# dataset setting
batch_size = 11

data_path_text = '/afs/inf.ed.ac.uk/group/cstr/projects/galatea/d02/Recordings_October_2014/Transcriptions/transcriptions_phrase_tables/Brian_Adam/Adam_02_e.TABLE'
data_path_motion = '/afs/inf.ed.ac.uk/group/cstr/projects/galatea/d02/Recordings_October_2014/DOF-hiroshi/Adam/Head/Normalised/Adam_02_e.axa'


def seq2seq_preprocess(transcript_path: str, motion_path: str) -> None:
    transcripts = []
    intervals = []
    whole_sentence = []
    phoney = BigPhoney()

    prev_sentense = ''
    prev_interval = [0,0]

    motions = np.loadtxt(motion_path, usecols=range(4), skiprows=17,
                         dtype='float')

    phon_split = []
    with open(transcript_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            # print(line)
            if not is_number(line[1]) or not is_number(line[2]):
                continue
            start_time = int(float(line[1]) * 100)
            end_time = int(float(line[2])*100)
            text = line[3:]
            # print(len(text))


            if(start_time > len(motions)):
                continue

            if(float(line[1]) - prev_interval[1]/100.0 <= 0.5) and (len(prev_sentense) + len(text) <= 50):
                prev_sentense += text
                prev_interval[1] = end_time

            elif len(text) <= 5:
                continue
#                 去掉过短句子
            else:
                temp_phon = []
                temp_split = []
                for i in prev_sentense:
                    temp_phon.append(phoney.phonize(i))
                    # print(i)
                    # print(phoney.phonize(i))
                # print(prev_interval[0])
                splited_phon = word_split_rule(temp_phon)
                the_sum = math.fsum(splited_phon)
                # phon_split.append(word_split_rule(temp_phon))
                time_distance = prev_interval[1] - prev_interval[0]
                # print(phon_split)
                # print(len(splited_phon))

                for j in range(len(splited_phon)):
                    time_float = splited_phon[j] / the_sum * time_distance
                    if (time_float - int(time_float) >= 0.5):
                        temp_split.append(int(time_float) + 1)
                    else:
                        temp_split.append(int(time_float))

                if(len(temp_split)) != 0:
                    temp_split[-1] += int(abs(time_distance - math.fsum(temp_split)))

                phon_split.append(temp_split)

                # print(temp_phon)
                # print(convert_to_ints(prev_sentense))
                transcripts.append(convert_to_ints(prev_sentense))
                intervals.append(prev_interval)
                prev_sentense = text
                prev_interval = [start_time,end_time]

    num_dof = 4
    targets = []
    for period in intervals[1:]:
        start_time = period[0]
        end_time = period[1]

        temp_motion = motions[start_time:end_time]
        temp_motion = np.array(temp_motion)

        if not temp_motion.any():
            print(motion_path)
            continue
        else:
            targets.append(temp_motion)

    inputs =  np.array(transcripts[1:])
    word_time_distribution = np.array(phon_split[1:])
    np.savez("head.npz", input=inputs, target=targets, word_time_distribution=word_time_distribution)




total_words = defaultdict(dict)
with open(data_path_text,'r') as f:
    for line in f.readlines():
            # print(line)
        line = line.strip().split()
            # print(line)
        if(len(line) == 1):
            words = (line[0].split(','))[3]
        else:
            words = token_regularizer(line[3])
            # print(words)
        for w in words:
            if(w not in total_words):
                total_words[w] = 1
            else:
                total_words[w] += 1
# print(total_words)

##所有文件中的词个数

vocab_to_int = {}
value = 0
vocab_to_int["<PAD>"] = value
value += 1
for word in total_words.keys():
    # print(word)
    if '-' not in word:
        vocab_to_int[word] = value
        value += 1
codes = ["<UNK>","<EOS>","<GO>","<Stammer>","<Long>"]

for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

int_to_vocab = {}
for word,value in vocab_to_int.items():
    int_to_vocab[value] = word



json.dump(vocab_to_int, open("vocab_to_int.txt",'w'))

dic = json.load(open("vocab_to_int.txt"))

def convert_to_ints(text):
    ints = []
    ints.append(vocab_to_int["<GO>"])
    for sentense in text:
        sentense_ints = 0
        for word in sentense.split():
            if word in vocab_to_int:
                sentense_ints = vocab_to_int[word]
            elif word[len(word)-1] == '-':
                sentense_ints = vocab_to_int["<Stammer>"]
                ints.append(sentense_ints)
            elif '-' in word:
                sentense_ints = vocab_to_int["<Long>"]
                ints.append(sentense_ints)
            elif "'" in word:
                # print(word)
                words = token_regularizer(word)
                for split_words in words:
                    if word[len(word)-1] == '-':
                        sentense_ints = vocab_to_int["<Stammer>"]
                    elif '-' in word:
                        sentense_ints = vocab_to_int["<Long>"]
                    if split_words in vocab_to_int:
                        ints.append(vocab_to_int[split_words])
                    else:
                        sentense_ints = vocab_to_int["<UNK>"]
            else:
                sentense_ints = vocab_to_int["<UNK>"]
                # print(word)
#         if eos:
#             sentense_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentense_ints)
    ints.append(vocab_to_int["<EOS>"])
    return ints

# seq2seq_preprocess(data_path_text,data_path_motion)
print("preprocess finished")
extro_data_test_path = 'head.npz'
word2idx = dic
test_set = Seq2SeqDataset(extro_data_test_path, word2idx)

test_dataloader = DataLoader(test_set,
                             batch_size=1,
                             collate_fn=seq2seq_collate_fn)


#%%
# buliding model

dof_num = 4
embed_dim = 100
learning_rate = 1e-3
encoder_hidden_dim = 32
decoder_hidden_dim = 32
model_save_folder = './saved_models/'


num_epochs = 50
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
                     pretrain_weight=None)

# setting optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nopad_mse_loss(reduction='mean')

# load saved models
save_state_dict = torch.load(model_save_folder+'last_checkpoint_Seq2Seq_0309')
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
word_time_distribution_list = []
losses = AverageMeter()
model.eval()
# print(test_dataloader)
for idx, (in_seq, tgt_seq,word_time_distribution, in_len, tgt_len) in enumerate(test_dataloader):
    # print(len(tgt_seq[0]))

    with torch.no_grad():
        # calculate separately
        # unpacked_out_enc, enc_mask = model.encoder(in_seq, in_len,word_time_distribution)
        # print(unpacked_out_enc)
        # print(enc_mask)
        output = model(in_seq, tgt_seq,word_time_distribution)
        # model.decoder.set_teacher_forcing_ratio(0.0)
        # output, attn_weights = model.decoder(tgt_seq, unpacked_out_enc, enc_mask)
        # output, attn_weights = model(in_seq, tgt_seq, in_len)
        # print(output)

    # save the results for subsequent analysis
    # print("out: "+str(output))
    # print("attn "+str(attn_weights))
    # print("outs: " + str(output.squeeze()))
    output_list.append(output.squeeze())
    # attn_weights_list.append(attn_weights.squeeze())
    in_seq_list.append(in_seq.squeeze())
    # print(in_seq_list)
    tgt_len_list.append(tgt_len.squeeze())
    target_list.append(tgt_seq.squeeze())
    word_time_distribution_list.append(word_time_distribution.squeeze())
    # print('.', end='')
# print(in_seq_list)
output_arr = [i for i in output_list]
target_arr = [i for i in target_list]
word_time_distribution_arr = [i for i in word_time_distribution_list]


# # print(word_time_distribution_arr)
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
# # print(len(word_time_distribution_arr[0]))
# # print(int(len(output_arr[0])))
# i = 0
# count = 0
# the_sum = word_time_distribution_arr[i][count]
# max_smooth_counter = 50
# smooth_counter = 50
# for k in range(int(len(output_arr[i]))):
#     if(k == the_sum):
#         count += 1
#         if(count < len(word_time_distribution_arr[i])):
#             the_sum += word_time_distribution_arr[i][count]
#         smooth_counter = 1
#     if(imaginary):
#         imaginary.append(output_arr[i][k][0] * smooth_counter/max_smooth_counter + imaginary[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter )
#         target_imaginary.append(target_arr[i][k][0])
#
#         real1.append(output_arr[i][k][1] * smooth_counter/max_smooth_counter + real1[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter )
#         target_real1.append(target_arr[i][k][1])
#
#         real2.append(output_arr[i][k][2] * smooth_counter/max_smooth_counter + real2[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter )
#         target_real2.append(target_arr[i][k][2])
#
#         real3.append(output_arr[i][k][3] * smooth_counter/max_smooth_counter + real3[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter )
#         target_real3.append(target_arr[i][k][3])
#     else:
#         imaginary.append(output_arr[i][k][0])
#         target_imaginary.append(target_arr[i][k][0])
#
#         real1.append(output_arr[i][k][1])
#         target_real1.append(target_arr[i][k][1])
#
#         real2.append(output_arr[i][k][2])
#         target_real2.append(target_arr[i][k][2])
#
#         real3.append(output_arr[i][k][3])
#         target_real3.append(target_arr[i][k][3])
#
#
# fig, axs = plt.subplots(2,2)
# axs[0,0].plot(imaginary)
# axs[0,0].plot(target_imaginary)
# # plt.setp(axs[0,0],ylim = (0.97,1.01))
# # axs[0,0].legend('real_imaginary','target_imaginary')
#
#
# axs[0,1].plot(real1)
# axs[0,1].plot(target_real1)
# # plt.setp(axs[0,1],ylim = (-0.1,0.1))
# # axs[0,1].legend('real1','target_real1')
#
#
# axs[1,0].plot(real2)
# axs[1,0].plot(target_real2)
# # plt.setp(axs[1,0],ylim = (-0.2,0.2))
# # axs[1,0].legend('real2','target_real2')
#
# axs[1,1].plot(real3)
# axs[1,1].plot(target_real3)
# # plt.setp(axs[1,1],ylim = (-0.1,0.1))
# # axs[1,1].legend('real3','target_real3')
#
#
# plt.savefig('./performance_head')





f = open("whole_head_motion.txt","w")
for i in range(len(target_arr)):
    imaginary = []
    real1 = []
    real2 = []
    real3 = []

    count = 0
    the_sum = word_time_distribution_arr[i][count]
    max_smooth_counter = 50
    smooth_counter = 50
    for k in range(int(len(output_arr[i]))):
        if(k == the_sum):
            count += 1
            if(count < len(word_time_distribution_arr[i])):
                the_sum += word_time_distribution_arr[i][count]
            smooth_counter = 1
        if(imaginary):
            img = output_arr[i][k][0] * smooth_counter/max_smooth_counter + imaginary[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter
            r1 = output_arr[i][k][1] * smooth_counter/max_smooth_counter + real1[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter
            r2 = output_arr[i][k][2] * smooth_counter/max_smooth_counter + real2[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter
            r3 = output_arr[i][k][3] * smooth_counter/max_smooth_counter + real3[-1] * (max_smooth_counter - smooth_counter)/max_smooth_counter
            imaginary.append(img)
            real1.append(r1)
            real2.append(r2)
            real3.append(r3)
            f.write(str(img.data.tolist()) + " " + str(r1.data.tolist()) + " " + str(r2.data.tolist()) + " " + str(r3.data.tolist()) + '\n')

        else:
            imaginary.append(output_arr[i][k][0])
            real1.append(output_arr[i][k][1])
            real2.append(output_arr[i][k][2])
            real3.append(output_arr[i][k][3])
            f.write(str(output_arr[i][k][0].data.tolist()) + " " + str(output_arr[i][k][1].data.tolist()) + " " + str(output_arr[i][k][2].data.tolist()) + " " + str(output_arr[i][k][3].data.tolist()) + '\n')


# f = open("whole_head_motion.txt","w")
# for i in range(len(target_arr)):
#     for j in range(len(target_arr[i])):
#         # if(output_arr[i][j][0] < 0.99):
#         #     output_arr[i][j][0] = 0.99
#         f.write(str(output_arr[i][j][0].data.tolist()) + " " + str(output_arr[i][j][1].data.tolist()) + " " + str(output_arr[i][j][2].data.tolist()) + " " + str(output_arr[i][j][3].data.tolist()) + '\n')


f = open("whole_head_motion_target.txt","w")
for i in range(len(target_arr)):
    for j in range(len(target_arr[i])):
        # if(output_arr[i][j][0] < 0.99):
        #     output_arr[i][j][0] = 0.99
        f.write(str(target_arr[i][j][0].data.tolist()) + " " + str(target_arr[i][j][1].data.tolist()) + " " + str(target_arr[i][j][2].data.tolist()) + " " + str(target_arr[i][j][3].data.tolist()) + '\n')
