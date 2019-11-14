import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TextMotionDataset(Dataset):
    """Dataset class for baseline DCT model

    input is [word_{t-1}, word_{t}, word_{t+1}]
    target is DCT features, length of 20freq * 3dim
    """
    def __init__(self,data_path):
        # load dataset in npz format
        assert os.path.exists(data_path)
        dataset = np.load(data_path, allow_pickle = True)

        self.input = dataset['input']
        self.target = dataset['target']

        self.size = len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]

    def __len__(self):
        return self.size


def seq2seq_collate_fn(data: list, down_sample: int = 1, lookback: int = 4):
    """Dataset class for Seq2Seq model

    Args:
        down_sample: only used for test program. Must be reset to 1 when traning
        lookback: how many previous steps used as input. Minimum is 1.

    Returns:
        in_seq: sentences accepted by encoder, shape
            [batch_size, max_seq_len_enc]
        tgt_seq: motion sequences accepted by decoder, shape
            [batch_size, max_seq_len, lookback, num_dof]
        new_target: target used to calculate loss, shape
            [batch_size, max_seq_len, num_dof]
        in_len: record length of each in_seq sample, shape [batch_size]
        tgt_len: record length of each tgt_seq sample, shape [batch_size]
    """
    # data has two columns, column 1st is input word, 2nd is target motion
    in_seq = [torch.LongTensor(pair[0]) for pair in data]
    target = [torch.Tensor(pair[1][::down_sample]) for pair in data]

    # sort input sequence from the longest to shortest for LSTM
    temp_in_seq = [(idx, seq) for idx, seq in enumerate(in_seq)]
    temp_in_seq.sort(key=lambda x: len(x[1]), reverse=True)
    sort_order = [idx for idx, _ in temp_in_seq]

    in_seq = [seq for _, seq in temp_in_seq]
    target = [target[idx] for idx in sort_order] # target seq follow order of input

    # use previous 'lookback' interval motions to predict the next motion
    tgt_seq, new_target = [], []
    for motion_seq in target:
        temp_tgt_seq, temp_target = [], []
        for idx in range(len(motion_seq) - lookback):
            temp_tgt_seq.append(motion_seq[idx:idx + lookback])
            temp_target.append(motion_seq[idx + lookback])
        tgt_seq.append(torch.stack(temp_tgt_seq))
        new_target.append(torch.stack(temp_target))

    # record length of each input for padding
    # use 'Byte', 'Short' just to save some memory
    in_len = [torch.ByteTensor([len(line)]) for line in in_seq]
    tgt_len = [torch.ShortTensor([len(motion)]) for motion in tgt_seq]
    in_len = torch.cat(in_len)
    tgt_len = torch.cat(tgt_len)

    # pad short sequences
    in_seq = pad_sequence(in_seq, batch_first=True)
    tgt_seq = pad_sequence(tgt_seq, batch_first=True)
    new_target = pad_sequence(new_target, batch_first=True)

    return in_seq, tgt_seq, new_target, in_len, tgt_len


class Seq2SeqDataset(Dataset):
    """Dataset class for Seq2Seq model

    input is sentence (variable length)
    target is head motion trajectory (variable length)   
    """
    def __init__(self, data_path, word2idx:dict):

        assert os.path.exists(data_path)
        dataset = np.load(data_path, allow_pickle = True)

        self.input = dataset['input']
        self.target = dataset['target']
        self.word2idx = word2idx
        self.idx2word = {v:k for k, v in word2idx.items()}

        self.size = len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]

    def __len__(self):
        return self.size
