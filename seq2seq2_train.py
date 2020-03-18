import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from models3 import Seq2SeqAttentionSharedEmbedding
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import json
from dataset import Seq2SeqDataset, seq2seq_collate_fn, BySequenceLengthSampler

if __name__ == '__main__':

    pad_len = 30
    batch_size = 50
    emb_dim = 100
    dim = 128
    vocab_size = 3933
    from sklearn.model_selection import ShuffleSplit

    # split_ratio = 0.05
    # sss = ShuffleSplit(n_splits=1, test_size=split_ratio, random_state=0)
    # df = pd.read_csv('data_6_remove_dup_train.csv')
    # X, y, tag = df['source'], df['target'], df['tag']
    # temp_up = 10000
    # X = X[:temp_up]
    # y = y[:temp_up]
    # tag = tag[:temp_up]
    # train_index, dev_index = next(sss.split(tag))
    #
    # X_train = [X[i] for i in train_index]
    # y_train = [y[i] for i in train_index]
    # tag_train = [tag[i] for i in train_index]
    # X_dev = [X[i] for i in dev_index]
    # y_dev = [y[i] for i in dev_index]
    # tag_dev = [tag[i] for i in dev_index]
    # del df, X, y, tag

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


    # training_set = EmotionDataLoaderStart(X_train, y_train, tag_train, pad_len, word2id)
    train_loader = DataLoader(train_set,
                              batch_size,
                              shuffle=True,
                              collate_fn=seq2seq_collate_fn)

    valid_loader = DataLoader(valid_set,
                              batch_size,
                              shuffle=True,
                              collate_fn=seq2seq_collate_fn)

    test_dataloader = DataLoader(test_set,
                             batch_size=1,
                             collate_fn=seq2seq_collate_fn)

    # test_set = EmotionDataLoaderStart(X_dev, y_dev, tag_dev, pad_len, word2id)
    # test_loader = DataLoader(test_set, batch_size=batch_size)
    # loader = iter(train_loader)
    # next(loader)

    model = Seq2SeqAttentionSharedEmbedding(
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        src_hidden_dim=dim,
        trg_hidden_dim=dim,
        ctx_hidden_dim=dim,
        attention_mode='dot',
        batch_size=batch_size,
        bidirectional=False,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
    )
    model.cuda()
    # model = nn.DataParallel(model).cuda()

    # model_path = 'checkpoint/new_simple_foo_epoch_9.model'
    # model.load_state_dict(torch.load(
    #     model_path
    # ))


    weight_mask = torch.ones(vocab_size).cuda()
    weight_mask[word2idx['<PAD>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # es = EarlyStop(2)
    for epoch in range(100):
        print('Training on epoch=%d -------------------------' % (epoch))
        train_loss_sum = 0
        for idx, (src, trg, in_len, tgt_len) in tqdm(enumerate(train_loader), total=int(len(train_set)/batch_size)):
            # print('i=%d: ' % (i))
            # print(src)
            # print(trg)
            # print("here")
            decoder_logit = model(Variable(src).cuda(), Variable(trg).cuda())
            optimizer.zero_grad()
            trg = torch.cat((torch.index_select(trg, 1, torch.LongTensor(list(range(1, pad_len)))),
                             torch.LongTensor(np.zeros([trg.shape[0], 1]))), dim=1)
            loss = loss_criterion(
                decoder_logit.contiguous().view(-1, vocab_size),
                Variable(trg).view(-1).cuda()
            )
            train_loss_sum += loss.data[0]
            loss.backward()
            optimizer.step()
            del loss, decoder_logit

        print("Training Loss", train_loss_sum)

        # Evaluate
        test_loss_sum = 0
        print("Evaluating:")
        for src_valid, trg_valid in tqdm(enumerate(valid_loader), total=int(len(valid_set)/batch_size)):

            valid_logit = model(Variable(src_valid, volatile=True).cuda(),
                               Variable(trg_valid, volatile=True).cuda())
            trg_valid = torch.cat((torch.index_select(trg_valid, 1, torch.LongTensor(list(range(1, pad_len)))),
                             torch.LongTensor(np.zeros([trg_valid.shape[0], 1]))), dim=1)
            valid_loss = loss_criterion(
                valid_logit.contiguous().view(-1, vocab_size),
                Variable(trg_valid).view(-1).cuda()
            )
            valid_loss_sum += valid_loss.data[0]
            del valid_loss, alid_logit

        print("Evaluation Loss", valid_loss_sum)
        # es.new_loss(valid_loss_sum)
        # if es.if_stop():
        #     print('Start over fitting')
        #     break
        # Save Model
        torch.save(
            model.state_dict(),
            open(os.path.join(
                'checkpoint',
                'new_simple_start' + '_epoch_%d' % (epoch) + '.model'), 'wb'
            )
        )
