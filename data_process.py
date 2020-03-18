#%%
import numpy as np
import os
from scipy.fftpack import dct

from utils import token_regularizer
from utils import pair_files
from utils import load_encode
from utils import word_split_rule

from collections import defaultdict
from torch.utils.data import random_split
from big_phoney import BigPhoney

import json
import random
import math
#%%

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        # print("Here")
        return False


def seq2seq_preprocess(transcript_path: str, motion_path: str) -> (np.ndarray, np.ndarray):
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
            # print(text)
            # print(len(text))


            if(start_time > len(motions)):
                continue

            if(float(line[1]) - prev_interval[1]/100.0 <= 0.5) and (len(prev_sentense) + len(text) <= 50):
                prev_sentense += text
                prev_interval[1] = end_time

            elif len(text) <= 5:
                continue

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
    # for i in transcripts:
    #     print(len(i))






    num_dof = 4
    targets = []
    for period in intervals[1:]:
        start_time = period[0]
        end_time = period[1]

        temp_motion = motions[start_time:end_time]
        # temp_motion = np.append(temp_motion,[[1.0,1.0,1.0,1.0]],axis = 0)
        # print(temp_motion)
        temp_motion = np.array(temp_motion)

        if not temp_motion.any():
            print(motion_path)
            continue
            # print(transcript_path)

            # print(start_time)
            # print(end_time)
            # print(temp_motion)
            # print('wow')
        else:
            # scaled_temp_motion = (temp_motion - temp_motion.mean())/temp_motion.std()
            # print(scaled_temp_motion)

            targets.append(temp_motion)


    inputs =  np.array(transcripts[1:])
    word_time_distribution = np.array(phon_split[1:])
    # print(inputs)
#     print(len(inputs))
#     print(targets)
    if(len(inputs) != len(targets)):
        print('wow')
    # print(len(inputs))
    # print(len(word_time_distribution))
    return inputs, targets, word_time_distribution



def make_dataset(process_function, paired_file_paths: list, ptype: str,
                 save_path: str) -> None:
    """"Calls preprocess methods and save processed data.

    Args:
        process_function: the function for proprecessing, it can be
            'dct_preprocess' or 'seq2seq_preprocess'.
        paired_file_paths: the output from 'pair_files' function. Its format is
            [NAME_NUM_[ine], actual text path, actual motion path]
        ptype: type of personality, can be '(e)xtroverted', '(i)ntroverted' and
            '(n)atural'.
        save_path: path to save the dataset
    """
    assert ptype in {
        'e', 'i', 'n'
    }, ('personality type should be "e","i" or "n", not %r' % ptype)

    all_input = []
    all_target = []
    all_word_distribution = []

    input_train = []
    input_valid = []
    input_test = []

    target_train = []
    target_valid = []
    target_test = []

    word_time_distribution_train = []
    word_time_distribution_valid  = []
    word_time_distribution_test = []
    # apply 'preprocess' method to for each text/motion pair
    for pair in paired_file_paths:
        # the last letter of file name means speaker's personality

        if pair[0][-1] == ptype:
            inputs, targets, word_time_distribution = process_function(pair[1], pair[2])
            # train_size = int(0.8 * len(inputs))
            # valid_size = int(0.1 * len(inputs))
            # test_size = len(inputs) - train_size - valid_size

            # print(len(inputs))

            # for i in range(len(inputs)):
            #     if (i < train_size):
            #         input_train_set.append(inputs[i])
            #         target_train_set.append(targets[i])
            #     elif (i < train_size+valid_size):
            #         input_valid_set.append(inputs[i])
            #         target_valid_set.append(targets[i])
            #     else:
            #         input_test_set.append(inputs[i])
            #         target_test_set.append(targets[i])


            all_input.append(inputs)
            all_target.append(targets)
            all_word_distribution.append(word_time_distribution)

    train_size = int(0.8 * len(all_input))
    valid_size = int(0.1 * len(all_input))
    test_size = len(all_input) - train_size - valid_size

    for i in range(0, train_size):
        for j in range(0, len(all_input[i])):
            input_train.append(all_input[i][j])
            target_train.append(all_target[i][j])
            word_time_distribution_train.append(all_word_distribution[i][j])


    for i in range(train_size, train_size + valid_size):
        for j in range(0, len(all_input[i])):
            input_valid.append(all_input[i][j])
            target_valid.append(all_target[i][j])
            word_time_distribution_valid.append(all_word_distribution[i][j])

    for i in range(train_size+valid_size, len(all_input)):
        for j in range(0, len(all_input[i])):
            input_test.append(all_input[i][j])
            target_test.append(all_target[i][j])
            word_time_distribution_test.append(all_word_distribution[i][j])


    sample_input_train = []
    sample_target_train = []
    sample_word_train = []
    for i in range(0,1):
        sample_input_train.append(input_train[i])
        sample_target_train.append(target_train[i])
        sample_word_train.append(word_time_distribution_train[i])
    np.savez(save_path+"_train_sample.npz", input=sample_input_train, target=sample_target_train,word_time_distribution = sample_word_train)


    np.savez(save_path+"_train.npz", input=input_train, target=target_train, word_time_distribution = word_time_distribution_train)
    np.savez(save_path+"_valid.npz", input=input_valid, target=target_valid, word_time_distribution = word_time_distribution_valid)
    np.savez(save_path+"_test.npz", input=input_test, target=target_test, word_time_distribution = word_time_distribution_test)

#%%
# this code is to make dataset for DCT baseline model
if True:
    the_path = '/afs/inf.ed.ac.uk/group/cstr/projects/galatea/d02'
    # print (os.path.exists('/afs/inf.ed.ac.uk/group/cstr/projects/galatea'))
    data_path_motion = the_path + '/Recordings_October_2014/DOF-hiroshi/'
    # print(data_path_motion)
    data_path_text = the_path + '/Recordings_October_2014/Transcriptions/transcriptions_word_tables/'

    paired_file_paths = pair_files(data_path_text, '', data_path_motion,'.rov')
    # 所有文件



total_words = defaultdict(dict)
for i in range(len(paired_file_paths)):
    with open(paired_file_paths[i][1],'r') as f:
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

# for i in vocab_to_int.keys():
#     print(i, vocab_to_int[i])

# f = open("vocab_to_int.txt","w")
# f.write( str(vocab_to_int) )
# f.close()


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

the_path = '/afs/inf.ed.ac.uk/group/cstr/projects/galatea/d02'
data_path_motion = the_path + '/Recordings_October_2014/DOF-hiroshi/'
data_path_text = the_path + '/Recordings_October_2014/Transcriptions/transcriptions_phrase_tables/'

paired_file_paths = pair_files(data_path_text, '.TABLE', data_path_motion,
                                '.axa')
# word2idx = vocab_to_int
word2idx = dic

# make three kinds of dataset
make_dataset(seq2seq_preprocess, paired_file_paths, 'e',
                './data/extro_seq2seq_dataset')
print('extroverted data finished')
make_dataset(seq2seq_preprocess, paired_file_paths, 'i',
                './data/intro_seq2seq_dataset')
print('introverted data finished')
make_dataset(seq2seq_preprocess, paired_file_paths, 'n',
                './data/natural_seq2seq_dataset')
print('natural data finished')
