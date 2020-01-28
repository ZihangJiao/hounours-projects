#%%
import numpy as np
import os
from scipy.fftpack import dct

from utils import token_regularizer
from utils import pair_files
from utils import load_encode

from collections import defaultdict
from torch.utils.data import random_split

import json
import random

#%%
def dct_preprocess(transcript_path: str, motion_path: str,
                   dictionary: dict) -> (np.ndarray, np.ndarray):
    """Load words and corresponding motion segments for one conversation

    Args:
        transcript_path: path to the transcription file
        motion_path: path to the motion file
        dictionary: wordmap that convert word to its index

    Returns:
        inputs: as the input of the network, shape [num_words, 3]
        targets: DCT features, corresponding motions, shape [num_words, 3*10]
    """
    # load transcript and motion files
    try:
        transcripts = np.loadtxt(transcript_path, usecols=3, dtype='str')
        # extract word and corresponding intervals in the form of
        # [word, start, end] and have the shape [num_word, 3]
        intervals = np.loadtxt(transcript_path,
                               usecols=range(4, 6),
                               dtype='int')
    except:
        # because some files use csv format
        transcripts = np.loadtxt(transcript_path,
                                 usecols=3,
                                 delimiter=',',
                                 dtype='str')
        intervals = np.loadtxt(transcript_path,
                               usecols=range(4, 6),
                               delimiter=',',
                               dtype='int')

    # motions has the shape [#intervals, 3(euler angles)]
    motions = np.loadtxt(motion_path,
                         usecols=range(3),
                         skiprows=17,
                         dtype='float')


    #### Text part
    # encode the transcripts based on dictionary.
    # Some words need to be split, so do intervals.
    new_transcripts = []
    new_intervals = []

    for idx, token in enumerate(transcripts):
        start_time = intervals[idx][0]
        length = intervals[idx][1] - intervals[idx][0]

        # in some conversations, transcript intervals are more than
        # that of motion, ignore subsequent words that out of range
        if start_time >= len(motions):
            break

        # split each word(token) and the intervals
        words = token_regularizer(token)
        for i, w in enumerate(words):
            # append words, use <unk> (0) for unknown ones
            if w in dictionary:
                new_transcripts.append(dictionary[w])
            else:
                new_transcripts.append(dictionary['<unk>'])

            # calculate and append the fraction of interval
            s = int(start_time + i / len(words) * length)
            e = int(start_time + (i + 1) / len(words) * length)
            new_intervals.append([s, e])

    # make [w_{t-1}, w_t, w_{t+1}] as one entry of network input
    inputs = np.zeros((len(new_transcripts), 3), dtype=np.int)
    inputs[1:, 0] = new_transcripts[:-1]
    inputs[:, 1] = new_transcripts
    inputs[:-1, 2] = new_transcripts[1:]

    # add <bos> and <eos> at the begin/end of word sequence
    inputs[0, 0] = dictionary['<bos>']
    inputs[-1, -1] = dictionary['<eos>']

    #### Motion part
    # process the motion data based on the intervals
    window_size = 50
    num_dof = 3
    targets = []

    for period in new_intervals:
        # keep moving the window until the end
        start_time = period[0]
        end_time = min(period[0] + window_size, len(motions))

        # make a discrete 'signal' with fixed size
        temp_motion = np.zeros((window_size, num_dof))
        temp_motion[:end_time - start_time] = motions[start_time:end_time]

        # apply DCT on it, keep the top 10 frequencies, filter out high
        # frequencies.  The result shape is [num_dof, 10]
        result = [dct(m, norm='ortho', axis=0)[:10] for m in temp_motion.T]
        # reshape to vector of length 30
        result = np.array(result).flatten()
        targets.append(result)
    targets = np.array(targets)

    return inputs, targets


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        # print("Here")
        return False

#%%
# def seq2seq_preprocess(transcript_path: str, motion_path: str,
#                         dictionary: dict) -> (np.ndarray, np.ndarray):
#     """Load sentences and corresponding motions for one conversation
#
#     Same as the DCT process function. Encode the sentences and keep the
#     original motion time series rather than performing DCT transformations.
#
#     Args:
#         transcript_path: path to the transcription file
#         motion_path: path to the motion file
#         dictionary: wordmap that convert word to its index
#
#     Returns:
#         inputs: as the input of the network,
#             shape [num_sentences, variable length of sentence]
#         targets: time series of corresponding motions,
#             shape [num_sentences, variable length of motion]
#     """
#     transcripts = []
#     intervals = []
#
#
#     # load transcript and motion files
#     with open(transcript_path, 'r') as f:
#         for line in f.readlines():
#             line = line.strip().split()
#
#             if not is_number(line[1]) or not is_number(line[2]):
#                 continue  ### there're wrong formats in some lines
#
#             # each interval is 10ms, time unit in this file is 'second'
#             # so *100 to make it a integer
#             start_time = int(float(line[1]) * 100)
#             end_time = int(float(line[2]) * 100)
#             text = line[3:]
#
#
#             # discard too short sequences
#             if len(text) < 5 or end_time - start_time < 20:
#                 continue
#             transcripts.append(text)
#             intervals.append([start_time, end_time])
#
#     motions = np.loadtxt(motion_path,
#                          usecols=range(4),
#                          skiprows=17,
#                          dtype='float')
#
#     #### Text part
#     # Encode the transcripts based on preprocessed dictionary
#     new_transcripts = []
#     new_intervals = []
#
#     for idx, line in enumerate(transcripts):
#         # check if the end interval is out of motion range
#
#         if intervals[idx][1] > len(motions):
#             break
#         new_intervals.append(intervals[idx])
#
#         new_sentence = []
#         new_sentence.append(dictionary['<bos>'])  # begin of sentence
#         for token in line:
#             # Some words need to be split, but intervals not.
#             words = token_regularizer(token)
#
#             # there may be several words after being splited
#             for w in words:
#                 if w in dictionary:
#                     new_sentence.append(dictionary[w])
#                 else:
#                     new_sentence.append(dictionary['<unk>'])
#
#         new_sentence.append(dictionary['<eos>'])  # end of sentence
#         new_transcripts.append(new_sentence)
#     inputs = np.array(new_transcripts)
#
#     #### Motion part
#     # Extract the motion data based on intervals
#     num_dof = 4
#     targets = []
#     for period in new_intervals:
#         start_time = period[0]
#         end_time = period[1]
#
#         temp_motion = motions[start_time:end_time]
#         temp_motion = np.array(temp_motion)
#
#         targets.append(temp_motion)
#     print(len(inputs))
#     return inputs, targets



def seq2seq_preprocess(transcript_path: str, motion_path: str,
                       dictionary: dict) -> (np.ndarray, np.ndarray):
    transcripts = []
    intervals = []

    prev_sentense = ''
    prev_interval = [0,0]

    motions = np.loadtxt(motion_path, usecols=range(4), skiprows=17,
                         dtype='float')


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
    # print(inputs)
#     print(len(inputs))
#     print(targets)
    if(len(inputs) != len(targets)):
        print('wow')
    return inputs, targets



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

    input_train = []
    input_valid = []
    input_test = []

    target_train = []
    target_valid = []
    target_test = []
    # apply 'preprocess' method to for each text/motion pair
    for pair in paired_file_paths:
        # the last letter of file name means speaker's personality

        if pair[0][-1] == ptype:
            inputs, targets = process_function(pair[1], pair[2], word2idx)
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

    train_size = int(0.8 * len(all_input))
    valid_size = int(0.1 * len(all_input))
    test_size = len(all_input) - train_size - valid_size

    for i in range(0, train_size):
        for j in range(0, len(all_input[i])):
            input_train.append(all_input[i][j])


    for i in range(train_size, train_size + valid_size):
        for j in range(0, len(all_input[i])):
            input_valid.append(all_input[i][j])

    for i in range(train_size+valid_size, len(all_input)):
        for j in range(0, len(all_input[i])):
            input_test.append(all_input[i][j])



    for i in range(0, train_size):
        for j in range(0, len(all_target[i])):
            target_train.append(all_target[i][j])


    for i in range(train_size, train_size + valid_size):
        for j in range(0, len(all_target[i])):
            target_valid.append(all_target[i][j])

    for i in range(train_size+valid_size, len(all_target)):
        for j in range(0, len(all_target[i])):
            target_test.append(all_target[i][j])


    np.savez(save_path+"_train.npz", input=input_train, target=target_train)
    np.savez(save_path+"_valid.npz", input=input_valid, target=target_valid)
    np.savez(save_path+"_test.npz", input=input_test, target=target_test)

    # concatenate the results together and save
    # all_input = [line for inputs in all_input for line in inputs]
    # all_target = [freq for targets in all_target for freq in targets]
    # print(len(all_input))
    # print(len(input_train_set))
    # print(len(input_valid_set))
    # print(len(input_test_set))

    # print(len(input_test_set))
    # np.savez(save_path+"_train.npz", input=input_train_set, target=target_train_set)
    # np.savez(save_path+"_valid.npz", input=input_valid_set, target=target_valid_set)
    # np.savez(save_path+"_test.npz", input=input_test_set, target=target_test_set)
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
    word2idx = load_encode('encode_dict.txt')

    # make three kinds of dataset
    # make_dataset(dct_preprocess, paired_file_paths, 'e',
    #              './data/extro_dct_dataset.npz')
    # print('extroverted data finished')
    # make_dataset(dct_preprocess, paired_file_paths, 'i',
    #              './data/intro_dct_dataset.npz')
    # print('introverted data finished')
    # make_dataset(dct_preprocess, paired_file_paths, 'n',
    #              './data/natural_dct_dataset.npz')
    # print('natural data finished')

#%%
# this code is to make dataset for seq2seq baseline model


# make_different_datasets(paired_file_paths)



# for personality in ['e','i','n']:
#     with open(personality + ".dic") as F:
#             personality_dic = json.loads(F.read())
#     personality_keys = list(personality_dic.keys())
#     train_size = int(0.8 * len(personality_keys))
#     valid_size = int(0.1 * len(personality_keys))
#     test_size = len(personality_keys) - train_size - valid_size
#
#     random.shuffle(personality_keys)
#     train_data = personality_keys[:train_size]
#     valid_data = personality_keys[train_size:(train_size + valid_size)]
#     test_data = personality_keys[(train_size + valid_size ):]
    # train_set, valid_set, test_set = random_split(e_keys, [train_size, valid_size, test_size])
    # print(e_keys)


    # print(len(e_keys))
    # print(len(train_data))
    # print(len(valid_data))
    # print(len(test_data))


    # print(paired_file_paths)
    # personality_train_list= [x for x in paired_file_paths if x[1] in train_data]
    # personality_valid_list= [x for x in paired_file_paths if x[1] in valid_data]
    # personality_test_list= [x for x in paired_file_paths if x[1] in test_data]
    #
    # # make three kinds of dataset
    # make_dataset(seq2seq_preprocess, personality_train_list, personality,
    #                 './train_valid_test_data/'+ personality +'_seq2seq_dataset_train.npz')
    # print(personality + ' training data finished')
    # make_dataset(seq2seq_preprocess, personality_valid_list, personality,
    #                 './train_valid_test_data/'+ personality +'_seq2seq_dataset_valid.npz')
    # print(personality + ' validation data finished')
    # make_dataset(seq2seq_preprocess, personality_test_list, personality,
    #                 './train_valid_test_data/'+ personality +'_seq2seq_dataset_test.npz')
    # print(personality + ' test data finished')

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
            elif '-' in word:
                sentense_ints = vocab_to_int["<Long>"]
            else:
                sentense_ints = vocab_to_int["<UNK>"]
#         if eos:
#             sentense_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentense_ints)
    ints.append(vocab_to_int["<EOS>"])
    return ints

the_path = '/afs/inf.ed.ac.uk/group/cstr/projects/galatea/d02'
data_path_motion = the_path + '/Recordings_October_2014/DOF-hiroshi/'
data_path_text = the_path + '/Recordings_October_2014/Transcriptions/transcriptions_phrase_tables/'

paired_file_paths = pair_files(data_path_text, '.TABLE', data_path_motion,
                                '.qtn')
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
