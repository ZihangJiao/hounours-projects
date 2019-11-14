#%%
import numpy as np
import os
from utils import pair_files
from utils import token_regularizer
from utils import load_vocab
from utils import load_encode
from utils import save_encode
#%%

def make_tokens(text_file_paths: list) -> set:
    """Extract tokens from all transcripts
    
    Args:
        text_file_paths: path of the transcript file paths
    Returns:
        a set of tokens that in glove vocabulary
    """
    token_set = set()
    for path in text_file_paths:
        try:
            words = np.loadtxt(path, usecols=3, dtype='str')
        except:
            # there is some files use csv format
            words = np.loadtxt(path, usecols=3, delimiter=',', dtype='str')

        tokens = [w.lower() for w in set(words)]
        # set rules to split words to deal with 's 're 've, etc
        for t in tokens:
            for word in token_regularizer(t):
                token_set.add(word)
    return set(token_set)

#%%
# use wordmap to creat pretrained embedding weights
def make_pretraiend_weight(embedding_file_path: str, embed_dim: int,
                           idx2word: dict) -> list:
    """
    
    Args:
        embedding_file_path:
        embed_dim: number of embedding dimention, for glove.6B.100d, it's 100.
        idx2word: wordmap that convert index back to word.
    
    Returns:
        glove_weight: weight matrix, shape [vocab_len, embed_dim]
    """
    # 1. load all glove weights first
    glove_embed_dict = {} 
    with open(embedding_file_path, 'r') as file:
        for line in file:
            temp = line.strip().split()
            glove_embed_dict[temp[0]] = [float(i) for i in temp[1:]]

    # 2. then make a 'lite' glove weights
    glove_weight = [[0] * embed_dim]  # 0 for <pad>

    # follow the order, generate pretrained weight for encode
    for i in range(1, len(idx2word)-3):
        glove_weight.append(glove_embed_dict[idx2word[i]])
    
    # use average weight as that of <unk>
    unk_embed = np.mean([v for v in glove_embed_dict.values()], axis=0)
    glove_weight.append(unk_embed)
    glove_weight.append([0] * embed_dim)  # 0 for <bos>
    glove_weight.append([0] * embed_dim)  # 0 for <eos>

    return glove_weight


#%%
data_path_motion = '../Recordings_October_2014/DOF-hiroshi/'
data_path_text = '../Recordings_October_2014/transcriptions_word_tables/transcriptions_word_tables'

# make paires of motion and text
paired_file_paths = pair_files(data_path_text, '', data_path_motion, '.rov')
text_file_paths = [row[1] for row in paired_file_paths]

# find all vocab in our dataset
token_list = list(make_tokens(text_file_paths))

# remove tokens that not in GloVe vocab
glove_vocab = load_vocab('glove_vocab.txt')
token_list = [t for t in token_list if t in glove_vocab]

# encode these tokens
word2idx = {t: idx + 1 for idx, t in enumerate(token_list)}
word2idx['<unk>'] = len(word2idx) + 1
word2idx['<bos>'] = len(word2idx) + 1
word2idx['<eos>'] = len(word2idx) + 1
word2idx['<pad>'] = 0

idx2word = {v: k for k, v in word2idx.items()}

# customize glove weights
glove_weight = make_pretraiend_weight('glove/glove.6B.100d.txt', 100, idx2word)
glove_weight = np.array(glove_weight)

# save it
save_path = './glove_pretrained_weights'
np.save(save_path, glove_weight)
