import os
import time
import glob
import numpy as np


def create_glove_vocab(glove_path: str) -> list:
    """Extract GloVe vocabulary

    If one only wants to check whether word in GloVe, it can
    make use of this returned list to save time and memory cost.

    Args:
        glove_path: the path string of GloVe embedding file.
        For example, './glove/glove.6B.100d.txt'

    Returns:
        glove_vocab: a list containing vocabs of GloVe

    """
    glove_vocab = []
    with open(glove_path, 'r') as file:
        for line in file:
            glove_vocab.append(line.split()[0])
    return glove_vocab


def save_vocab(path: str, tokens) -> None:
    """Save some words

    Args:
        path: the save file path
        tokens: a list/set that contains words
    """
    with open(path, 'w') as f:
        for t in tokens:
            f.write(t + '\n')  # one token per line


def load_vocab(path: str) -> set:
    """Load words from file

    Usage example:
        glove_vocab = load_vocab('glove_vocab.txt')
        save_vocab('glove_vocab.txt', glove_vocab)

    Args:
        path: the words file path

    Returns:
        tokens: a set of words, use set for fast operation,
            like set intersection, searching
    """

    tokens = set()
    with open(path, 'r') as f:
        for line in f:
            tokens.add(line.strip())
    return tokens


# token encoding dict
def save_encode(path: str, word2idx: dict):
    """Save the word map

    Use after encode word to index

    Args:
        path: the path to word map file
        word2idx: a dictionary saving word:index pairs
    """
    with open(path, 'w') as f:
        for k, v in word2idx.items():
            f.write(k + ' ' + str(v) + '\n')


def load_encode(path: str) -> dict:
    """Load the word map

    Usage example:
        word2idx = load_encode('encode_dict.txt')
        save_encode('encode_dict.txt', word2idx)

    Args:
        path: the file path to save word map

    Returns:
        word2idx: a dictionary saving word:index pairs
    """

    word2idx = dict()
    with open(path, 'r') as f:
        for line in f:
            temp = line.strip().split()
            word2idx[temp[0]] = int(temp[1])
    return word2idx


def pair_files(folder_path_text: str, text_suffix: str,
               folder_path_motion: str, motion_suffix: str) -> list:
    """Pair motion and corresponding text file by file name

    After getting folders of motion and text, this function make a map
    that link these two files for subsequent processing.  The Python
    module 'glob' is used to look for files by wildcard.

    Args:
        folder_path_text: path of text file
        text_suffix: can be '' and '.TABLE'
        folder_path_motion: path of motion file
        motion_suffix: can be '.axa', '.rov', '.quat'

    Returns:
        paired_file_paths: its shape is
            [NAME_NUM_[ine], actual text path, actual motion path]
    """

    # check existing and complete the path
    assert os.path.isdir(folder_path_motion)
    assert os.path.isdir(folder_path_text)
    folder_path_motion += '/' if folder_path_motion[-1] != '/' else ''
    folder_path_text += '/' if folder_path_text[-1] != '/' else ''

    def path_list2dict(file_list: list, suffix: str) -> dict:
        """
        Convert list of file path to dict. Return a dictionary,
        whose keys real name value is actual path of this.
        Example: {Adam_06_e:'../DOF/Adam/Adam_06_e.axa'}
        """
        file_dict = {}
        for path in file_list:
            key = path[path.rfind('/') + 1:len(path) - len(suffix)]
            file_dict[key] = path
        return file_dict

    # 1. read names of all transcript files
    # folder structure: folder/NAME/NAME_NUM_[ine].suffix
    path_WC = folder_path_text + '*/*_??_[ine]' + text_suffix
    transcript_file_list = glob.glob(path_WC, recursive=True)
    print(len(transcript_file_list))
    transcript_file_dict = path_list2dict(transcript_file_list, text_suffix)

    # 2. read names of all MOTION file
    # folder structure: folder/NAME/Head/Normalised/NAME_NUM_[ine].suffix

    path_WC = folder_path_motion + '*/Head/Normalised/*_??_[ine]' + motion_suffix
    motion_file_list = glob.glob(path_WC, recursive=True)
    motion_file_dict = path_list2dict(motion_file_list, motion_suffix)

    # only keep shared motion-text pairs
    shared_names = set.intersection(
        set(motion_file_dict.keys()), set(transcript_file_dict.keys()))

    paired_file_paths = [
        [name, transcript_file_dict[name], motion_file_dict[name]]
        for name in shared_names
        if (name in transcript_file_dict) and (name in motion_file_dict)
    ]
    # print((paired_file_paths))

    return paired_file_paths


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    Copied from
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def now_time() -> str:
    """
    Generate string of current time accurate to second.
    Example return value: 0714_143436
    """

    return time.strftime('%m%d_%H%M%S', time.localtime())


def token_regularizer(word: str) -> list:
    """Expand contractions and remove other unwanted characters

    Args:
        word: one word to process

    Returns:
        word_list: a list containing one or more words
    """
    word = word.lower()
    if "'" in word:
        # use contractions dict to split contractions
        if word in contractions_dict:
            return contractions_dict[word].split()
        else:
            # most case (not in dict) is possessive
            # just discard /'s/
            return [word[:word.rfind("'")]]
    elif '-' in word:
        return [word.strip("-")]
    else:
        return [word]


contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "could't": "could not",
    "couldn't've": "could not have",
    "d'you": "do you",  # add
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "here'd": "here had",  # add
    "here's": "here is",  # add
    "here'll": "here will",  # add
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i had",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "iit will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "that'll": "that will",  # add
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "there'll": "there will",  # add
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what'd": "what would",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "who'd": "who would",
    "why's": "why is",
    "why'd": "why would",  # add
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
