import os
import pickle as pkl
from collections import Counter

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3


class Vocab(object):
    def __init__(self):
        self.token2id = {
            "<s>": START_TOKEN, "</s>": END_TOKEN,
            "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN
        }

        self.id2token = dict((i, token) for token, i in self.token2id.items())

        self.length = 4

    def add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = self.length
            self.id2token[self.length] = token
            self.length += 1

    def __len__(self):
        return self.length


def build_vocab(formulas_dir, min_count=10):
    """
    Build vocabulary of tokens with the given file and store it
    """
    vocab = Vocab()
    counter = Counter()
    # get the list of formulas into an array
    formulas = open(formulas_dir, 'r').read().split('\n')[:-1]
    # split the formulas and count the tokens
    for formula in formulas:
        split_formula = formula.split()
        counter.update(split_formula)
    # add most common tokens into the vocab
    for token, count in counter.most_common():
        if count >= min_count:
            vocab.add_token(token)
    # name of file where the vocab will be saved to
    vocab_file = os.path.join(os.path.split(formulas_dir)[:-1], "vocab.pkl")
    # save the vocab to vocab_file
    with open(vocab_file, "wb") as w:
        pkl.dump(vocab, w)


def load_vocab(vocab_dir):
    with open(vocab_dir, 'rb') as f:
        vocab = pkl.load(f)

    return vocab
