# coding = utf-8
"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys

from constants import *

data_dir = DATA_DIR

class Batch:
    def __init__(self):
        self.docs = []
        self.labels = []
        self.length = 0
        self.max_length = MAX_LENGTH
        self.raw_text = []

    def add_instance(self, row, ind2l, l2ind, w2ind, num_labels):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """
        labels = set()
        text = row[0]
        length = int(row[2])
        labels_idx = np.zeros(num_labels)
        labelled = False

        # get labels as a multi-hot vector
        for l in row[1].split("|"):
            if l in l2ind.keys():
                label = int(l2ind[l])
                labels_idx[label] = 1
                labelled = True
        if not labelled:
            return

        # OOV words are given a unique index at end of vocab lookup
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        # truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]
            length = self.max_length

        # build instance
        self.docs.append(text)
        self.labels.append(labels_idx)

        # reset length
        self.length = max(self.length, length)

        self.raw_text.append(row[0])

    def pad_docs(self):
        # pad all docs to have self.length
        padded_docs = []
        for doc in self.docs:
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)
        self.docs = padded_docs

    def to_ret(self):
        return np.array(self.docs), np.array(self.labels), self.raw_text


def data_generator(filename, dicts, batch_size, num_labels):
    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
        Yields:
            np arrays with data for training loop.
    """
    ind2w, w2ind, ind2l, l2ind = dicts[0], dicts[1], dicts[2], dicts[3]
    with open(filename, "r") as infile:
        r = csv.reader(infile)
        # skip header
        next(r)
        cur_inst = Batch()
        for row in r:
            # find the next "batch_size" instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs()
                yield cur_inst.to_ret()
                # clear
                cur_inst = Batch()
            cur_inst.add_instance(row, ind2l, l2ind, w2ind, num_labels)
        cur_inst.pad_docs()
        yield cur_inst.to_ret()


def load_vocab_dict(vocab_file):
    # reads vocab_file into two lookups
    ind2w = defaultdict(str)
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                ind2w[i+1] = line.rstrip()
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind


def load_full_labels(train_path):
    """
        Inputs:
            train_path: path to train dataset
        Outputs:
            label lookup
    """
    # build label lookups from appropriate datasets
    ind2l = defaultdict(str)
    with open('{}label_list.csv'.format(data_dir), 'r') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            ind2l[i] = line

    return ind2l


def load_lookups(train_path, vocab_file):
    """
        Inputs:
            train_path: path to train dataset
            vocab_file: path to vocab
        Outputs:
            vocab lookups, label lookups
    """
    # get vocab lookups
    ind2w, w2ind = load_vocab_dict(vocab_file)

    # get label
    ind2l = load_full_labels(train_path)
    l2ind = {l:i for i,l in ind2l.items()}

    dicts = (ind2w, w2ind, ind2l, l2ind)
    return dicts