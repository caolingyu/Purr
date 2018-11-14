# coding = utf-8
import datasets
from models import Conv_Attn

import torch
import numpy as np
import csv
from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm

# Define some constants
MODEL_DIR = "./saved_models/your_model"
VOCAB_FILE = "./data/vocab.csv"
LABEL_FILE = "./data/label_list.csv"
EMBED_FILE = None
FILTER_SIZE = 4
NUM_FILTER_MAPS = 500
GPU = False
EMBED_SIZE = 200
BATCH_SIZE = 32


def load_label_list(label_list):
    ind2l = defaultdict(str)
    with open(label_list, "r") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            ind2l[i] = line
    return ind2l


def input2array(batch_input):
    data = []
    max_len = 0
    for item in batch_input:
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in item.split()]
        data.append(text)
        max_len = max(len(text), max_len) 
    padded_data = pad_input(data, max_len)
    data = np.array(padded_data)
    return data


def pad_input(batch_data, max_len):
    padded_data = []
    for text in batch_data:
        if len(text) < max_len:
            text.extend([0] * (max_len - len(text)))
            padded_data.append(text)
        else:
            padded_data.append(text)
    return padded_data


ind2w, w2ind = datasets.load_vocab_dict(VOCAB_FILE)
ind2l = load_label_list(LABEL_FILE)
l2ind = {l:i for i,l in ind2l.items()}
dicts = (ind2w, w2ind, ind2l, l2ind)

model = torch.load(MODEL_DIR, map_location="cpu")


def do_inference(data):
    batch_data = []
    result_total = []
    for i in data:
        text = i.strip()
        batch_data.append(text)
        if len(batch_data) == BATCH_SIZE:
            result = [[] for i in range(len(batch_data))]
            text = input2array(batch_data)
            data = Variable(torch.LongTensor(text), volatile=True) # set volatile=True for inference
            model.train(False)
            output, _, _ = model(data, target=None)
            output = output.data.cpu().numpy()
            output = np.round(output)
            non_zero_index = np.nonzero(output)

            for i, x in enumerate(non_zero_index[0]):
                result[x].append(ind2l[non_zero_index[1][i]])

            result_total.extend(result)
            batch_data = []

    if len(batch_data) != 0:
        result = [[] for i in range(len(batch_data))]
        text = input2array(batch_data)
        data = Variable(torch.LongTensor(text), volatile=True)
        model.train(False)
        output, _, _ = model(data, target=None)
        output = output.data.cpu().numpy()
        output = np.round(output)
        non_zero_index = np.nonzero(output)

        for i, x in enumerate(non_zero_index[0]):
            result[x].append(ind2l[non_zero_index[1][i]])

        result_total.extend(result)
    
    return result_total
 