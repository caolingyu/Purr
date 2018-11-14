# coding = utf-8
import sys
sys.path.append('../')
import argparse

import datasets
from preprocess import build_vocab, word_embeddings, extract_wvs
from constants import DATA_DIR

import pandas as pd
from sklearn.model_selection import train_test_split

import csv
import re
import codecs
from tqdm import tqdm

def raw_to_csv(f_in, f_out):
    f_out = codecs.open(f_out, "w")
    writer = csv.writer(f_out)
    writer.writerow(["text", "label", "length"])
    count = 1
    i = 0
    prev_key = ""
    prev_value = ""
    cur_value = ""
    with open(f_in, "r") as f:
        for line in tqdm(f):
            line = line.split("\t")
            if len(line) != 2:
                continue
            cur_key = line[1].strip() # text
            cur_value = line[0] + "\n" # label
            if i != 0:
                if cur_key != prev_key:
                    label = []
                    for item in prev_value.split("\n"):
                        if item and item not in label:
                            label.append(item)
                    writer.writerow([prev_key, "|".join(label), len(prev_key)])
                    prev_value = ""
                    count += 1
            prev_key = cur_key
            prev_value += cur_value
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    # f_in = "{}raw_data".format(DATA_DIR)
    f_in = args.data_path
    f_out = "{}data.csv".format(DATA_DIR)
    raw_to_csv(f_in, f_out)

    train_file = '{}train.csv'.format(DATA_DIR)
    test_file = '{}test.csv'.format(DATA_DIR)

    df = pd.read_csv(f_out, engine="python")
    train, test = train_test_split(df, test_size=0.2)
    train = train.sample(frac=1)
    test = test.sample(frac=1)
    # train = train.sort_values(["length"])
    # test = test.sort_values(["length"])

    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

    label_list_raw = train["label"].value_counts().index.values
    label_list = set()
    for item in label_list_raw:
        for l in item.split("|"):
            label_list.add(l)
    with open("{}label_list.csv".format(DATA_DIR), "w") as of:
        w = csv.writer(of)
        for label in label_list:
            w.writerow([label])

    # build vocabulary
    vocab_min = 3
    vname = "{}vocab.csv".format(DATA_DIR)
    build_vocab.build_vocab(vocab_min, train_file, vname)

    # train word embeddings
    w2v_file = word_embeddings.word_embeddings(train_file, 100, 0, 5)
    extract_wvs.gensim_to_embeddings("{}processed.w2v".format(DATA_DIR), "{}vocab.csv".format(DATA_DIR))
