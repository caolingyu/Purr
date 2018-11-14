# coding = utf-8
"""
    Various methods are kept here to keep the other code files simple
"""
import csv
import json
import math

import torch
# import torch.nn as nn

import models
from constants import *
import datasets
import persistence
import numpy as np

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
    Y = get_num_labels()
    if args.model == "conv_attn":
        filter_size = int(args.filter_size)
        model = models.Conv_Attn(Y, args.embed_file, filter_size, args.num_filter_maps, args.gpu, dicts, embed_size=args.embed_size, dropout=args.dropout)
    elif args.model == "saved":
        model = torch.load(args.test_model)
    if args.gpu:
        model.cuda()
    return model


def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    param_vals = [args.filter_size, args.dropout, args.num_filter_maps,  
                  args.command, args.weight_decay,  args.data_path, args.vocab, args.embed_file, args.lr]
    param_names = ["filter_size", "dropout", "num_filter_maps", "command",
                   "weight_decay", "data_path", "vocab", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params


def get_num_labels():
    num_labels = LABEL_SIZE
    return num_labels