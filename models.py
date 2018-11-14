# coding = utf-8
"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

import numpy as np

import math
import random
import sys
import time

sys.path.append('../')
from constants import *
from preprocess import extract_wvs

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts,  dropout=0.5, gpu=False, embed_size=100):
        super(BaseModel, self).__init__()
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)

        # make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1])
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            vocab_size = len(dicts[0])
            self.embed = nn.Embedding(vocab_size+2, embed_size)


    def get_loss(self, yhat, target):
        # calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        return loss

    def params_to_optimize(self):
        return self.parameters()

# =========================================================================================

class Conv_Attn(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu, dicts, embed_size=100, dropout=0.5):
        super(Conv_Attn, self).__init__(Y, embed_file, dicts, dropout=dropout, gpu=gpu, embed_size=embed_size)

        # initialize conv layer
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(kernel_size/2))
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention
        self.U = nn.Linear(num_filter_maps, Y)
        self.U.bias.data.fill_(0)
        self.U.bias.requires_grad = False

        # final layer: create a matrix to use for the L binary classifiers
        if gpu:
            self.final = Variable(torch.zeros((Y, num_filter_maps)).cuda())
            self.final_bias = Variable(torch.zeros(Y).cuda())
        else:
            self.final = Variable(torch.zeros((Y, num_filter_maps)))
            self.final_bias = Variable(torch.zeros(Y))
        xavier_uniform(self.final)
        self.final.requires_grad = True
        self.final_bias.requires_grad = True
        
    def forward(self, x, target):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2))
        y = []
        ss = []
        alphas = []
        for x_i in x:
            # apply attention
            alpha_i = F.softmax(self.U(x_i).t())
            # document representations are weighted sums using the attention. Can compute all at once as a matmul
            m_i = alpha_i.mm(x_i)

            # final layer classification
            y_i = self.final.mul(m_i).sum(dim=1).add(self.final_bias)

            # save attention
            alphas.append(alpha_i)
            y.append(y_i)

        r = torch.stack(y)
        alpha = torch.stack(alphas)
            
        # final sigmoid to get predictions
        yhat = F.sigmoid(r)
        if target is not None:
            loss = self.get_loss(yhat, target)
        else:
            loss = 0
        return yhat, loss, alpha
    
    def params_to_optimize(self):
        # use this to include final layer parameters and exclude attention (U)'s bias term
        ps = []
        for param in self.embed.parameters():
            ps.append(param)
        for param in self.conv.parameters():
            ps.append(param)
        for i,param in enumerate(self.U.parameters()):
            # don't add bias
            if i == 0:
                ps.append(param)
        ps.append(self.final)
        ps.append(self.final_bias)
        return ps