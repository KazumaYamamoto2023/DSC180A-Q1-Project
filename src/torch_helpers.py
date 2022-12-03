import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re


def torch_preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.A

def torch_preprocess_adj(adj):
    adj_normalized = torch_normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized.A

def torch_normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def torch_batch_gd(model, criterion, optimizer, train_iter, test_iter, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    
    for it in range(epochs):
        train_loss = []
        for inputs, targets in train_iter:
            targets = targets.view(-1,1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        # Get train loss and test loss
        train_loss = np.mean(train_loss)
        test_loss = []
        for inputs, targets in test_iter:
            targets = targets.view(-1,1).float()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        
        # save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        print(f"Epoch {it+1}/{epochs}, Train Loss: {train_loss:4f},\
             Test Loss: {test_loss:4f}")
    return train_losses, test_losses

class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0., num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, \
                       output_dim, \
                       support, \
                       act_func = None, \
                       featureless = False, \
                       dropout_rate = 0., \
                       bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless
        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))
        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)
        if self.act_func is not None:
            out = self.act_func(out)
        self.embedding = out
        return out

class GCN(nn.Module):
    def __init__(self, input_dim, \
                       support,\
                       dropout_rate=0., \
                       num_classes=10):
        super(GCN, self).__init__()
        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(200, num_classes, support, dropout_rate=dropout_rate)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class CNN(nn.Module):
    def __init__(self,n_vocab,embed_dim,n_outputs):
        super(CNN,self).__init__()
        self.V = n_vocab
        self.D = embed_dim
        self.K = n_outputs
        self.embed = nn.Embedding(self.V,self.D)
        # conv layers
        self.conv1 = nn.Conv1d(self.D,32,3,padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32,64,3,padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64,128,3,padding=1)
        self.fc = nn.Linear(128,self.K)
    
    def forward(self, X):
        # embedding layer
        out = self.embed(X)        
        # conv layers
        out = out.permute(0,2,1)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = out.permute(0,2,1)
        # max pool
        out, _ = torch.max(out,1)
        #final dense layer
        out = self.fc(out)
        return out