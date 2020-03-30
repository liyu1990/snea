#!/usr/bin/env python
# coding:utf-8
# @author : liyu

import argparse
from random import randint
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import warnings
import datetime

from utils import str2bool, load_data, calculate_class_weights
from encoders import LayerEncoder
from aggregators import FirstLayerAggregator, NonFirstLayerAggregator
from model import SNEA

warnings.filterwarnings("ignore")

# ================================================================================= #

parser = argparse.ArgumentParser(description="""This is the code to run the SNEA.""")
parser.add_argument('--cuda_available', type=bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)  # -1 for cpu;
parser.add_argument('--network_file_name', type=str, required=True)
parser.add_argument('--feature_file_name', type=str, required=True)
parser.add_argument('--test_network_file_name', type=str, required=False, default=None)
parser.add_argument('--lambda_structure', type=float, default=4.0)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--model_regularize', type=float, default=0.01)
parser.add_argument('--num_layers', type=int, default=2)  # required=True)
# bitcoinAlpha, bitcoinOtc: 1000; Slashdot, Epinions: 5000
parser.add_argument('--batch_size', type=int, default=1000)
# bitcoinAlpha, bitcoinOtc: 10; Slashdot, Epinions: 100
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=randint(0, 2147483648))
# bitcoinAlpha, bitcoinOtc: 1000; Slashdot, Epinions: 4000
parser.add_argument('--total_minibatches', type=int, default=1000)
parser.add_argument('--save_embeddings_interval', type=int, default=100)
parser.add_argument('--num_neighbors_sample', type=int, default=None)  # 250
parser.add_argument('--num_input_features', type=int, default=64)  # none means all
parser.add_argument('--modify_input_features', type=str2bool, nargs='?', const=True, default=True)
# we can use the calculate_class_weights() to generate class weights for pos, neg, no.
parser.add_argument('--class_weights', type=lambda s: [float(item) for item in s.split('a')], default=None)
# assign "no" link weight, as a parameter for function calculate_class_weights()
parser.add_argument('--class_weight_no', type=float, default=0.35)  # 0.35 as default
parser.add_argument('--model_path', type=str,
                    default="modules/model_snea{}.pkl".format((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))

parameters = parser.parse_args()
args = {}
for arg in vars(parameters):
    args[arg] = getattr(parameters, arg)
# ================================================================================= #


rnd_seed = args['random_seed']
np.random.seed(rnd_seed)
random.seed(rnd_seed)
torch.manual_seed(rnd_seed)

cuda = args['cuda_available']
if args['cuda_device'] == -1:
    cuda = False
if cuda:
    print("Using {} CUDA!!!".format(args['cuda_device']))
    torch.cuda.set_device(args['cuda_device'])
    torch.cuda.manual_seed(rnd_seed)
else:
    print("Using CPU!!!")

saved_model = args["model_path"].strip()
num_neighbors_sample = args['num_neighbors_sample']  # None

# ================================================================================= #


num_nodes, num_edges, adj_lists_pos, adj_lists_neg, num_feats, feat_data, test_adj_lists_pos, test_adj_lists_neg = \
    load_data(args['network_file_name'], args['feature_file_name'], args['test_network_file_name'],
              args['num_input_features'],  # test_size=0.2
              )

if args['num_input_features'] is None:
    args['num_input_features'] = num_feats

#args['class_weights'] = calculate_class_weights(num_nodes, num_edges[0], num_edges[1], w_no=None)
args['class_weights'] = calculate_class_weights(num_nodes, num_edges[0], num_edges[1], w_no=args['class_weight_no'])

features = nn.Embedding(num_nodes, num_feats)
if args['modify_input_features']:
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=True)
else:
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

if cuda:
    features.cuda()

##################################################################
# We use two aggregation layers in our experiments.
##################################################################
if args['num_layers'] == 2:
    layer1_in_dim, layer1_out_dim, nheads1 = num_feats, 32, 1
    layer2_in_dim, layer2_out_dim, nheads2 = layer1_out_dim * nheads1, 32, 1
    final_in_dim, final_out_dim = layer2_out_dim * nheads2 * 2, 64

    agg1 = FirstLayerAggregator(1, features, only_layer=False, cuda=cuda,
                                in_feat_dim=layer1_in_dim, out_feat_dim=layer1_out_dim, nheads=nheads1)
    enc1 = LayerEncoder(1, adj_lists_pos, adj_lists_neg, agg1,
                        num_sample=num_neighbors_sample, base_model=None, cuda=cuda, last_layer=False)

    agg2 = NonFirstLayerAggregator(2, lambda nodes: enc1(nodes), cuda=cuda,
                                   in_feat_dim=layer2_in_dim, out_feat_dim=layer2_out_dim, nheads=nheads2)
    enc2 = LayerEncoder(2, adj_lists_pos, adj_lists_neg, agg2,
                        num_sample=num_neighbors_sample, base_model=enc1, cuda=cuda, last_layer=True)

    snea = SNEA(num_nodes, final_in_dim, final_out_dim, enc2,
                args['class_weights'], args['lambda_structure'], cuda_available=cuda)
else:
    raise NotImplementedError('we advise using 2 layers... see code to use additional layers')

if cuda:
    snea.cuda()

train = list(np.random.permutation(list(range(0, num_nodes))))
total_batches = args['total_minibatches']
batch_size = args['batch_size']
batch_start = 0
batch_end = batch_size

if test_adj_lists_pos is not None:
    test_interval = args['test_interval']
else:
    test_interval = total_batches + 1  # i.e., never run the test_func

optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, snea.parameters()),
                                lr=args['learning_rate'], weight_decay=args['model_regularize'])

epoch_losses, batch_losses = [], []
epoch_loss, epoch = 0, 1
cnt_wait, minimal_loss, minimal_batch = 0, 1e9, 0

for batch in range(total_batches):
    snea.train()
    if batch_end > len(train):
        epoch += 1
        epoch_losses.append(epoch_loss)
        epoch_loss = 0
        batch_start = 0
        batch_end = batch_size
        random.shuffle(train)
    batch_center_nodes = train[batch_start:batch_end]
    batch_start = batch_end
    batch_end += batch_size

    # forward step
    optimizer.zero_grad()
    loss = snea.loss(batch_center_nodes, adj_lists_pos, adj_lists_neg)

    print('batch {} loss: {} patience: {}'.format(batch, loss, cnt_wait))
    sys.stdout.flush()
    if loss < minimal_loss:
        minimal_loss = loss
        minimal_batch = batch
        cnt_wait = 0
        torch.save(snea.state_dict(), saved_model)
    else:
        cnt_wait += 1

    loss.backward()
    optimizer.step()
    batch_loss = loss.item()
    batch_losses.append(batch_loss)
    epoch_loss += batch_loss

    if (batch + 1) % test_interval == 0 or batch == total_batches - 1:
        snea.eval()
        optimizer.zero_grad()
        auc, f1 = snea.test_func(adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg)
        if batch != total_batches - 1:
            print(batch, ' test_func sign prediction (auc,f1) :', auc, '\t', f1)
        else:
            print('{}{} Val(auc,f1):{} {}'.format("#" * 10, "LAST EPOCH", auc, f1))
        sys.stdout.flush()

print('Loading {}th epoch'.format(minimal_batch))
snea.load_state_dict(torch.load(saved_model))
snea.eval()
optimizer.zero_grad()
auc, f1 = snea.test_func(adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg)
print('{}{} Val(auc,f1):{} {}'.format("#" * 10, "BEST EPOCH", auc, f1))
