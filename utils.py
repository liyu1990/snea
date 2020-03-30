### adopted from SGCN Implementation provided by Tyler Derr.

import argparse
from random import randint
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pickle
import datetime


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_in_undirected_network(file_name):
    links = {}
    with open(file_name) as fp:
        n, m = [int(val) for val in fp.readline().split()[-2:]]
        for l in fp:
            rater, rated, sign = [int(val) for val in l.split()]
            assert (sign != 0)
            sign = 1 if sign > 0 else -1

            edge1, edge2 = (rater, rated), (rated, rater)
            if edge1 not in links:
                links[edge1], links[edge2] = sign, sign
            elif links[edge1] == sign:  # we had it before and it was the same
                pass
            else:  # we had it before and now it's a different value
                links[edge1], links[edge2] = -1, -1  # set to negative

    adj_lists_pos, adj_lists_neg = defaultdict(set), defaultdict(set)
    num_edges_pos, num_edges_neg = 0, 0
    for (i, j), s in links.items():
        if s > 0:
            adj_lists_pos[i].add(j)
            num_edges_pos += 1
        else:
            adj_lists_neg[i].add(j)
            num_edges_neg += 1
    num_edges_pos /= 2
    num_edges_neg /= 2

    return n, [num_edges_pos, num_edges_neg], adj_lists_pos, adj_lists_neg


def read_in_feature_data(feature_file_name, num_input_features):
    feat_data = pickle.load(open(feature_file_name, "rb"))
    if num_input_features is not None:
        # we perform a shrinking as to which features we are using
        feat_data = feat_data[:, :num_input_features]
    num_nodes, num_feats = feat_data.shape
    # standardizing the input features
    feat_data = StandardScaler().fit_transform(feat_data)  # .T).T

    return num_feats, feat_data


def load_data(network_file_name, feature_file_name, test_network_file_name, num_input_features):
    num_nodes, num_edges, adj_lists_pos, adj_lists_neg = read_in_undirected_network(network_file_name)
    num_feats, feat_data = read_in_feature_data(feature_file_name, num_input_features)

    if test_network_file_name is not None:
        test_num_nodes, test_num_edges, test_adj_lists_pos, test_adj_lists_neg = \
            read_in_undirected_network(test_network_file_name)
    else:
        test_num_nodes, test_adj_lists_pos, test_adj_lists_neg = None, None, None

    return num_nodes, num_edges, adj_lists_pos, adj_lists_neg, \
           num_feats, feat_data, test_adj_lists_pos, test_adj_lists_neg


def calculate_class_weights(num_V, num_pos, num_neg, w_no=None):
    num_E = num_pos + num_neg
    num_V = num_V * 2  # sampling 2 non-connected nodes for each node in network.

    num_total = num_E + num_V

    if w_no is None:
        w_no = round(num_V * 1.0 / num_total, 2)
    else:
        assert isinstance(w_no, float) and 0 < w_no < 1
    w_pos_neg = 1 - w_no

    w_pos = round(w_pos_neg * num_neg / num_E, 2)
    w_neg = round(w_pos_neg - w_pos, 2)

    return w_pos, w_neg, w_no
