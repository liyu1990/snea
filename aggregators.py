################################################################################
# SNEA/aggregators.py
# Used to define the aggregation methods for the SNEA
# Note: This is based on the SGCN Implementation provided by Tyler Derr.
################################################################################

import random
import torch
import torch.nn as nn

from layers import SpGraphAttentionLayer


class NonFirstLayerAggregator(nn.Module):
    def __init__(self, _id, features, cuda=False, in_feat_dim=32, out_feat_dim=32, nheads=1):
        super(NonFirstLayerAggregator, self).__init__()
        self.id = _id
        self.features = features
        self.cuda = cuda

        """
        In this part, we only use the one-head attention mechanism.
        Maybe you can set nheads>1 to modify it to multi-head, and use dropout method to get a better result.
        As a result, more computation time will be required.
        """

        self.attentions_bal = [
            SpGraphAttentionLayer(in_features=in_feat_dim, out_features=out_feat_dim, cuda_available=cuda)
            for _ in range(nheads)]
        self.attentions_unbal = [
            SpGraphAttentionLayer(in_features=in_feat_dim, out_features=out_feat_dim, cuda_available=cuda)
            for _ in range(nheads)]

        for i, attention in enumerate(self.attentions_bal + self.attentions_unbal):
            self.add_module('attention_{}_{}'.format(self.id, i), attention)

    def forward(self, nodes, to_neighs_pos, to_neighs_neg, num_sample=None):
        """
        nodes --- list of nodes in a batch
        to_neighs_pos --- list of sets, each set is the set of positive neighbors for node in batch
        to_neighs_neg --- list of sets, each set is the set of negative neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs_pos = [_set(_sample(to_neigh, num_sample, ))
                               if len(to_neigh) >= num_sample else to_neigh
                               for to_neigh in to_neighs_pos]
            samp_neighs_neg = [_set(_sample(to_neigh, num_sample, ))
                               if len(to_neigh) >= num_sample else to_neigh
                               for to_neigh in to_neighs_neg]
        else:
            samp_neighs_pos = to_neighs_pos
            samp_neighs_neg = to_neighs_neg

        self_nodes = [set([nodes[i]]) for i, samp_neigh in enumerate(nodes)]
        unique_nodes_list = list(set.union(*samp_neighs_pos).union(*samp_neighs_neg).union(*self_nodes))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        column_indices_pos = [unique_nodes[n] for samp_neigh in samp_neighs_pos for n in samp_neigh]
        column_indices_neg = [unique_nodes[n] for samp_neigh in samp_neighs_neg for n in samp_neigh]
        column_indices_self = [unique_nodes[selfnode] for selfset in self_nodes for selfnode in selfset]

        row_indices_pos = [i for i in range(len(samp_neighs_pos)) for _ in range(len(samp_neighs_pos[i]))]
        row_indices_neg = [i for i in range(len(samp_neighs_neg)) for _ in range(len(samp_neighs_neg[i]))]
        row_indices_self = [i for i in range(len(self_nodes)) for _ in range(len(self_nodes[i]))]

        adj_bal = torch.tensor([row_indices_pos, column_indices_pos])
        adj_unbal = torch.tensor([row_indices_neg, column_indices_neg])
        adj_self = torch.tensor([row_indices_self, column_indices_self])

        if self.cuda:
            adj_bal = adj_bal.cuda()
            adj_unbal = adj_unbal.cuda()
            adj_self = adj_self.cuda()
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        embed_matrix_bal = embed_matrix[0]
        embed_matrix_unbal = embed_matrix[1]

        x_unbal = torch.cat(
            [att(embed_matrix_unbal, torch.cat((adj_self, adj_bal), dim=1), embed_matrix_bal, adj_unbal,
                 shape=(len(self_nodes), len(unique_nodes)))
             for att in self.attentions_unbal], dim=1)
        x_bal = torch.cat(
            [att(embed_matrix_bal, torch.cat((adj_self, adj_bal), dim=1), embed_matrix_unbal, adj_unbal,
                 shape=(len(self_nodes), len(unique_nodes)))
             for att in self.attentions_bal], dim=1)

        return x_bal, x_unbal


class FirstLayerAggregator(nn.Module):
    def __init__(self, _id, features, only_layer, cuda=False, in_feat_dim=64, out_feat_dim=32, nheads=1):
        super(FirstLayerAggregator, self).__init__()
        self.id = _id
        self.features = features
        self.cuda = cuda
        self.only_layer = only_layer

        self.attentions_bal = [
            SpGraphAttentionLayer(in_features=in_feat_dim, out_features=out_feat_dim, cuda_available=cuda)
            for _ in range(nheads)]
        self.attentions_unbal = [
            SpGraphAttentionLayer(in_features=in_feat_dim, out_features=out_feat_dim, cuda_available=cuda)
            for _ in range(nheads)]

        for i, attention in enumerate(self.attentions_bal + self.attentions_unbal):
            self.add_module('attention_{}_{}'.format(self.id, i), attention)

    def forward(self, nodes, to_neighs_pos, to_neighs_neg, num_sample=None):
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs_pos = [_set(_sample(to_neigh, num_sample, ))
                               if len(to_neigh) >= num_sample else to_neigh
                               for to_neigh in to_neighs_pos]
            samp_neighs_neg = [_set(_sample(to_neigh, num_sample, ))
                               if len(to_neigh) >= num_sample else to_neigh
                               for to_neigh in to_neighs_neg]
        else:
            samp_neighs_pos = to_neighs_pos
            samp_neighs_neg = to_neighs_neg

        if self.only_layer:
            self_nodes = [set([nodes[i]]) for i, samp_neigh in enumerate(nodes)]
        else:
            self_nodes = [{nodes[i].item()} for i, samp_neigh in enumerate(nodes)]

        unique_nodes_list = list(set.union(*samp_neighs_pos).union(*samp_neighs_neg).union(*self_nodes))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        column_indices_pos = [unique_nodes[n] for samp_neigh in samp_neighs_pos for n in samp_neigh]
        column_indices_neg = [unique_nodes[n] for samp_neigh in samp_neighs_neg for n in samp_neigh]
        column_indices_self = [unique_nodes[selfnode] for selfset in self_nodes for selfnode in selfset]

        row_indices_pos = [i for i in range(len(samp_neighs_pos)) for _ in range(len(samp_neighs_pos[i]))]
        row_indices_neg = [i for i in range(len(samp_neighs_neg)) for _ in range(len(samp_neighs_neg[i]))]
        row_indices_self = [i for i in range(len(self_nodes)) for _ in range(len(self_nodes[i]))]

        adj_bal = torch.tensor([row_indices_pos, column_indices_pos])
        adj_unbal = torch.tensor([row_indices_neg, column_indices_neg])
        adj_self = torch.tensor([row_indices_self, column_indices_self])

        if self.cuda:
            adj_bal = adj_bal.cuda()
            adj_unbal = adj_unbal.cuda()
            adj_self = adj_self.cuda()
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        x_bal = torch.cat(
            [att(embed_matrix, torch.cat((adj_self, adj_bal), dim=1), shape=(len(samp_neighs_pos), len(unique_nodes)))
             for att in self.attentions_bal], dim=1)
        x_unbal = torch.cat(
            [att(embed_matrix, torch.cat((adj_self, adj_unbal), dim=1), shape=(len(samp_neighs_neg), len(unique_nodes)))
             for att in self.attentions_unbal], dim=1)

        return x_bal, x_unbal
