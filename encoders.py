################################################################################
# SNEA/encoders.py
# Used to define the encoder methods for the SNEA
# Note: This is based on the SGCN Implementation provided by Tyler Derr.
################################################################################
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class LayerEncoder(nn.Module):
    def __init__(self, _id, adj_lists_pos, adj_lists_neg, aggregator,
                 num_sample=10, base_model=None, cuda=False, last_layer=False):
        super(LayerEncoder, self).__init__()

        self.id = _id
        self.last_layer = last_layer
        self.adj_lists_pos = adj_lists_pos
        self.adj_lists_neg = adj_lists_neg
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model is not None:
            self.base_model = base_model
        self.aggregator.cuda = cuda
        self.act_func = F.tanh

    def forward(self, nodes):
        if self.last_layer:
            feat_bal, feat_unbal = \
                self.aggregator.forward(nodes,
                                        [self.adj_lists_pos[node] for node in nodes],
                                        [self.adj_lists_neg[node] for node in nodes],
                                        num_sample=self.num_sample
                                        )
        else:
            feat_bal, feat_unbal = \
                self.aggregator.forward(nodes,
                                        [self.adj_lists_pos[node.item()] for node in nodes],
                                        [self.adj_lists_neg[node.item()] for node in nodes],
                                        num_sample=self.num_sample
                                        )

        mapped_feat_bal = self.act_func(feat_bal)
        mapped_feat_unbal = self.act_func(feat_unbal)

        return mapped_feat_bal, mapped_feat_unbal
