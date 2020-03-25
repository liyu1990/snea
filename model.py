#!/usr/bin/env python
# coding:utf-8
# author: liyu
# Note: This is based on the SGCN Implementation provided by Tyler Derr.
################################################################################

from __future__ import print_function

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn import init

from random import randint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


################################################################################

class SNEA(nn.Module):
    def __init__(self, num_nodes, final_in_dim, final_out_dim, enc,
                 class_weights, lambda_structure, cuda_available=False):
        super(SNEA, self).__init__()
        self.num_nodes = num_nodes
        self.enc = enc
        self.lambda_structure = lambda_structure
        self.cuda_available = cuda_available
        if class_weights is None:
            self.CrossEntLoss = nn.CrossEntropyLoss()
        else:
            self.CrossEntLoss = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights)
            )
        self.structural_distance = nn.PairwiseDistance(p=2)
        self.weight = nn.Parameter(torch.FloatTensor(final_in_dim, final_in_dim))
        self.param_src = nn.Parameter(torch.FloatTensor(2 * final_out_dim, 3))
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.param_src)
        self.act_func = F.tanh

    def forward(self, nodes):
        embeds_bal, embeds_unbal = self.enc(nodes)
        combined_embedding = torch.cat([embeds_bal, embeds_unbal], dim=1)
        final_embedding = self.act_func(self.weight.mm(combined_embedding.t()))
        return final_embedding.t()

    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg):
        max_node_index = self.num_nodes - 1
        # get the correct nodes based on this minibatch
        i_loss2 = []
        pos_no_loss2 = []
        no_neg_loss2 = []

        i_indices = []
        j_indices = []
        ys = []
        all_nodes_set = set()
        skipped_nodes = []
        for i in center_nodes:
            # if no links then we can ignore
            if (len(adj_lists_pos[i]) + len(adj_lists_neg[i])) == 0:
                skipped_nodes.append(i)
                continue
            all_nodes_set.add(i)
            for j_pos in adj_lists_pos[i]:
                i_loss2.append(i)
                pos_no_loss2.append(j_pos)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                no_neg_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_pos)
                ys.append(0)
                all_nodes_set.add(j_pos)
            for j_neg in adj_lists_neg[i]:
                i_loss2.append(i)
                no_neg_loss2.append(j_neg)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                pos_no_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_neg)
                ys.append(1)
                all_nodes_set.add(j_neg)

            need_samples = 2  # number of sampling of the no links pairs
            cur_samples = 0
            while cur_samples < need_samples:
                temp_samp = randint(0, max_node_index)
                if (temp_samp not in adj_lists_pos[i]) and (temp_samp not in adj_lists_neg[i]):
                    # got one we can use
                    i_indices.append(i)
                    j_indices.append(temp_samp)
                    ys.append(2)
                    all_nodes_set.add(temp_samp)
                cur_samples += 1

        all_nodes_list = list(all_nodes_set)
        all_nodes_map = {node: i for i, node in enumerate(all_nodes_list)}
        final_embedding = self.forward(all_nodes_list)

        i_indices_mapped = [all_nodes_map[i] for i in i_indices]
        j_indices_mapped = [all_nodes_map[j] for j in j_indices]
        ys = torch.LongTensor(ys)
        if self.cuda_available:
            ys = ys.cuda()

        # now that we have the mapped indices and final embeddings we can get the loss
        loss_entropy = self.CrossEntLoss(
            torch.mm(torch.cat((final_embedding[i_indices_mapped],
                                final_embedding[j_indices_mapped]), 1),
                     self.param_src),
            ys)

        i_loss2 = [all_nodes_map[i] for i in i_loss2]
        pos_no_loss2 = [all_nodes_map[i] for i in pos_no_loss2]
        no_neg_loss2 = [all_nodes_map[i] for i in no_neg_loss2]

        tensor_zeros = torch.zeros(len(i_loss2))
        if self.cuda_available:
            tensor_zeros = tensor_zeros.cuda()

        loss_structure = torch.mean(
            torch.max(
                tensor_zeros,
                self.structural_distance(final_embedding[i_loss2], final_embedding[pos_no_loss2]) ** 2
                - self.structural_distance(final_embedding[i_loss2], final_embedding[no_neg_loss2]) ** 2
            )
        )

        return loss_entropy + self.lambda_structure * loss_structure

    def test_func(self, adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg):
        all_nodes_list = list(range(self.num_nodes))
        # no map necessary for ids as we are using all nodes
        final_embedding = self.forward(all_nodes_list)
        if self.cuda_available:
            final_embedding = final_embedding.detach().cpu().numpy()
        else:
            final_embedding = final_embedding.detach().numpy()
        # training dataset
        X_train = []
        y_train = []
        X_val = []
        y_test_true = []
        for i in range(self.num_nodes):
            for j in adj_lists_pos[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(1)

            for j in adj_lists_neg[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(-1)

            for j in test_adj_lists_pos[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(1)

            for j in test_adj_lists_neg[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(-1)

        y_train = np.asarray(y_train)
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        y_test_true = np.asarray(y_test_true)
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_val)

        auc = roc_auc_score(y_test_true, y_test_pred)
        f1 = f1_score(y_test_true, y_test_pred)

        return auc, f1
