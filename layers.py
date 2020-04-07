################################################################################
# SNEA/layers.py
# Used to define the attention mechanism for SNEA.
# Note: This is based on the GAT Implementation.
################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, cuda_available=False):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cuda_available = cuda_available

        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        # nn.init.xavier_normal_(self.W.data, gain=1.414)
        nn.init.xavier_normal_(self.W.data, gain=1)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * self.out_features)))
        # nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.act_func = F.tanh
        self.special_spmm = SpecialSpmm()

    def forward(self, _input1, adj1, _input2=None, adj2=None, adj_self=None, shape=None):
        if shape is None:
            N = _input1.size()[0]
            M = N
        else:
            N, M = shape

        if _input2 is not None and adj2 is not None:
            edge1 = adj1
            edge2 = adj2
            num_pos = edge1.size()[1]
            num_neg = edge2.size()[1]
            edge = torch.cat((edge1, edge2), dim=1)
            h1 = torch.mm(_input1, self.W)
            h2 = torch.mm(_input2, self.W)
            assert not torch.isnan(h1).any()
            assert not torch.isnan(h2).any()
            edge_h1 = torch.cat((h1[edge1[0, :], :], h1[edge1[1, :], :]), dim=1)
            edge_h2 = torch.cat((h2[edge2[0, :], :], h2[edge2[1, :], :]), dim=1)
            edge_h = torch.cat((edge_h1, edge_h2), dim=0).t()
        else:
            edge = adj1
            h = torch.mm(_input1, self.W)
            assert not torch.isnan(h).any()
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()

        edge_e = torch.exp(-self.act_func(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        tensor_ones = torch.ones(size=(M, 1))
        if self.cuda_available:
            tensor_ones = tensor_ones.cuda()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, M]), tensor_ones) + 1e-8

        if _input2 is not None and adj2 is not None:
            h_prime1 = self.special_spmm(edge1, edge_e[:num_pos], torch.Size([N, M]), h1)
            h_prime2 = self.special_spmm(edge2, edge_e[-num_neg:], torch.Size([N, M]), h2)
            h_prime = h_prime1 + h_prime2
        else:
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, M]), h)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        return h_prime  # return the result without non-linear transformation
