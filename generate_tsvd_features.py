#!/usr/bin/env python
# coding:utf-8


import pickle
import scipy.sparse as sps
import sys
import os


def read_in_undirected_graph(file_name):
    with open(file_name) as fp:
        n, m = [int(val) for val in fp.readline().split()[-2:]]
        A = sps.dok_matrix((n, n), dtype=float)
        for l in fp:
            i, j, s = [int(val) for val in l.split()]
            A[i, j] = s
            A[j, i] = s
    A = A.asformat('csr')
    return A


def tsvd(A, dim):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=dim, n_iter=30, random_state=42)
    svd.fit(A)
    X = svd.components_.T
    return X


if __name__ == "__main__":
    training_network_file_name = sys.argv[1]
    saved_path = sys.argv[2]
    k_to_keep = int(sys.argv[3])  # number of features
    A = read_in_undirected_graph(training_network_file_name)
    vec = tsvd(A, dim=k_to_keep)

    filename = os.path.split(training_network_file_name)[-1].split('.')[0]  # To obtain the name of input network.
    output = os.path.join(saved_path, '{}_features{}_tsvd.pkl'.format(filename, k_to_keep))
    print('Wrote the file to: {}'.format(output))
    with open(output, "wb") as fp:
        pickle.dump(vec, fp)
