'''
    Joel Pfeiffer
    jpfeiffe@gmail.com
'''
import numpy as np
import numpy.random as rnd
import pandas
from scipy.sparse import csr_matrix, bsr_matrix

def random_edge_generator(labels, sparse = True, **kwargs):
    '''
    Generates a network given a set of labels.  If the labels are equal, samples from a normal
    with one std, and if they aren't equal it samples from another.  Sparsity sets the cutoff point.

    :y: Labels to utilize
    :mu_match: Normal mean for matching labels
    :mu_nomatch: Normal mean for non-matching labels
    :std: standard deviation of the normals
    :sparsity: Fraction of non-zero edges
    :symmetric: Is the returned matrix symmetric
    :remove_loops: Take out self loops (generally good)
    '''

    # Build matrix of mu values
    matrix = np.zeros((labels.shape[0],labels.shape[0]))
    matrix[np.random.rand(labels.shape[0],labels.shape[0]) > .5] = 1
    if sparse:
        matrix = csr_matrix(matrix.astype(np.int64))
        return matrix

    return matrix
