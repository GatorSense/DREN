# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:27:49 2020

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  Out_of_Sample.py
    *
    *  Desc:  This file contains code for out-of-sample embedding methods.
    *
    *  Written by:  Connor H. McCurley
    *
    *  Latest Revision:  2020-04-16
    *
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

# General packages
import numpy as np
import scipy.io
import random
import math
from itertools import combinations, combinations_with_replacement 
from scipy.spatial import distance_matrix
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph
from cvxopt import solvers, matrix

######################################################################
####################### Function Definitions #########################
######################################################################

def lse(A, b, B, d):
    """
    ******************************************************************
    Equality-contrained least squares.
    The following algorithm minimizes ||Ax - b|| subject to the
    constrain Bx = d.
    Parameters
    ----------
    A : array-like, shape=[m, n]
    B : array-like, shape=[p, n]
    b : array-like, shape=[m]
    d : array-like, shape=[p]
    Reference
    ---------
    Matrix Computations, Golub & van Loan, algorithm 12.1.2
    Examples
    --------
    >>> A = np.array([[0, 1], [2, 3], [3, 4.5]])
    >>> b = np.array([1, 1])
    >>> # equality constrain: ||x|| = 1.
    >>> B = np.ones((1, 3))
    >>> d = np.ones(1)
    >>> lse(A.T, b, B, d)
    array([-0.5,  3.5, -2. ])
    ******************************************************************
    """
    from scipy import linalg
    if not hasattr(linalg, 'solve_triangular'):
        # compatibility for old scipy
        solve_triangular = linalg.solve
    else:
        solve_triangular = linalg.solve_triangular
    A, b, B, d = map(np.asanyarray, (A, b, B, d))
    p = B.shape[0]
    Q, R = linalg.qr(B.T)
    y = solve_triangular(R[:p, :p].T, d)
    A = np.dot(A, Q)
    z = linalg.lstsq(A[:, p:], b - np.dot(A[:, :p], y))[0].ravel()
    return np.dot(Q[:, :p], y) + np.dot(Q[:, p:], z)


def unmix_cvxopt(data, endmembers, gammaConst=0, P=None):
    """
    ******************************************************************
    unmix finds an accurate estimation of the proportions of each endmember
    Syntax: P2 = unmix(data, endmembers, gammaConst, P)
    This product is Copyright (c) 2013 University of Missouri and University
    of Florida
    All rights reserved.
    CVXOPT package is used here. Parameters H,F,L,K,Aeq,beq are corresbonding to 
    P,q,G,h,A,B, respectively. lb and ub are element-wise bound constraints which 
    are added to matrix G and h respectively.
    
    Inputs:
    data            = DxN matrix of N data points of dimensionality D 
    endmembers      = DxM matrix of M endmembers with D spectral bands
    gammaConst      = Gamma Constant for SPT term
    P               = NxM matrix of abundances corresponding to N input pixels and M endmembers
    
    Returns:
    P2              = NxM matrix of new abundances corresponding to N input pixels and M endmembers

    ******************************************************************
    """

    solvers.options['show_progress'] = False
    X = data  
    M = endmembers.shape[1]  # number of endmembers # endmembers should be column vectors
    N = X.shape[1]  # number of pixels
    # Equation constraint Aeq*x = beq
    # All values must sum to 1 (X1+X2+...+XM = 1)
    Aeq = np.ones((1, M))
    beq = np.ones((1, 1))
     # Boundary Constraints ub >= x >= lb
    # All values must be greater than 0 (0 ? X1,0 ? X2,...,0 ? XM)
    lb = 0
    ub = 1
    g_lb = np.eye(M) * -1
    g_ub = np.eye(M)
    
    # import pdb; pdb.set_trace()

    G = np.concatenate((g_lb, g_ub), axis=0)
    h_lb = np.ones((M, 1)) * lb
    h_ub = np.ones((M, 1)) * ub
    h = np.concatenate((h_lb, h_ub), axis=0)

    if P is None:
        P = np.ones((M, 1)) / M

    gammaVecs = np.divide(gammaConst, sum(P))
    H = 2 * (endmembers.T @ endmembers)
    cvxarr = np.zeros((N,M))
    for i in range(N):
        F = ((np.transpose(-2 * X[:, i]) @ endmembers) + gammaVecs).T
        cvxopt_ans = solvers.qp(P=matrix(H.astype(np.double)), q=matrix(F.astype(np.double)), G=matrix(G.astype(np.double)), h=matrix(h.astype(np.double)), A=matrix(Aeq.astype(np.double)), b=matrix(beq.astype(np.double)))
        cvxarr[i, :] = np.array(cvxopt_ans['x']).T
    cvxarr[cvxarr < 0] = 0
    return cvxarr



def embed_out_of_sample(X_train, X_manifold, X_out, K, beta, neighbor_measure):
    """
    ******************************************************************
        *
        *  Func:    embed_out_of_sample(X_train, X_manifold, X_out, K, beta, neighbor_measure)
        *
        *  Desc:    Embeds out-of-sample points into lower-dimensional space.
        *           Uses a k-nearest neighbor, constrained least square reconstruction.
        *
        *  Inputs:
        *           X_train - NxD matrix of training data coordinates
        *
        *           X_manifold - NxK matrix of low-dimensional training data coordinates
        *
        *           X_out - MxD data matrix of out-of-sample points
        *
        *           K - dimensionality of embedding space
        *
        *           beta - bandwidth of RBf affinity function
        *
        *           neighbor_measure - number of neighbors to consider in k-NN graph
        *          
        *  Outputs:
        *           Z_out - MxK data matrix of embedded out of sample points
        * 
    ******************************************************************
    """
    
    print("\nEmbedding out of sample data...")
    
    ## Extract constants
    num_total = np.shape(X_train)[0] ## Number of training data points
    num_out_sample = np.shape(X_out)[0] ## Number of out-of-sample-data-points
    input_dim = np.shape(X_out)[1] ## Dimesnionality of input space
    
    Z_out = np.zeros((num_out_sample,K)) ## Initialize out of sample embedded coordinate matrix
    
    ##### Affinity of out-of-sample with training set #####
    print("Computing affinity matrices...")
    
    ## Define K-nearest neighbor graph
    W_L2 = distance_matrix(X_out, X_train, p=2)
    W_neighbors = W_L2
    
    ## Square L2 distances, divide by negative bandwidth and exponentiate
    W_total = np.exp((-1/beta)*(W_L2**2))
    print("Embedding out-of-sample points...")
    for idx in range(0,num_out_sample):
        temp_row = W_neighbors[idx, :]
        
        ## indicies of nearest neighbors according to L2 distance
        valid_ind = np.argpartition(temp_row, neighbor_measure) 
        
        ##### Find reconstruction weights of current out of sample NO bias ######
        X_recon = X_train[valid_ind[0:neighbor_measure],:].T
        x_current = X_out[idx,:]
        x_current= x_current.astype(np.double)
        X_recon - X_recon.astype(np.double)
        w_recon = unmix_cvxopt(np.expand_dims(x_current, axis=1), X_recon, gammaConst=0, P=None)
        w_recon = np.squeeze(w_recon)
        
        ## Embed sample as reconstruction of low-dimensional training data embeddings
        Z_recon = X_manifold[valid_ind[0:neighbor_measure],:].T
        z = np.dot(Z_recon, w_recon)
        
        Z_out[idx,:] = z
        
    print('Done!')
           
    return Z_out

