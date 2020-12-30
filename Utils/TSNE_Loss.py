# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:26:52 2020

@author: jpeeples
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb
from geomloss import SamplesLoss


class TSNE_Loss(nn.Module):

    def __init__(self, reduction='mean', device='cuda:0', dof=1, alpha=1, perplexity=30, loss_metric='Renyi'):

        # inherit nn.module
        super(TSNE_Loss, self).__init__()

        # Set aggregation of loss, device and degrees of freedom
        self.reduction = reduction
        self.device = device
        self.dof = dof
        self.alpha = alpha
        self.perplexity = perplexity
        self.loss_metric = loss_metric

    def forward(self, input_features, embedding):
        
        #Compute probabilities from original features (this should be target)
        prob_matrix = self.compute_joint_probabilities(input_features.detach().cpu().numpy(),
                                                       perplexity=self.perplexity)
        
        #Convert P to tensor. Should update model without pairwise similarity 
        #in computational graph (defaults to false)
        prob_matrix = torch.from_numpy(prob_matrix).to(self.device)

        # Compute Renyi divergence
        if self.loss_metric == 'EMD':
            loss = self.emd_loss(prob_matrix, embedding)
        else:
            loss = self.tsne_loss(prob_matrix, embedding)

        return loss

    def tsne_loss(self, P, activations):
        n = activations.size(0)
        eps = 1e-15
        sum_act = torch.sum(torch.pow(activations, 2), 1)
        Q = sum_act + sum_act.view([-1, 1]) - 2 * torch.matmul(activations, torch.transpose(activations, 0, 1))
        Q = Q / self.dof
        Q = torch.pow(1 + Q, -(self.dof + 1) / 2)
        Q = Q * torch.from_numpy(np.ones((n, n)) - np.eye(n)).to(self.device)  # Zero out diagonal
        Q = Q / torch.sum(Q)
        if self.alpha == 0:
            C = -1 * torch.log(Q[torch.where(P > 0)] + eps)
            if self.reduction == 'sum':
                C = torch.sum(C)
            elif self.reduction == 'mean':
                C = torch.mean(C)
        elif self.alpha == .5:
            C = torch.pow((P + eps) * (Q + eps), 1 / 2)
            if self.reduction == 'sum':
                C = -2 * torch.log(torch.sum(C))
            elif self.reduction == 'mean':
                C = -2 * torch.log(torch.mean(C))
        elif self.alpha == 1:  # KL divergence
            C = torch.log((P + eps) / (Q + eps))
            if self.reduction == 'sum':
                C = torch.sum(P * C)
            elif self.reduction == 'mean':
                C = torch.mean(P * C)
        elif self.alpha == 2:
            C = torch.log(torch.mean((P+eps)/ (Q+eps)))
            if self.reduction == 'sum':
                C = torch.sum(C)
            elif self.reduction == 'mean':
                C = torch.mean(C)
        else:
            C = (torch.pow(P + eps, self.alpha)) / (torch.pow(Q + eps, self.alpha - 1))
            if self.reduction == 'sum':
                C = (1 / (self.alpha - 1)) * torch.log(torch.sum(C))
            elif self.reduction == 'mean':
                C = (1 / (self.alpha - 1)) * torch.log(torch.mean(C))
        #   else:
        #       assert 'Reduction not supported, please use mean or sum'
        return C

    def emd_loss(self, P, activations):
        n = activations.size(0)
        eps = .01
        sum_act = torch.sum(torch.pow(activations, 2), 1)
        Q = sum_act + sum_act.view([-1, 1]) - 2 * torch.matmul(activations, torch.transpose(activations, 0, 1))
        Q = Q / self.dof
        Q = torch.pow(1 + Q, -(self.dof + 1) / 2)
        Q = Q * torch.from_numpy(np.ones((n, n)) - np.eye(n)).to(self.device)  # Zero out diagonal
        Q = Q / torch.sum(Q)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        C = loss(P,Q)
        #C = self.sinkhorn_loss(P,Q,eps,len(P),1)
        return C

    def Hbeta(self,D, beta):
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
        P = P / sumP
        return H, P
    
    def x2p(self,X, u=15, tol=1e-4, print_iter=2500, max_tries=50, verbose=0):
        # Initialize some variables
        n = X.shape[0]                     # number of instances
        P = np.zeros((n, n))               # empty probability matrix
        beta = np.ones(n)*.01              # empty precision vector (need smaller intial value or use cosine distance)
        logU = np.log(u)                   # log of perplexity (= entropy)
        
        # Compute pairwise distances
        if verbose > 0: print('Computing pairwise distances...')
        sum_X = np.sum(np.square(X), axis=1)
        # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
        D = sum_X + sum_X[:,None] + -2 * X.dot(X.T)
        
        # Run over all datapoints
        if verbose > 0: print('Computing P-values...')
        for i in range(n):
            
            if verbose > 1 and print_iter and i % print_iter == 0:
                print('Computed P-values {} of {} datapoints...'.format(i, n))
            
            # Set minimum and maximum values for precision
            betamin = float('-inf')
            betamax = float('+inf')
            
            # Compute the Gaussian kernel and entropy for the current precision
            indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
            Di = D[i, indices] #ignores pii
            H, thisP = self.Hbeta(Di, beta[i])
            
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while abs(Hdiff) > tol and tries < max_tries:
                
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i]
                    if np.isinf(betamax):
                        beta[i] *= 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i]
                    if np.isinf(betamin):
                        beta[i] /= 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2
                
                # Recompute the values
                H, thisP = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1
            
            # Set the final row of P
            P[i, indices] = thisP
            
        if verbose > 0: 
            print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
            print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
            print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))
        
        return P, beta
    
    def compute_joint_probabilities(self,features, perplexity=30, tol=1e-5, verbose=0):
        
        # Compute joint probabilities for features (do they set diagonal to zero? pii=0)
        if verbose > 0: print('Precomputing P-values...')  
        P, beta = self.x2p(features, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
        P[np.isnan(P)] = 0                              # make sure we don't have NaN's
        P = (P + P.T) # / 2                             # make symmetric (need /2? not really since normalized by sum)
        P = P / P.sum()                                 # obtain estimation of joint probabilities
        P = np.maximum(P, np.finfo(P.dtype).eps)
    
        return P

