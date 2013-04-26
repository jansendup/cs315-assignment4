'''
Created on 26 Apr 2013

@author: 16033493
'''
import pickle
import utils
import numpy as np
import gmm
from classifiers import optcriterion

def snll(data, mean, cov, logprior):
    nll = gmm.calcresps(data, np.atleast_1d(np.array(data.shape[1])), mean, np.array([cov]), hard=True)[1]
    return nll - logprior

def normalize(dic):
    data = list()
    for v in dic:
        for s in dic[v]:
            data = data + list(s)
    data = np.array(data)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    
    points = list()
    labels = list()
    for v in dic:
        for s in dic[v]:
            points.append(((s - mean)/std).T)
            labels.append(v)
    return points,labels

if __name__ == '__main__':
    dic = pickle.load(open('../data/speech.dat','r'))
    trainpoints, trainlabels = normalize(dic['train'])
    testpoints, testlabels = normalize(dic['test'])
    classes = np.unique(trainlabels)
    
    traindata = list()
    for p in trainpoints:
        traindata = traindata + list(p.T)
    traindata = np.array(traindata).T
    
    
    K = len(dic['train'])

    bnll = 1e80
    for i in xrange(20):
        nums, means, covs, nll = gmm.gmm(traindata, None, K=K, hard=True, diagcov=False)
        if(nll < bnll):
            bnums, bmeans, bcovs, bnll = nums, means, covs, nll
    print bnll
    weights,nll = gmm.calcresps(traindata, bnums, bmeans, bcovs, hard=True)
    
    criteria = [snll for i in xrange(K)]
    kwparams = [{'mean':np.atleast_2d(bmeans[:,i]).T, 'cov':bcovs[i], 'logprior':np.log(bnums[i]/np.sum(bnums))} for i in xrange(K)]
    labels = optcriterion(trainpoints, classes, criteria, kwparams=kwparams, _max=False)
    utils.confusion(trainlabels, labels, True)
    
    
    
    