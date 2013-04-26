'''
Created on 26 Apr 2013

@author: 16033493
'''
import pickle
import utils
import numpy as np
import gmm
from classifiers import optcriterion
import matplotlib.pyplot as pl

def snll(data, nums, means, covs):
    return gmm.calcresps(data, nums, means, covs, True)[1]

def normalize(dic):
    data = list()
    Nk = list()
    for v in dic['train']:
        N = 0
        for s in dic['train'][v]:
            data = data + list(s)
            N += s.shape[0]
        Nk.append(N)
    Nk = np.array(Nk,dtype=np.float)/np.sum(Nk)
    data = np.array(data)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)

    for x in dic:
        for v in dic[x]:
            for i in xrange(len(dic[x][v])):
                dic[x][v][i] = ((dic[x][v][i] - mean)/std).T

def pack(x):
    l = list()
    for p in x:
        l += list(p.T)
    return np.array(l).T

def doit(dic,classes,K,diag):
    err = {'train':list(), 'test':list()}
    for k in K:
        nums, means, covs, nll  = {},{},{},{}
        numsd, meansd, covsd, nlld  = {},{},{},{}
        # Build GMM models
        for dif in dic['train']:
            data = pack(dic['train'][dif])
            for i in xrange(4):
                _nums,_means,_covs,_nll = gmm.gmm(data, weights=None, K=k, hard=True, diagcov=diag)
                if(i != 0):
                    if(_nll > nll[dif]):
                        continue
                nums[dif],means[dif],covs[dif],nll[dif] =  _nums,_means,_covs,_nll
        
        criteria = [snll for dif in dic['train']]
        kwparams = [{'nums':nums[dif], 'means':means[dif], 'covs':covs[dif]} for dif in dic['train']]
        
        # Evaluate
        for x in dic:
            labels, labels_est = [], []
            for dif in dic[x]:
                points = dic[x][dif]
                labels += [dif for i in xrange(len(points))]
                labels_est += optcriterion(points, classes, criteria, kwparams=kwparams, _max=False);
            e = 100.0*sum( np.array(labels) != np.array(labels_est) ) / len(labels)
            err[x].append( e )
            
            print 'Confusion marix for' , x , 'data'
            utils.confusion(labels, labels_est, True)
            print '% Error: ', e,'\n'
    
    pl.plot(K,err['train'])

if __name__ == '__main__':
    K = range(1,6+1)
    dic = pickle.load(open('../data/speech.dat','r'))
    normalize(dic)
    classes = dic['train'].keys()
    
    doit(dic, classes, K, False)
    doit(dic, classes, K, True)
    
    pl.show()
    pass
    
    
    