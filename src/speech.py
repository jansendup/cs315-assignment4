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

def snll(data, nums, means, covs, prior):
    return gmm.calcresps(data, nums, means, covs, True)[1] - np.log(prior)

def normalize(dic):
    data = list()
    
    for v in dic['train']:
        for s in dic['train'][v]:
            data = data + list(s)
            
    data = np.array(data)
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)

    for x in dic:
        for v in dic[x]:
            for i in xrange(len(dic[x][v])):
                dic[x][v][i] = ((dic[x][v][i] - mean)/std).T
def priors(dic):
    pr = {}
    N = 0.0
    for v in dic:
        pr[v] = 0.0
        for s in dic[v]:
            pr[v] += s.shape[0]
        N += pr[v]
    for v in dic:
        pr[v]/=N
    return pr
     
def pack(x):
    l = list()
    for p in x:
        l += list(p.T)
    return np.array(l).T

def doit(dic,priors,classes,K,diag):
    err = {'train':list(), 'test':list()}
    for k in K:
        nums, means, covs, nll  = {},{},{},{}
        # Build GMM models
        for dif in dic['train']:
            data = pack(dic['train'][dif])
            for i in xrange(6):
                _nums,_means,_covs,_nll = gmm.gmm(data, weights=None, K=k, hard=True, diagcov=diag)
                if(i != 0):
                    if(_nll > nll[dif]):
                        continue
                nums[dif],means[dif],covs[dif],nll[dif] =  _nums,_means,_covs,_nll
        
        criteria = [snll for dif in dic['train']]
        kwparams = [{'nums':nums[dif], 'means':means[dif], 'covs':covs[dif], 'prior':priors[dif]} for dif in dic['train']]
        
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
    
    pl.plot(K,err['train'],'--', label= 'Train'+(' (Diag)' if diag else ''))
    pl.plot(K,err['test'], label= 'Test'+(' (Diag)' if diag else ''))

if __name__ == '__main__':
    K = range(1,6+1)
    dic = pickle.load(open('../data/speech.dat','r'))
    normalize(dic)
    priors = priors(dic['test'])
    classes = dic['train'].keys()
    
    pl.figure()
    pl.hold(True)
    doit(dic, priors, classes, K, False)
    doit(dic, priors, classes, K, True)
    pl.legend(loc='best')
    pl.title('Classification error rate vs K')
    pl.xlabel('K')
    pl.ylabel('% Error')
    pl.show()
    
    pass
    
    
    