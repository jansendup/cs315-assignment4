'''
Created on Mar 20, 2013

@author: Jansen
'''
import numpy as np
import os
import utils
import matplotlib.pyplot as pl
import gmm
import dimred
import density

CLASS_COLORS = ['r','g','b']
CLASS_MARKER = ['s','o','D']

def readData(file):
    data = np.genfromtxt(file,delimiter=',').T
    labels = data[data.shape[0]-1,:]
    data = data[0:data.shape[0]-1,:]
    classes = np.unique(labels)
    return data,classes,labels

def splitClasses(data,classes,labels):
    return [ data[:,np.where(labels == c)[0]] for c in classes ]

def weight2col(weights):
    c = 0.05
    weights[np.where(weights < 1e-8)] = 1e-8
    return 1.0/(1.0 + ((1-weights)/weights)**c)

def plotData(x,y,weights):
    col = weight2col(weights);
    pl.scatter(x,y,c=col,s=50);

if __name__ == '__main__':
    hard = False
    # Load data
    data, classes, labels = readData(os.path.join('..','data','wine.data'))
    
    lda = dimred.LDA(data, classes, labels, center=True, scale=True)
    projData = lda.transform(data)
    x = projData[0,:]
    y = projData[1,:]
    
    points = splitClasses(data,classes,labels)
    gaussians = [density.Gaussian(data=p) for p in points]
    means = np.array([g.mean() for g in gaussians]).T
    covs = np.array([g.cov() for g in gaussians])
    nums = np.array([ p.shape[1] for p in points ])
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=hard)
    
    nums, means, covs, nll = gmm.gmm(data, weights, K=3, hard=hard, diagcov=False)
    numsd,meansd,covsd,nlld = gmm.gmm(data, weights, K=3, hard=hard, diagcov=True)
    
    print 'NLL for non-diagonal: ', nll
    print 'NLL for diagonal:     ', nlld
    
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=hard)

    pl.figure()
    
    pl.subplot(1,2,1)
    plotData(x,y,weights)
    
    nums, means, covs, nll = gmm.gmm(data, None, K=3, hard=hard, diagcov=False)
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=hard)
    
    pl.subplot(1,2,2)
    plotData(x,y,weights)
    pl.show()
    