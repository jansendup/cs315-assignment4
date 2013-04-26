'''
Created on Mar 20, 2013

@author: Jansen
'''
import numpy as np
import os
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
    
    print 'Question 2'
    print 'NLL for diagcov=False: ', nll
    print 'NLL for diagcov=True:  ', nlld
    print 'The diagcov=False seems to be the better choice.'
    print 'Because we have enough data per class we can use a full estimate '
    print 'of the covariance matrix without over-fitting. '
    print 'These extra parameters allow us to make a better model which is '
    print 'evident in the differance in negative-log-likelihood(NNL).\n'
    
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=hard)

    pl.figure()
    
    pl.subplot(1,2,1)
    plotData(x,y,weights)
    pl.xlabel('$W_0^TX$')
    pl.ylabel('$W_1^TX$')
    pl.title('Initial weights by actual class label information')
    
    nums, means, covs, nll = gmm.gmm(data, None, K=3, hard=hard, diagcov=False)
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=hard)
    
    pl.subplot(1,2,2)
    plotData(x,y,weights)
    pl.xlabel('$W_0^TX$')
    pl.ylabel('$W_1^TX$')
    pl.title('Initial weights by nubskmeans')
    
    print 'Question 5'
    print '*'*40
    print 'K\tdiagcov=False\tdiagcov=True'
    print '*'*40
    for k in xrange(1,5):
            print k,'\t',gmm.gmm(data, None, K=k, hard=hard, diagcov=False)[3],'\t',gmm.gmm(data, None, K=k, hard=hard, diagcov=True)[3]
    
    print '\nIn this table we see that diagcov=False beats diagcov=True '
    print 'for all these cases. It is best at diagcov=False and K=3. '
    print 'This is what was expected since us humans classified 3 classes. '
    pl.show()
    