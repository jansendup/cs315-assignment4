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

def buildPoints(data):
    return [np.atleast_2d(d).T for d in data.T]

def plotResults(dataTrain, dataTest, classes, labelsTrain, labelsTest, labelsMLTrain, labelsMLTest, title):
    #Split data by to labels
    trainSplit = splitClasses( dataTrain, classes, labelsTrain)
    testSplit = splitClasses( dataTest, classes, labelsTest)
    MLTrainSplit = splitClasses( dataTrain, classes, labelsMLTrain)
    MLTestSplit = splitClasses( dataTest, classes, labelsMLTest)
    
    # Calculate misclassified points
    missTrain = dataTrain[:,np.where(labelsTrain != labelsMLTrain)[0]]
    missTest = dataTest[:,np.where(labelsTest != labelsMLTest)[0]]
    if missTrain.size != 0 and missTest.size != 0:
        miss = np.concatenate([missTrain, missTest], axis=1)
    elif missTrain.size == 0 and missTest.size != 0:
        miss = missTest
    elif missTrain.size != 0 and missTest.size == 0:
        miss = missTrain
    else:
        miss = None
    
    # Render figure
    pl.figure(figsize=(12, 8))
    ax = pl.subplot(111)
    pl.hold(True)
    for i in xrange(classes.size):
        cc = {'c':CLASS_COLORS[i], 'marker':CLASS_MARKER[i], 'ls':' '}
        
        d = MLTrainSplit[i]
        pl.plot(d[0,:],d[1,:],alpha=0.4, ms=10, **cc)
        
        d = MLTestSplit[i]
        pl.plot(d[0,:],d[1,:],label='Class {:.0g} (ML)'.format(classes[i]),alpha=0.4, ms=10, **cc)
        
        d = trainSplit[i]
        pl.plot(d[0,:],d[1,:],label='Class {:.0g} (Train)'.format(classes[i]), **cc)
        d = np.mean(d,axis=1)
        pl.plot(d[0],d[1],label='Class {:.0g} (Mean)'.format(classes[i]), c=CLASS_COLORS[i], marker='x', ls=' ', ms=16,mew=2.0)
        
        d = testSplit[i]
        pl.plot(d[0,:],d[1,:],label='Class {:.0g} (Test)'.format(classes[i]),fillstyle='none',mew=2.0, **cc)
        
    
    pl.plot(miss[0,:],miss[1,:],'kx',label='Misclassified',ms=12)
    # Shink current axis's width by 30%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0,
                     box.width * 0.7, box.height])
    
    # Put a legend right of axis
    ax.legend(loc=2, bbox_to_anchor=(1.02, 1),
              fancybox=True, shadow=True)
    
    
    pl.title(title)
    pl.show()

def weight2col(weights):
    c = 0.05
    return 1.0/(1.0 + ((1-weights)/weights)**c)

def plotData(x,y,weights):
    col = weight2col(weights);
    pl.scatter(x,y,c=col,s=50);

if __name__ == '__main__':
    
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
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=False)
    
    nums, means, covs, nll = gmm.gmm(data, weights, K=3, hard=False, diagcov=False)
    numsd,meansd,covsd,nlld = gmm.gmm(data, weights, K=3, hard=False, diagcov=True)
    
    print 'NLL for non-diagonal: ', nll
    print 'NLL for diagonal:     ', nlld
    
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=False)

    pl.figure()
    
    pl.subplot(1,2,1)
    plotData(x,y,weights)
    
    nums, means, covs, nll = gmm.gmm(data, None, K=3, hard=False, diagcov=False)
    weights,nll = gmm.calcresps(data, nums, means, covs, hard=False)
    
    pl.subplot(1,2,2)
    plotData(x,y,weights)
    pl.show()
    