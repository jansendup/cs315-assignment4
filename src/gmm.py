'''
Created on Mar 28, 2012

@author: kroon
'''

from __future__ import division
from warnings import warn
import random
from heapq import heappush, heappop

import numpy as np
import numpy.testing as npt
import time

class EmptyComponentError(Exception):
    '''Error when number of points associated with a GMM/K-Means component reaches zero'''
    
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
     
def calcweights(data, means):
    '''Calculate the assignment of points to means for the current means in the K-means algorithm
    
    Assign each point in C{data} to the closest vector in C{means} and return this assignment in a matrix,
    using a one-hot encoding.

    >>> data = np.array(range(20)).reshape(4,5)
    >>> alloc, dist = calcweights(data,  np.array([[j + 5 * i for j in [1.3, 2.6]] for i in range(4)]))
    >>> npt.assert_almost_equal(alloc, np.reshape(np.array(2 * [1, 0] + 3 * [0, 1]), (5, 2)), decimal=4)
    >>> npt.assert_almost_equal(dist, 17.04, decimal=4)

    @param data: Data from which parameters are being estimated
    @type data: 2-dimensional NumPy array (d x N)
    @param means: Current means in model
    @type means: 2-dimensional NumPy array (d x K)
    
    @return: (one-hot encoding of calculated (point, mean) allocation, distortion for that allocation)
    @rtype: tuple (2-dimensional NumPy array (N x K), float)
    '''
    K = means.shape[1]
    N = data.shape[1]
    t = np.zeros((N, K))
    for k in xrange(K):
        t[:,k] = np.sum((data - np.atleast_2d(means[:,k]).T)**2,axis=0)
    i = np.argmin(t, axis=1)
    dist = np.sum(t[xrange(N),i])
    t[:,:] = 0;
    t[ xrange(N) , i ] = 1
    return t, dist

def calcresps(data, nums, means, covs, hard=True):
    '''Calculate responsibilities maximizing the likelihood of the given GMM parameters
    
    Calculate the responsibilities maximizing the likelihood of the given parameters in a GMM model.  The
    boolean keyword C{hard} indicates whether hard or soft allocation of points to components should be
    performed.  In the case of hard allocation, responsibilities are constrained to be either 0 or 1, with
    exactly one responsibility per data point set to 1.

    >>> data = np.array([[ 0.3323902 ,  1.39952168],
    ...        [-3.09393968,  0.85202915],
    ...        [ 0.3932616 ,  4.14018981],
    ...        [ 2.71301182,  1.48606545],
    ...        [ 0.76624929,  1.48450185],
    ...        [-2.68682389, -2.20487651],
    ...        [-1.50746076, -1.3965284 ],
    ...        [-3.35436652, -2.70017904],
    ...        [ 0.62831278, -0.14266899],
    ...        [-3.13713063, -1.35515249]])
    >>> data = np.transpose(data)
    >>> w = np.reshape(3 * [1, 0] + 4 * [0, 1] + 3 * [1, 0], (10, 2))
    >>> n = updatenums(w)
    >>> m = updatemeans(w, n, data)
    >>> c = updatecovs(w, m, n, data)
    >>> resps, nll = calcresps(data, n, m, c)
    >>> npt.assert_almost_equal(resps, np.array([[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0]]), decimal=4)
    >>> npt.assert_almost_equal(nll, 40.340684911249667, decimal=4)
    >>> resps, nll = calcresps(data, n, m, c, hard=False)
    >>> npt.assert_almost_equal(resps, np.array([[  7.79374979e-01,   2.20625021e-01], [  9.99999900e-01,   1.00345187e-07], [  1.00000000e+00,   2.30021208e-10], [  8.22976856e-02,   9.17702314e-01], [  5.53999927e-01,   4.46000073e-01], [  3.81391842e-01,   6.18608158e-01], [  3.16978845e-01,   6.83021155e-01], [  4.10097126e-01,   5.89902874e-01], [  2.13783567e-01,   7.86216433e-01], [  8.49852402e-01,   1.50147598e-01]]), decimal=4)
    >>> npt.assert_almost_equal(nll, 37.622423073925454, decimal=4)    

    @param data: Data from which parameters are being estimated
    @type data: 2-dimensional NumPy array (d x N)
    @param nums: Current effective number of points per component
    @type nums: 1-dimensional NumPy array (K)
    @param means: Current means in model
    @type means: 2-dimensional NumPy array (d x K)
    @param covs: Current covariance matrices in model
    @type covs: 3-dimensional NumPy array (K x d x d)
    @keyword hard: Indicates whether hard or soft allocation of points to components should be performed
    @type hard: boolean
    
    @return: The updated responsibilities; the negative log-likelihood associated with these responsibilities
    @rtype: tuple (2-dimensional NumPy array (N x K); floating-point value)
    '''
    K = means.shape[1]
    N = data.shape[1]
    R = np.zeros((N,K))
    
    Z = np.log((2*np.pi)**means.shape[0])
    
    for k in xrange(K):
        cov = covs[k]
        
        X = data - np.atleast_2d(means[:,k]).T
        ldet = np.linalg.slogdet(cov)[1]
        R[:,k] = nums[k] * np.exp( -0.5*(np.sum(X * np.linalg.solve(cov,X), axis=0) + Z + ldet) )
    
    if(hard):
        i = np.argmax(R,axis=1)
        S = R[xrange(N),i]
        R[:] = 0
        R[xrange(N),i] = 1
    else:
        S = np.sum(R,axis=1)
        R = (R.T/S).T
    
    nll = -np.sum( np.log(S) ) + N*np.log(N) 
    return R, nll

def updatenums(weights):
    '''Update (effective) number of points per component/mean in EM or K-means algorithm
    
    Update the effective number of points for each component of a GMM for C{data}, given the current responsibilities.
    Alternatively, calculate the number of points currently allocated to each mean in the K-means algorithm.
    If any of the values calculated are zero, an EmptyComponentError should be raised.

    >>> w = np.reshape(np.array(3 * [1, 0] + 2 * [0, 1]), (5, 2))
    >>> npt.assert_almost_equal(updatenums(w), np.array([3, 2]), decimal=4)
    >>> w = np.reshape(np.array(3 * [0.7, 0.3] + 2 * [0.2, 0.8]), (5, 2))
    >>> npt.assert_almost_equal(updatenums(w), np.array([2.5, 2.5]), decimal=4)

    @param weights: Current responsibilities in model
    @type weights: 2-dimensional NumPy array (N x K)
    
    @return: The calculated numbers of points
    @rtype: 1-dimensional NumPy array (K)
    '''
    s = np.sum(weights,axis=0);
    if ( s*s < 1e-6 ).any():
        raise EmptyComponentError();
    return s

def updatemeans(weights, nums, data):
    '''Update estimates of mean vectors in both EM and K-means algorithms
    
    Update the means in the K-means algorithms for the given allocation of points to means, or
    update the estimated means for each component of a GMM for C{data}, given the current responsibilities.
    
    >>> data = np.array(range(20)).reshape(4,5)
    >>> w = np.reshape(np.array(3 * [1, 0] + 2 * [0, 1]), (5, 2))
    >>> npt.assert_almost_equal(updatemeans(w, updatenums(w), data), np.array([[j + 5 * i for j in [1, 3.5]] for i in range(4)]), decimal=4)
    >>> w = np.reshape(np.array(3 * [0.7, 0.3] + 2 * [0.2, 0.8]), (5, 2))
    >>> npt.assert_almost_equal(updatemeans(w, updatenums(w), data), np.array([[j + 5 * i for j in [1.4, 2.6]] for i in range(4)]), decimal=4)

    @param weights: Current responsibilities/point allocations in algorithm
    @type weights: 2-dimensional NumPy array (N x K)
    @param nums: (Effective) number of points per component
    @type nums: 1-dimensional NumPy array (K)
    @param data: Data from which parameters are being estimated
    @type data: 2-dimensional NumPy array (d x N)
    
    @return: The estimated means
    @rtype: 2-dimensional NumPy array (d x K)
    '''
    return data.dot(weights)/nums

def updatecovs(weights, means, nums, data, diagcov=False):
    '''Update estimates of covariance matrices in EM algorithm
    
    Estimate the covariance matrices for each component of a GMM for C{data}, given the current responsibilities and means.
    If C{diagcov} is true, the returned matrices should be obtained via the usual update rule by setting the off-diagonal
    entries to zero. 

    >>> data = np.array([[ 0.3323902 ,  1.39952168],
    ...        [-3.09393968,  0.85202915],
    ...        [ 0.3932616 ,  4.14018981],
    ...        [ 2.71301182,  1.48606545],
    ...        [ 0.76624929,  1.48450185],
    ...        [-2.68682389, -2.20487651],
    ...        [-1.50746076, -1.3965284 ],
    ...        [-3.35436652, -2.70017904],
    ...        [ 0.62831278, -0.14266899],
    ...        [-3.13713063, -1.35515249]])
    >>> data = np.transpose(data)
    >>> w = np.reshape(3 * [1, 0] + 4 * [0, 1] + 3 * [1, 0], (10, 2))
    >>> n = updatenums(w)
    >>> m = updatemeans(w, n, data)
    >>> c = updatecovs(w, m, n, data)
    >>> npt.assert_almost_equal(c, np.array([[[ 3.33881418,  2.61431608], [ 2.61431608,  4.69524387]], [[ 4.32780425,  3.27144308], [ 3.27144308,  2.78110481]]]), decimal=4)
    >>> c = updatecovs(w, m, n, data, diagcov=True)
    >>> npt.assert_almost_equal(c, np.array([[[ 3.33881418,  0.], [ 0.,  4.69524387]], [[ 4.32780425,  0.], [ 0.,  2.78110481]]]), decimal=4)
    >>> w = np.reshape(3 * [0.7, 0.3] + 4 * [0.2, 0.8] + 3 * [0.8, 0.2], (10, 2))
    >>> n = updatenums(w)
    >>> m = updatemeans(w, n, data)
    >>> c = updatecovs(w, m, n, data)
    >>> npt.assert_almost_equal(c, np.array([[[ 3.70078013,  2.65039888], [ 2.65039888,  4.38797089]], [[ 4.23779562,  2.83814043], [ 2.83814043,  3.55031824]]]), decimal=4)
    
    @param weights: Current responsibilities in model
    @type weights: 2-dimensional NumPy array (N x K)
    @param means: Current means in model
    @type means: 2-dimensional NumPy array (d x K)
    @param nums: Effective number of points per component
    @type nums: 1-dimensional NumPy array (K)
    @param data: Data from which parameters are being estimated
    @type data: 2-dimensional NumPy array (d x N)
    @keyword diagcov: Indicates whether the estimated covariance matrices should be constrained to be diagonal
    @type diagcov: boolean
    
    @return: The estimated covariance matrices
    @rtype: 3-dimensional NumPy array (K x d x d)
    '''
    d = means.shape[0]
    K = means.shape[1]
    covs = np.zeros((K,d,d))
    for k in xrange(K):
        X = data.T - means[:,k]
        if(diagcov):
            covs[k,:,:][np.diag_indices(d)] = np.sum((X*X).T * weights[:,k],axis=1)/nums[k]
        else:
            covs[k,:,:] = (X.T*weights[:,k]).dot(X)/nums[k]
    return covs

def nubskmeans(data, K=1, reps=3):
    '''Perform the K-means algorithm using non-uniform binary splitting (NUBS) for initialization of the means
    
    Perform the K-means algorithm on the provided C{data}, using C{K} means, where means are initialized
    using NUBS.  For each binary split performed, perform C{reps} 2-class K-means fits, and use the
    one leading to the lowest distortion.
    
    @param data: The data used to calculate the means
    @type data: 2-dimensional NumPy array (d x N)
    @keyword K: Number of means to be used
    @type K: integer
    @keyword reps: number of 2-class K-means fits to perform per split
    @type reps: integer
    
    @return: (means returned by algorithm, allocation of points to means using one-hot encoding)
    @rtype: tuple (2-dimensional NumPy array (d x K), 2-dimensional NumPy array (N x K))
    ''' 

    def distortion(data):
        '''Returns the distortion of C{data}'''
        mean = np.mean(data, axis=1).reshape((-1,1))
        return np.sum((data - mean) ** 2)

    h = []
    heappush(h, (-distortion(data), data))
    for _ in range(1, K):
        _, tosplit = heappop(h)
        _, newweights = kmeans(tosplit, K=2)
        bestqual = float("inf")
        for _ in range(reps):
            data0 = tosplit[:, newweights[:, 0] == 1]
            dist0 = distortion(data0)
            data1 = tosplit[:, newweights[:, 1] == 1]
            dist1 = distortion(data1)
            if dist0 + dist1 < bestqual:
                bestqual = dist0 + dist1
                best0 = data0
                best1 = data1
                bestdist0 = dist0
                bestdist1 = dist1
        heappush(h, (-bestdist0, best0))
        heappush(h, (-bestdist1, best1))
    nubsmeans = np.transpose(np.array([np.mean(heappop(h)[1], axis=1) for i in range(len(h))]))
    return kmeans(data, K, means = nubsmeans)

    
def kmeans(data, K=1, means=None, maxiters=30, rtol=1e-4):
    '''Perform the K-means algorithm
    
    Perform the K-means algorithm on the provided C{data}, using C{K} means.  If C{means} is specified, use
    these points to initialize the means in the algorithm, and infer C{K}.  (If C{means} is specified, the
    value assigned to C{K} will be ignored.)  Otherwise, select the initial means by a random sample from
    the data points (without replacement).

    >>> data = np.array(range(20)).reshape(4,5)
    >>> means = data[ :, :2]
    >>> m = data[ :, [0,3]]
    >>> means, labels = kmeans(data, means=m)
    >>> npt.assert_almost_equal(means, np.array([[j + 5 * i for j in [0.5, 3]] for i in range(4)]), decimal=4)
    >>> m = data[ :, [1,4]]
    >>> means, labels = kmeans(data, means=m)
    >>> npt.assert_almost_equal(means, np.array([[j + 5 * i for j in [1, 3.5]] for i in range(4)]), decimal=4)
    
    @param data: The data used to calculate the means
    @type data: 2-dimensional NumPy array (d x N)
    @keyword K: Number of means to be used
    @type K: integer
    @keyword means: Points to use for initializing means
    @type means: 2-dimensional NumPy array (d x K)
    @keyword maxiters: The maximum number of iterations of the K-means loop to be performed.  If this is reached, a warning
    is printed, and the last values calculated are returned
    @type maxiters: integer
    @keyword rtol: Threshold on relative change in distortion over a K-means iteration below which convergence
    is assumed
    @type rtol: float
    
    @return: (means returned by algorithm, allocation of points to means using one-hot encoding)
    @rtype: tuple (2-dimensional NumPy array (d x K), 2-dimensional NumPy array (N x K))
    ''' 
    iters = 0
    if means is None:
        means = data[:, random.sample(xrange(data.shape[1]), K)]
    oldweights, olddistortion = calcweights(data, means)
    converged = False
    while not converged and iters <  maxiters:
        nums = updatenums(oldweights)
        means = updatemeans(oldweights, nums, data)
        newweights, newdistortion = calcweights(data, means)
        if (olddistortion - newdistortion) / olddistortion < rtol:
            converged = True
        oldweights, olddistortion = newweights, newdistortion
        iters += 1
    if iters >= maxiters:
        warn("Maximum number of iterations reached - kmeans may not have converged")
    return means, newweights

def gmm(data, weights=None, K=1, hard=True, diagcov=False, maxiters=20, rtol=1e-4):
    '''Perform parameter estimation for a Gaussian mixture model (GMM)
    
    Perform parameter estimation for a GMM of C{data} with C{K} components using
    the expectation-maximization (EM) algorithm.  Initial class responsibilities are specified in
    C{weights}.  If no initial class responsibilities are provided, values should be obtained by
    running the NUBS K-means algorithm on C{data}, using C{K} components.
    
    >>> # Data generated based on http://www.mathworks.com/help/toolbox/stats/bq_679x-24.html using
    >>> # np.random.seed(5)
    >>> # np.vstack([np.random.multivariate_normal([1, 2], [[3, .2], [.2, 2]], 5), np.random.multivariate_normal([-1, -2], [[2, 0], [0, 1]], 5)])
    >>> data = np.array([[ 0.3323902 ,  1.39952168],
    ...        [-3.09393968,  0.85202915],
    ...        [ 0.3932616 ,  4.14018981],
    ...        [ 2.71301182,  1.48606545],
    ...        [ 0.76624929,  1.48450185],
    ...        [-2.68682389, -2.20487651],
    ...        [-1.50746076, -1.3965284 ],
    ...        [-3.35436652, -2.70017904],
    ...        [ 0.62831278, -0.14266899],
    ...        [-3.13713063, -1.35515249]])
    >>> data = np.transpose(data)
    >>> n, m, c, nll = gmm(data, weights=np.reshape(3 * [1, 0] + 4 * [0, 1] + 3 * [1, 0], (10, 2)))
    >>> npt.assert_almost_equal(n, np.array([5, 5]), decimal=4)
    >>> npt.assert_almost_equal(m, np.array([[-0.94783384, -0.84146531], [1.304218 , -0.9916375 ]]), decimal=4)
    >>> npt.assert_almost_equal(c, np.array([[[3.15487645, 2.20538737], [2.20538737, 3.07220946]], [[4.9916252, 3.37132928], [3.37132928, 2.28295183]]]), decimal=4)
    >>> npt.assert_almost_equal(nll, 30.461793041351186, decimal=4)
    >>> n, m, c, nll = gmm(data, weights=np.reshape(3 * [1, 0] + 4 * [0, 1] + 3 * [1, 0], (10, 2)), diagcov=True)
    >>> npt.assert_almost_equal(n, np.array([8, 2]), decimal=4)
    >>> npt.assert_almost_equal(m, np.array([[-1.55321961,  1.73963056], [-0.1759581 ,  1.48528365]]), decimal=4)
    >>> npt.assert_almost_equal(c, np.array([[[  2.68965903e+00,   0.00000000e+00], [  0.00000000e+00,   4.44220348e+00]], [[  9.47471087e-01,   0.00000000e+00], [  0.00000000e+00,   6.11211240e-07]]]), decimal=4)
    >>> npt.assert_almost_equal(nll, 28.943272676612935, decimal=4)
    >>> n, m, c, nll = gmm(data, weights=np.reshape(3 * [1, 0] + 4 * [0, 1] + 3 * [1, 0], (10, 2)), hard=False)
    >>> npt.assert_almost_equal(n, np.array([5.11865545,  4.88134455]), decimal=4)
    >>> npt.assert_almost_equal(m, np.array([[-0.93462854, -0.85272701], [ 1.25697127, -0.99790135]]), decimal=4)
    >>> npt.assert_almost_equal(c, np.array([[[ 3.16380585,  2.17689271], [ 2.17689271,  3.12725162]], [[ 5.02927041,  3.39900947], [ 3.39900947,  2.30303275]]]), decimal=4)
    >>> npt.assert_almost_equal(nll, 30.358516118237446, decimal=4)
    >>> n, m, c, nll = gmm(data, weights=np.reshape(3 * [1, 0] + 4 * [0, 1] + 3 * [1, 0], (10, 2)), hard=False, diagcov=True)
    >>> npt.assert_almost_equal(n, np.array([ 4.99278276,  5.00721724]), decimal=4)
    >>> npt.assert_almost_equal(m, np.array([[-2.75771246,  0.96304258], [-1.36093232,  1.66913907]]), decimal=4)
    >>> npt.assert_almost_equal(c, np.array([[[ 0.43464687,  0.], [ 0.,  1.48290904]], [[ 0.79499714,  0.], [ 0.,  1.91644595]]]), decimal=4)
    >>> npt.assert_almost_equal(nll, 35.224438294366749, decimal=4)

    @param data: The data used to estimate the parameters
    @type data: 2-dimensional NumPy array (d x N)
    @keyword weights: Initial responsibilities for each (class, data point) combination
    @type weights: 2-dimensional NumPy array (N x K)
    @keyword K: Number of components for the GMM.  This should be ignored if C{weights} is specified (in which case,
    the number of components is implicitly specified
    @type K: integer
    @keyword hard: Indicates whether hard or soft allocation of data points to components should be performed
    @type hard: boolean
    @keyword diagcov: Indicates whether the covariance matrix should be restricted to be diagonal or not
    @type diagcov: boolean
    @keyword maxiters: The maximum number of iterations of the EM loop to be performed.  If this is reached, a warning
    is printed, and the parameters of the current model are returned
    @type maxiters: integer
    @keyword rtol: Threshold on relative change in negative log-likelihood (NLL) over an EM iteration below which convergence
    is assumed
    @type rtol: float
    
    @return: (effective number of points per component, means of each component, covariance matrices of each component,
    NLL of parameters returned )
    @rtype: tuple ( 1-dimensional NumPy array (K), 2-dimensional NumPy array (d x K), 3-dimensional NumPy array (K x d x d,
    float)
    ''' 
    if(weights == None):
        means, weights = nubskmeans(data, K, reps=3)
    
    nums = updatenums(weights)
    
    means = updatemeans(weights,nums,data)
    covs = updatecovs(weights, means, nums, data, diagcov)
    
    #if(hard and diagcov):
        #nums = np.array([8, 2])
        #means = np.array([[-1.55321961,  1.73963056], [-0.1759581 ,  1.48528365]])
        #covs = np.array([[[  2.68965903e+00,   0.00000000e+00], [  0.00000000e+00,   4.44220348e+00]], [[  9.47471087e-01,   0.00000000e+00], [  0.00000000e+00,   6.11211240e-07]]])
        
    
    nll_pre = 0
    for i in xrange(maxiters):
        weights,nll = calcresps(data, nums, means, covs, hard)
        
        covs = updatecovs(weights, means, nums, data, diagcov)
        means = updatemeans(weights,nums,data)
        nums = updatenums(weights)
        
        #if(hard and diagcov):
            #print nums, nll
        
        if( i != 0 ):
            if(np.abs((nll-nll_pre)/nll_pre) < rtol):
                break

        nll_pre = nll
        
    return nums,means,covs,nll

def _test():
    import doctest
    doctest.testmod()  
    

if __name__ == '__main__':
    _test()
