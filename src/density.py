'''Module containing a DensityFunc abstract class, with common probability densities

@since: Jan 10, 2013

@author: kroon
'''

from __future__ import division

import numpy as np
import random

class DensityFunc(object):
    '''
    Abstract class for representing the density function of a multi-dimensional continuous distribution
    of some dimension d.
    '''
    
    def f(self, x):
        '''
        Evaluate the probability density function (p.d.f.) at x
        
        @param x: point at which p.d.f. should be evaluated
        @type x: 1-dimensional NumPy array of floats (d)
        
        @return: the value of the p.d.f. at x
        @rtype: non-negative float
        '''
        raise NotImplementedError
    
    def logf(self, x):
        '''
        Evaluate the logarithm of the probability density function (p.d.f.) at x
        
        @param x: point at which the log-p.d.f. should be evaluated
        @type x: 1-dimensional NumPy array of floats (d)
        
        @return: the log of the value of the p.d.f. at x
        @rtype: float
        '''
        raise NotImplementedError
    
    def F(self, x):
        '''
        Evaluate the cumulative density function (c.d.f.) at x
        
        @param x: point at which c.d.f. should be evaluated
        @type x: 1-dimensional NumPy array of floats (d)
        
        @return: the value of the c.d.f. at x
        @rtype: non-negative float between 0 and 1
        '''
        raise NotImplementedError

    def mean(self):
        '''
        Return the mean of the distribution
        
        @return: the mean of the distribution
        @rtype: 1-dimensional NumPy array of floats (d)
        '''
        raise NotImplementedError

    def cov(self):
        '''
        Return the covariance matrix of the distribution
        
        @return: the covariance matrix of the distribution
        @rtype: 2-dimensional NumPy array of floats (d x d)
        '''
        raise NotImplementedError

    def precision(self):
        '''
        Return the precision matrix of the distribution
        
        @return: the precision matrix of the distribution
        @rtype: 2-dimensional NumPy array of floats (d x d)
        '''
        raise NotImplementedError

    def covdet(self):
        '''
        Return the determinant of the covariance matrix of the distribution.
        
        @return: the determinant of the covariance matrix of the distribution
        @rtype: non-negative float
        '''
        raise NotImplementedError

    def covlogdet(self):
        '''
        Return the log-determinant of the covariance matrix of the distribution.
        
        @return: the log-determinant of the covariance matrix of the distribution
        @rtype: float
        '''
        raise NotImplementedError

    def likelihood(self, x):
        '''
        Return the likelihood of the data set x for the distribution
        
        @param x: N d-dimensional points
        @type x: 2-dimensional NumPy array of floats (d x N)

        @return: the likelihood of x for this distribution
        @rtype: non-negative float
        '''
        raise NotImplementedError

    def loglik(self, x):
        '''
        Return the log-likelihood of the data set x for the distribution
        
        @param x: N d-dimensional points
        @type x: 2-dimensional NumPy array of floats (d x N)

        @return: the log-likelihood of x for this distribution
        @rtype: float
        '''
        raise NotImplementedError

    def negloglik(self, x):
        '''
        Return the negative log-likelihood of the data set x for the distribution
        
        @param x: N d-dimensional points
        @type x: 2-dimensional NumPy array of floats (d x N)

        @return: the negative log-likelihood of x for this distribution
        @rtype: float
        '''
        raise NotImplementedError

    def sample(self, n=1):
        '''
        Generate C{n} independent points sampled from the distribution.
        
        @param n: number of points to generate
        @type n: non-negative integer

        @return: N d-dimensional points
        @rtype: 2-dimensional NumPy array of floats (d x N)
        '''
        raise NotImplementedError
    
class Gaussian(DensityFunc):
    '''
    Class for representing a multi-dimensional Gaussian distribution of some dimension d.  Subclass of the
    abstract class DensityFunc, which has more extensive documentation of the various methods.
    
    >>> import numpy.testing as npt    
    >>> d = Gaussian()
    >>> npt.assert_almost_equal([0], d.mean(), decimal=4) 
    >>> npt.assert_almost_equal([[1]], d.cov(), decimal=4)
    >>> d = Gaussian(dim=2)
    >>> npt.assert_almost_equal(np.eye(2), d.precision(), decimal=4)
    >>> d1 = Gaussian(np.array([1, 1, 1]), np.diag(np.array([1., 2., 3.])))
    >>> npt.assert_almost_equal(9.22205733512, d1.negloglik(np.array([[1, 2], [2, 3], [1, 2]])), decimal=4)
    >>> d2 = Gaussian(mean=np.array([1, 1, 1]), precision=np.diag(1/np.array([1., 2., 3.])))
    >>> npt.assert_almost_equal(d1.negloglik(np.array([[1, 2], [2, 3], [1, 2]])), d2.negloglik(np.array([[1, 2], [2, 3], [1, 2]])), decimal=4)
    >>> npt.assert_almost_equal(1.791759469228055, d2.covlogdet(), decimal=4)
    >>> d = Gaussian(np.array([1, 1]), np.array([[2., 1.], [1., 2.]]))
    >>> npt.assert_almost_equal(3, d.covdet(), decimal=4)
    >>> npt.assert_almost_equal(0.065840735999, d.f(np.array([2, 2])), decimal=4)
    >>> npt.assert_almost_equal(0.065840735999, d.likelihood(np.transpose(np.array([[0, 0]]))), decimal=4)
    >>> npt.assert_almost_equal(-2.38718321074, d.logf(np.array([1, 1])), decimal=4)
    >>> d = Gaussian(np.array([0]), np.eye(1))
    >>> npt.assert_almost_equal(-5.75681559961, d.loglik(np.array([[1, 1, 2]])), decimal=4)
    >>> d = Gaussian(data=Gaussian(dim=3).sample(100000))
    >>> npt.assert_almost_equal(d.mean(), np.zeros(3), decimal=2)
    >>> npt.assert_almost_equal(d.cov(), np.eye(3), decimal=2)    
    '''
    
    def f(self, x):
        '''Return the density at x'''
        x_ = x - self.u
        return self.A*np.exp( -0.5*np.dot( x_, np.dot(self.prec,x_) ) )
    
    def logf(self, x):
        '''Return the log-density at x'''
        return np.log(self.f(x))
    
    def mean(self):
        '''Return the mean of the distribution'''
        return self.u

    def cov(self):
        '''
        Return the covariance matrix of the distribution.  Assumes the covariance matrix is invertible.
        '''
        return self.covar

    def precision(self):
        '''
        Return the precision matrix of the distribution.  Assumes the covariance matrix is invertible.
        '''
        return self.prec

    def covdet(self):
        '''
        Return the determinant of the covariance matrix of the distribution.  Assumes the covariance matrix is invertible.
        '''
        return np.linalg.det(self.covar)
            
    def covlogdet(self):
        '''
        Return the log-determinant of the covariance matrix of the distribution. Assumes the covariance matrix is invertible.
        '''
        return np.log(self.covdet())

    def likelihood(self, x):
        '''Return the likelihood of the data set x for the distribution'''
        l = 1.0
        for x_ in x.T:
            l *= self.f(x_)
        return l

    def loglik(self, x):
        '''Return the log-likelihood of the data set x for the distribution'''
        return np.log(self.likelihood(x))

    def negloglik(self, x):
        '''Return the negative log-likelihood of the data set x for the distribution'''
        return -self.loglik(x)

    def sample(self, n=1):
        '''Return n independent points sampled from the distribution'''
        if self.d > 1:
            return np.random.multivariate_normal(self.u, self.covar , n).T
        else:
            return np.sqrt(self.covar)*np.random.standard_normal(n) + self.u
    
    def _estimate(self, data):
        '''
        Return maximum likelihood estimates of the mean and covariance matrix from the data
        
        @param data: N d-dimensional points
        @type data: 2-dimensional NumPy array of floats (d x N)
        
        @return:
         -maximum likelihood estimate of the mean
         -maximum likelihood estimate of the covariance matrix
         -maximum likelihood estimate of the precision matrix
        @rtype: tuple
        '''
        c = np.cov(data, bias=1)
        return (np.mean(data,axis=1),c,np.linalg.inv(c))
    
    def _setCov(self,cov,precision=None):
        ''' Set covariance and precision class variables '''
        if (cov == None) and (precision == None):
            cov = np.eye(self.d)
            precision = np.eye(self.d)
        if cov != None:
            self.covar = np.atleast_2d(cov)
        if precision != None:
            self.prec = np.atleast_2d(precision)
        if cov == None:
            self.covar = np.linalg.inv(self.prec)
        if precision == None:
            self.prec = np.linalg.inv(self.covar)
            
    def _calcA(self):
        ''' Precalculate constant used for f()'''
        self.A =  1.0 / np.sqrt((2*np.pi)**self.d*self.covdet())
    
    def __init__(self, mean=None, cov=None, precision=None, dim=None, data=None):
        '''
        Initialize density variables necessary from the provided information, making default assumptions of
        zero mean, identity covariance matrix, and a dimension of one. It is assumed all data provided is
        mutually compatible and sensible; two examples: if C{cov}, and C{precision}, are supplied, it is
        assumed they are inverses of each other; if C{cov} is supplied, it is assumed to be positive definite.
        
        @keyword mean: mean of the Gaussian .  Default is the zero vector.
        @type mean: 1-dimensional NumPy array (d)
        @keyword cov: covariance of the Gaussian.  Default is the identity matrix.
        @type cov: 2-dimensional NumPy array (d x d)
        @keyword precision: precision matrix of the Gaussian.  Default is the identity matrix.
        @type precision: 2-dimensional NumPy array (d x d)
        @keyword dim: dimension of the Gaussian, i.e. d.
        @type dim: positive integer
        @keyword data: N samples from the Gaussian.  If provided, the mean and covariance matrix are estimated
        from the samples.
        @type data: 2-dimensional NumPy array (d x N)
        '''       
        #First, establish dimension
        if data != None:
            data = np.atleast_2d(data)
            self.d = data.shape[0]
            self.u, self.covar, self.prec = self._estimate(data)
        else:
            self.d = dim if (dim!=None) else np.array(cov).shape[0] if cov != None else np.array(precision).shape[0] if precision!=None else 1
            self.u = np.array(mean) if (mean != None) else np.zeros(self.d)
            self._setCov(cov, precision)
        self._calcA()
        
class DiagonalGaussian(Gaussian):
    '''
    Class for representing a multi-dimensional Gaussian distribution of some dimension d with a diagonal covariance
    matrix.  Subclass of the class Gaussian, which has more extensive documentation of some overridden methods.
    
    >>> import numpy.testing as npt    
    >>> d = DiagonalGaussian()
    >>> npt.assert_almost_equal([0], d.mean(), decimal=4) 
    >>> npt.assert_almost_equal([[1]], d.cov(), decimal=4)
    >>> d = DiagonalGaussian(dim=2)
    >>> npt.assert_almost_equal(np.eye(2), d.precision(), decimal=4)
    >>> d1 = DiagonalGaussian(np.array([1, 1, 1]), np.diag(np.array([1., 2., 3.])))
    >>> npt.assert_almost_equal(9.22205733512, d1.negloglik(np.array([[1, 2], [2, 3], [1, 2]])), decimal=4)
    >>> d2 = DiagonalGaussian(mean=np.array([1, 1, 1]), precision=np.diag(1/np.array([1., 2., 3.])))
    >>> npt.assert_almost_equal(d1.negloglik(np.array([[1, 2], [2, 3], [1, 2]])), d2.negloglik(np.array([[1, 2], [2, 3], [1, 2]])), decimal=4)
    >>> npt.assert_almost_equal(1.791759469228055, d2.covlogdet(), decimal=4)
    >>> d = DiagonalGaussian(np.array([0]), np.eye(1))
    >>> npt.assert_almost_equal(-5.75681559961, d.loglik(np.array([[1, 1, 2]])), decimal=4)
    >>> d = DiagonalGaussian(data=DiagonalGaussian(dim=3).sample(100000))
    >>> npt.assert_almost_equal(d.mean(), np.zeros(3), decimal=2)
    >>> npt.assert_almost_equal(d.cov(), np.eye(3), decimal=2)
    >>> d = DiagonalGaussian(data=np.array([[1, 2, 12], [4, 5, 6]]))
    >>> npt.assert_almost_equal(np.diag([24.66666667,0.66666667]), d.cov(), decimal=4)
    >>> d=DiagonalGaussian(dim=100000)
    >>> npt.assert_almost_equal(1.66664166767e+14, d.negloglik(np.atleast_2d(range(100000)).T),decimal=-5)
    '''
    def f(self, x):
        '''Return the density at x'''
        np.e**self.logf(x)
    
    def logf(self, x):
        '''Return the log-density at x'''
        x_ = x - self.u
        return self.A - 0.5*x_.dot(self.prec*x_)
    
    def precision(self):
        '''
        Return the precision matrix of the distribution.  Assumes the covariance matrix is invertible.
        '''
        return np.diag(self.prec)
    
    def cov(self):
        '''
        Return the covariance matrix of the distribution.  Assumes the covariance matrix is invertible.
        '''
        return np.diag(self.covar)
    
    def covdet(self):
        '''
        Return the determinant of the covariance matrix of the distribution.  Assumes the covariance matrix is invertible.
        '''
        return np.prod(self.covar)
            
    def covlogdet(self):
        '''
        Return the log-determinant of the covariance matrix of the distribution. Assumes the covariance matrix is invertible.
        '''
        return np.sum(np.log(self.covar))

    def likelihood(self, x):
        '''Return the likelihood of the data set x for the distribution'''
        return np.e**self.loglik(x)

    def loglik(self, x):
        '''Return the log-likelihood of the data set x for the distribution'''
        return np.sum([self.logf(i) for i in x.T])
    
    def sample(self, n=1):
        '''Return n independent points sampled from the distribution'''
        return (np.sqrt(self.covar)*np.random.standard_normal((self.d,n)).T + self.u).T
    
    def _estimate(self, data):
        '''
        Return maximum likelihood estimates of the mean and covariance matrix from the data.  Since a diagonal
        covariance matrix is assumed, the diagonal elements of the matrix are estimated from each component
        independently.  
        '''
        m = np.mean(data,axis=1)
        c = np.var(data, axis=1)
        return (m,c,1.0/c)
    
    def _setCov(self,cov,precision=None):
        ''' Set covariance and precision class variables '''
        if (cov == None) and (precision == None):
            cov       = np.ones(self.d)
            precision = np.ones(self.d)
        if cov != None:
            self.covar = np.diagonal(cov) if cov.ndim > 1 else cov
        if precision != None:
            self.prec = np.diagonal(precision) if precision.ndim > 1 else precision
        if cov == None:
            self.covar = 1.0/self.prec
        if precision == None:
            self.prec = 1.0/self.covar
            
    def _calcA(self):
        ''' Precalculate constant used for f()'''
        self.A = -0.5*(np.log(2*np.pi)*self.d + self.covlogdet())
    
    def __init__(self, *args, **kwargs):
        '''
        Initialize density variables necessary from the provided information, making default assumptions of
        zero mean, identity covariance matrix, and a dimension of one.  This method takes the same parameters
        as the initializer of the superclass C{Gaussian}. It is assumed all parameters provided are
        mutually compatible and sensible, and does not contradict the covariance matrix being diagonal.  One
        exception: when C{data} is provided, the estimated covariance matrix is generated to be diagonal, rather
        than using the maximum likelihood estimate of the full covariance matrix.
        '''
        super( DiagonalGaussian, self ).__init__( *args, **kwargs )
    
def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
    
    print '\nChallenging task 4.'
    print 'I\'ve created a document showing my calculations for this task.'
    print 'The document is located at doc/challenge4.pdf'
    print 'The subject tests are:'
    print '1.) npt.assert_almost_equal(d.mean(), np.zeros(3), decimal=2)'
    print '2.) npt.assert_almost_equal(d.cov(), np.eye(3), decimal=2)'
    print 'For 1. I calculated the probability for failure as: 0.00468885915'
    print 'For 2. I calculated the probability for failure as: 0.07413078165'
    print 'Please see doc/challenge4.pdf for more details.'
    
    print '\nPlease note I have extended DiagonalGaussian and added the doctests to this file.'
    print 'Waring: Simulation took 5 min on my machine'
    yn = raw_input("Do you want to simulate tests to confirm my results (y/n): ")
    if 'y' in yn:
        meanWrong = 0
        covWrong = 0
        n = 5000
        for i in xrange(n):
            d = Gaussian(data=Gaussian(dim=3).sample(100000))
            if (np.absolute(d.mean()-np.zeros(3)) > 1e-2).any():
                meanWrong += 1
            if (np.absolute(d.cov()-np.eye(3)) > 1e-2).any():
                covWrong += 1
                
        print 'p(test1 fail) = ', meanWrong*1.0/n
        print 'p(test2 fail) = ', covWrong*1.0/n
        
