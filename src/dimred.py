'''Module containing classes for dimension reduction

B{Exported classes}
 - PCA - class for principal components analysis
 - LDA - class for linear discriminant analysis

@since: 6 Jan 2012

@author: skroon
'''

from __future__ import division
from warnings import warn

import numpy as np

import utils

class PCA:
    '''
    Class for performing principal component analysis.
    
    Initial part of example from http://courses.ee.sun.ac.za/Pattern_Recognition_813/lectures/lecture01/node5.html :
    
    >>> import numpy.testing as npt
    >>> data = np.array([[1, 2, 2, 1, 1, -2],
    ... [0, 1, 4, 2, -1, -2],
    ... [2, 4, 1, 2, 1, -2]])
    >>> result = PCA(data, center=True, scale=False)
    >>> components = np.array([[-0.49231217, 0.13911478, -0.85922977], [-0.65101487, -0.7140919, 0.25739542], [-0.57776151, 0.68609025, 0.44212193]])
    >>> npt.assert_almost_equal(result.components()*result.components(), components*components, decimal=4)
    >>> result.setn(2)
    >>> components = np.array([[-0.49231217, 0.13911478], [-0.65101487, -0.7140919 ], [-0.57776151, 0.68609025]])
    >>> npt.assert_almost_equal(result.components()*result.components(), components*components, decimal=4)
    >>> npt.assert_almost_equal(result.getprop(), 0.98145396394045681, decimal=4)
    >>> npt.assert_almost_equal(result.props, [ 0.76139975, 0.98145396, 1.], decimal=4)
    >>> result.setfrac(0.7)
    >>> npt.assert_almost_equal(result.getprop(), 0.76139975223017509, decimal=4)
    >>> npt.assert_almost_equal(result.transform(data), [[-0.03321646, -2.33206653, -2.55182659, -1.33524619, 1.19555992, 5.05679585]], decimal=4)
    >>> npt.assert_almost_equal(result.untransform(result.transform(data)), [[ 0.8496862, 1.98143808, 2.08962863, 1.49069129, 0.24474463, -1.65618883], [ 0.68829108, 2.18487665, 2.32794372, 1.53593179, -0.11166062, -2.62538261], [ 1.35252452, 2.68071162, 2.80768052, 2.10478719, 0.64258483, -1.58828868]], decimal=4)
    >>> result.setn(3)
    >>> npt.assert_almost_equal(np.cov(result.transform(data, whiten=True), bias=1), [[ 1, 0, 0], [0, 1, 0], [0, 0, 1]], decimal=4)
    >>> newdata = np.array([[3, 2, -1],
    ... [1, 0, 2],
    ... [2, -2, -2]])
    >>> transforms = np.array([[-1.66885568,  1.78551742,  1.9604242 ], [ 0.52077823, -1.64860567, -3.49413381], [-1.48111808, -2.64777143,  0.44470872]])
    >>> npt.assert_almost_equal(result.transform(newdata)*result.transform(newdata), transforms*transforms, decimal=4)
    >>> correct = np.array([[-0.64048822,  0.6852617 ,  0.75238898], [ 0.37178059, -1.17693013, -2.49444208], [-3.64219227, -6.51108967,  1.09357566]])
    >>> whitetrans = result.transform(newdata, whiten = True)
    >>> npt.assert_almost_equal(whitetrans*whitetrans, correct*correct, decimal=4)
    >>> data = np.cov(data, bias=1)
    >>> result = PCA(data, center=False, scale=False, cov = True)
    >>> correct = np.array([[-1.26016091, 0.06558901, 0.13271629], [ 0.76774541, -0.78096531, -2.09847726], [ 3.53136623,  6.40026362, -1.2044017 ]])
    >>> whitetrans = result.transform(newdata, whiten = True)
    >>> npt.assert_almost_equal(whitetrans*whitetrans, correct*correct, decimal=4)
           
    @ivar cov: Indicates whether passed data is a covariance matrix or not
    @type cov: boolean
    @ivar N: number of data points in A
    @type N: integer
    @ivar mean: mean of data points in A if centering enabled (d)
    @type mean: 1-dimensional NumPy array
    @ivar std: standard deviation of data points in A if scaling enabled (d)
    @type std: 1-dimensional NumPy array    
    @ivar U: eigenvectors of covariance matrix corresponding to elements of C{D} (d x C{r})
    @type U: 2-dimensional NumPy array
    @ivar D: non-zero eigenvalues of covariance matrix (in descending order) (C{r})
    @type D: 1-dimensional NumPy array
    @ivar r: total number of PCs
    @type r: integer
    @ivar props: NumPy array of length d giving cumulative proportions of variance
             explained by the first i PCs, i.e. props[i] is the cumulative proportion
             of variance explained by the first i+1 PCs
    @type props: 1-dimensional NumPy array
    @ivar n: current number of PCs being used
    @type n: integer
    '''
    def setn(self, x):
        '''
        Set the number of principal components in use to C{x}.
        
        @param x: the new number of components to use
        @type x: non-negative integer
        '''
        self.n = x;

    def setfrac(self, frac):
        '''
        Sets the number of principal components in use to be the fewest to explain at least a proportion
        C{frac} of the variance of the transformed original data.
        The number is set to a value between 1 and d inclusive - the number is set to 1 even
        if C{frac} == 0.  If you wish to use zero principal components, use L{setn} instead.
        
        @param frac: the proportion of variance to be explained.  Should be between 0 and 1 (inclusive).
        @type frac: float
        '''
        if frac > 1.0:
            frac = 1.0
        i = 0
        while (self.props[i]+self.tol) < frac:
            i+=1
        self.setn(i+1)

    def getprop(self):
        '''
        Get the proportion of variance explained by the current principal components
        
        @return: The proportion of variance in the transformed original data explained by the currently
        used principal components.  The value should be between 0 and 1 (inclusive). 
        @rtype: float
        '''
        return self.props[self.n-1]

    def transform(self, B, whiten=False):
        '''
        Transform C{B} onto the axes of the current principal components as was done with the original data.
        
        @param B: New data set (d x N2)
        @type B: 2-dimensional NumPy array
        @keyword whiten: True if the scaling transformation for whitening A should also be applied to the result.
        @type whiten: boolean  
        
        @return: The result after normalizing C{B} in the same way as the original data and projecting
        this normalized data onto the axes of the current principal components (n x N2).  If L{whiten}
        is C{True}, this result should further be transformed in the same way needed to whiten the
        original data, i.e. to ensure the transformed original data has an identity covariance matrix
        (when dividing by N, not N-1).
        @rtype: 2-dimensional NumPy array 
        '''
        B = np.asarray(np.copy(B),float)
        for i in range(B.shape[1]):
            B[:,i] = (B[:,i]-self.mean)/self.std
            
        proj = np.dot(self.U[:, 0:self.n].T,B)
        
        if(whiten):
            proj = np.dot( np.diag( 1.0/np.sqrt(self.D[0:self.n]) ) , proj )
        return proj

    def untransform(self, B, whiten=False):
        '''
        Transform C{B} from the axis system of the current principal components back to the original input
        space.  This method thus finds the original input mapping to C{B} in the current principal components
        space.  Both the rotation involved in the projection onto the current PCs, as well as any
        transformations for centring and scaling need to be reversed for this reconstruction. In addition, if
        C{B} has been obtained through whitening, the whitening step must also be undone.  (If the
        principal components were found from a covariance matrix instead of a data matrix,
        transformations to account for scaling and translation in the input space can not be made, however.) 
        
        @param B: New data set (d x N2)
        @type B: 2-dimensional NumPy array
        @keyword whiten: True if C{B} should be "dewhitened" as part of the "untransformation"
        @type whiten: boolean  
        
        @return: The result after finding the pre-image of C{B} in the original input space, under the
        mapping used in the C{transform} method (with the same value of C{whiten}).
        @rtype: 2-dimensional NumPy array 
        '''
        B = np.asarray(np.copy(B),float)
        d = B.shape[0]
        if(whiten):
            proj = np.dot( np.diag( np.sqrt(self.D[0:d]) ) , B )
                
        proj = np.dot(self.U[:, 0:d],B)
        
        for i in range(B.shape[1]):
            proj[:,i] = proj[:,i]*self.std+self.mean
        
        return proj

    def components(self):
        '''
        Return the current principal components in the original basis
        
        @return: The currently used principal components, one per column, in the original axis system. (d x n)
        @rtype: 2-dimensional NumPy array
        '''
        return np.copy(self.U[:, 0:self.n])

    def __init__(self, A, frac=1.0, center=True, scale=True, cov=False, tol=1e-8):
        '''
        Perform the initial principal components calculation, and store information for future queries.
        Note that your principal components should not include any axes corresponding to zero eigenvalues.
        
        @param A: Original data matrix (d x N)
        @type A: 2-dimensional NumPy array
        @keyword frac: The initial proportion of variance to be explained.  This is used to select the number of principal
                       components, using L{setfrac}.  Should be between 0 and 1 (inclusive).
        @type frac: float
        @keyword center: C{True} if C{A} should be centred to zero mean before processing
        @type center: boolean
        @keyword scale: C{True} if C{A} should be scaled to unit variance in each component before processing
        @type scale: boolean
        @keyword cov: C{True} if C{A} represents a covariance matrix.  This should typically be used with C{center=False},
                      C{scale=False}, and C{frac=1.0}.
        @type cov: boolean
        '''
        self.d = A.shape[0]
        self.N = A.shape[1]
        self.center = center
        self.scale = scale
        self.tol = tol
        A, self.mean, self.std = utils.centerscale(A,center,scale)
        self.mean = self.mean.ravel()
        self.std = self.std.ravel()
        
        if(cov == False):
            self.U, self.D = np.linalg.svd( A )[:2]
            self.D *= self.D
            self.D /= self.N
        else:
            self.D, self.U = np.linalg.eigh( A )
            idx = self.D.argsort()[::-1]  
            self.D = self.D[idx]
            self.U = self.U[:,idx]
        
        self.r = np.count_nonzero(self.D)
        
        self.props = np.zeros(self.r);
        self.props[0] = self.D[0]
        for i in range(1,self.r):
            self.props[i] = self.props[i-1] + self.D[i]
        self.props /= self.props[self.r-1]
        self.setfrac(frac)


class LDA:
    '''
    Class that performs multi-class LDA.

    Initial part of example from http://courses.ee.sun.ac.za/Pattern_Recognition_813/lectures/lecture01/node10.html :

    >>> import numpy.testing as npt
    >>> data = np.array([[1, 2, 1, 2, 1, 1, -2, 1],
    ... [0, 1, 0, 4, 2, -1, -2, -1],
    ... [2, 4, 2, 1, 2, 1, -2, 1]])
    >>> classes = ["Class 1", "Class 2", "Class 3"]
    >>> labels = np.array(["Class 1", "Class 1", "Class 1", "Class 2", "Class 2", "Class 3", "Class 3", "Class 3"])
    >>> result = LDA(data, classes, labels, center=False, scale=False)
    >>> transform1 = np.array([[-0.44791249, -0.21328877, -0.49135282], [-0.46021238, -1.27410287,  0.97259381], [-3.27842535,  2.603001  ,  1.85865785]])
    >>> npt.assert_almost_equal(result.pca1.transform(np.eye(3), whiten=True)*result.pca1.transform(np.eye(3), whiten=True), transform1*transform1, decimal=4)
    >>> transform2 = np.array([[-0.23151384, -0.4310839 ,  0.87210551], [ 0.48009477, -0.83031982, -0.28298058]])
    >>> npt.assert_almost_equal(result.pca2.transform(np.eye(3))*result.pca2.transform(np.eye(3)), transform2*transform2, decimal=4)
    >>> newdata = np.array([[1, 1, -1],
    ... [0, 4, -1],
    ... [1, 1, -1]])
    >>> transform3 = np.array([[ -1.24161353,  10.23325065,  -1.62710251], [ -0.47461019,   0.40105103,   0.25569488]])
    >>> npt.assert_almost_equal(result.transform(newdata)*result.transform(newdata), transform3*transform3, decimal=4)

    @ivar N: number of data points in A
    @type N: integer
    @ivar d: number of features of data points
    @type d: integer
    @ivar mean: mean of data points in A if centering enabled (d)
    @type mean: 1-dimensional NumPy array
    @ivar std: standard deviation of data points in A if scaling enabled (d)
    @type std: 1-dimensional NumPy array
    @ivar classmeans: class means, one per column (d x k)
    @type classmeans: 2-dimensional NumPy array
    @ivar aveclasscov: weighted average of the class-conditional covariance matrices (d x d)
    @type aveclasscov: 2-dimensional NumPy array
    @ivar sizes: number of points in the various classes (k)
    @type sizes: list of integers
    @ivar pca1: result of PCA on the aveclasscov
    @type pca1: dimred.PCA
    @ivar pca2: result of PCA on the whitened transformed classmeans onto the PCs from pca1.
            Specifically, if each observation in the original matrix A were replaced by its
            class mean to obtain a new matrix B, and B were then whitened using pca1's transform
            method to obtain C, then pca2 should be constructed from C's covariance matrix.  (It
            is not necessary to calculate B and C explicitly to do this) 
    @type pca2: dimred.PCA
    '''

    def transform(self, B):
        '''
        Apply the pre-processing and LDA transformation determined from the original data set to C{B}.

        @param B: Data to be transformed (d x N2)
        @type B: 2-dimensional NumPy array
        
        @return: Transformed data (p x N2, where p is the number of LDA axes)
        @rtype: 2-dimensional NumPy array
        '''
        B = np.asarray(np.copy(B),float)
        for i in range(B.shape[1]):
            B[:,i] -= self.mean
            B[:,i] /= self.std
        return np.dot(self.W.T,B)
    
    def __init__(self, A, classes, labels, center=True, scale=True):
        '''
        Perform multi-class linear discriminant analysis on C{A}.
        
        @param A: data (d x N)
        @type A: 2-dimensional NumPy array
        @param classes: class names (k)
        @type classes: list
        @param labels: class membership of each point (N)
        @type labels: list
        @keyword center: C{True} if C{A} should be centred to zero mean before processing
        @type center: boolean
        @keyword scale: C{True} if C{A} should be scaled to unit variance in each component before processing
        @type scale: boolean
        '''
        self.d = A.shape[0]
        self.N = A.shape[1]
        k = len(classes)
        
        self.center = center
        self.scale = scale
        A, self.mean, self.std = utils.centerscale(A,center,scale)
        self.mean = self.mean.ravel()
        self.std = self.std.ravel()
        
        mean = np.sum(A,axis=1)/self.N
        
        self.aveclasscov = np.zeros((self.d, self.d), dtype=np.float)
        self.classmeans = np.zeros((self.d, k), dtype=np.float)
        self.sizes = np.zeros((k,), dtype=np.float)
        sbCalc = np.zeros((self.d, k), dtype=np.float)
        for i in range(k):
            idx = np.where(labels==classes[i])[0]
            B = A[:,idx]
            n = idx.shape[0]
            p = n/self.N
            self.sizes[i] = n
            self.aveclasscov += np.cov(B,bias=1) * p
            self.classmeans[:,i] = np.sum(B,axis=1)/n
            sbCalc[:,i] = (self.classmeans[:,i] - mean) * np.sqrt(p)
        
        self.pca1 = PCA(self.aveclasscov, 1.0, False, False, True, tol=0.1e-5)
        self.pca2 = PCA(self.pca1.transform(sbCalc, True), 1.0, False, False, False )
        self.W = self.pca2.transform(self.pca1.transform(np.eye(self.d),True)).T

def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
