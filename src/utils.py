'''Module containing various utility functions

@since: 10 Jan 2012

@author: skroon
'''
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

def centerscale(A, center=True, scale=True, scalefactor=None):
    '''
    Centre and scale C{A} as requested, returning the centring
    and scaling vectors used.  Any component of C{A} with zero variance should
    not be scaled, and the corresponding entries of the returned scaling
    vector should be set to 1.

    Note that you usually would not want to use this function with C{center=False} and
    C{scale=True}.

    >>> import numpy.testing as npt
    >>> result = centerscale(np.array(range(10)).reshape(2,5), scale=False)
    >>> npt.assert_almost_equal(result[0], np.array([[-2., -1.,  0.,  1.,  2.], [-2., -1.,  0.,  1.,  2.]]), decimal=4)
    >>> npt.assert_almost_equal(result[1], np.array([[ 2.], [ 7.]]), decimal=4)
    >>> npt.assert_almost_equal(result[2], np.array([[ 1.], [ 1.]]), decimal=4)
    >>> result = centerscale(np.array(range(10)).reshape(2,5))
    >>> npt.assert_almost_equal(result[0], np.array([[-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356], [-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356]]), decimal=4)
    >>> npt.assert_almost_equal(result[1], np.array([[ 2.], [ 7.]]), decimal=4)
    >>> npt.assert_almost_equal(result[2], np.array([[ 1.41421356], [ 1.41421356]]), decimal=4)
    >>> result = centerscale(np.array(range(10)).reshape(2,5), scale=True, scalefactor=2)
    >>> npt.assert_almost_equal(result[0], np.array([[-1. , -0.5,  0. ,  0.5,  1. ], [-1. , -0.5,  0. ,  0.5,  1. ]]), decimal=4)
    >>> npt.assert_almost_equal(result[1], np.array([[ 2.], [ 7.]]), decimal=4)
    >>> npt.assert_almost_equal(result[2], np.array([[ 2.], [ 2.]]), decimal=4)

    @param A: The data to be adjusted (d x N)
    @type A: 2-dimensional NumPy array
    @keyword center: Indicates whether the data should be centred
    @type center: boolean
    @keyword scale: Indicates whether scaling should be applied.  If C{scalefactor} is not specified, and this is
                    C{True} the data should be scaled to
                    have unit standard deviation in each component after any centring.
    @type scale: boolean
    @keyword scalefactor: A factor by which all components should be scaled down, if provided.  This scaling is
                          not applied if C{scale} is C{False}.
    @type scalefactor: float

    @return:
     - The modified version of C{A} as a 2-dimensional NumPy array (d x N)
     - The translation applied to each component of the points in C{A}
       as a 1-dimensional NumPy array (i.e. the amounts subtracted from the components) (d)
     - The scaling factor applied to each component of the points in C{A}
       as a 1-dimensional NumPy array (i.e. the values the components were divided by) (d)
    @rtype: tuple
    
    '''
    A = np.asarray(np.copy(A),float)
    if(center == True):
        mean = np.mean(A,axis=1)
        for i in range(A.shape[1]):
            A[:,i] -= mean
        mean = np.reshape(mean, (A.shape[0],1))
    else:
        mean = np.zeros((A.shape[0],1),dtype=float)
            
    if(scale == True):
        if(scalefactor == None):
            scalefactor = np.std(A,axis=1)
            for i in range(A.shape[1]):
                A[:,i] = A[:,i] / scalefactor
            scalefactor = np.reshape(scalefactor, (A.shape[0],1))
        else:
            A /= scalefactor
    else:
        scalefactor = np.ones((A.shape[0],1),dtype=float)
        
    return A,mean,scalefactor

def confusion(a, b, print_=False):
    '''
    Generate and optionally print a confusion matrix.  For the printing, the column widths containing
    the numbers should all be equal, and should be wide enough to accommodate the widest class name  as
    well as the widest value in the matrix.

    >>> orig = ["Yellow", "Yellow", "Green", "Green", "Blue", "Yellow"]
    >>> pred = ["Yellow", "Green", "Green", "Blue", "Blue", "Yellow"]
    >>> result = confusion(orig, pred, print_=True)
              Blue  Green Yellow
       Blue      1      0      0
      Green      1      1      0
     Yellow      0      1      2
    >>> result
    {('Yellow', 'Green'): 1, ('Green', 'Blue'): 1, ('Green', 'Green'): 1, ('Blue', 'Blue'): 1, ('Yellow', 'Yellow'): 2}

    @param a: true labels
    @type a: list
    @param b: predicted labels
    @type b: list
    @keyword print_: C{True} if the confusion matrix should be printed
    @type print_: boolean 
    
    @return: the confusion matrix
    @rtype: dictionary
    '''
    classes = np.unique(a)
    n = len(classes)
    confMatrix = np.zeros((n,n),np.int)
    for i in range(len(a)):
        x = np.where(classes == a[i])
        y = np.where(classes == b[i])
        confMatrix[x,y] += 1
        
    if print_:
        maxl = 0
        for c in classes:
            if(len(str(c)) > maxl):
                maxl = len(str(c))
        longString = str(maxl+1)
        row_format =("{:>"+longString+"}")*(len(classes)+1)
        print row_format.format("", *classes)
        for cl, row in zip(classes, confMatrix):
            print row_format.format(cl, *row)
    re = {}
    for i in range(n):
        for ii in range(n):
            re[(classes[i],classes[ii])] = confMatrix[i,ii]
    return re

def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
