'''Module containing classification algorithms.

@since: 16 Jan 2012
@author: skroon
'''

import numpy as np

def optcriterion(points, classes, criteria, kwparams=None, _max=True):
    '''Classify C{points} on the basis of which of the C{criteria} yields an optimum.
    
    Classify the elements of C{points} into the classes specified by C{classes}. C{criteria} is
    a list of functions, for which any corresponding keyword arguments to be used may be specified in the
    list of dictionaries C{kwparams}. C{_max} indicates whether the maximum or minimum of the criterion is
    sought.  Each element of C{points} is a NumPy array.

    Example: our data comes from 3 classes, each generated from the same type of distribution, but each class
    has a different mean.  C{lik} is the likelihood function, and accepts the C{mean} as a keyword and a point
    at which the likelihood should be evaluated as a parameter.  Then, C{criteria}
    would be C{[lik]*3}, while C{criteria} would be a list of dictionaries,each containing a value associated with the
    key C{mean}.
    
    Of course, when we are considering from which distribution our data might come, the elements of the C{criteria}
    sequence may differ.
    
    >>> import numpy.testing as npt
    >>> def f(x, mean=0, var=0): return ((x-mean)**2)/var
    ...
    >>> npt.assert_equal(optcriterion([1, 2, 3], ["Class 1", "Class 2"], [f]*2, [{"mean":2., "var":6.},{"mean":3., "var":6.}], _max=False), ['Class 1', 'Class 1', 'Class 2'])
    >>> import density
    >>> d1 = density.Gaussian([1.1], [[1.]])
    >>> d2 = density.Gaussian([2.2], [[1.]])
    >>> npt.assert_equal(optcriterion([np.array([[1, 0.9]]), np.array([[1.8, 0.9]]), 
    ... np.array([[2]]), np.array([[2.3]]), 
    ... np.array([[3]])], ["Class 1", "Class 2"],
    ... [d1.negloglik, d2.negloglik], _max=False),
    ... ['Class 1', 'Class 1', 'Class 2', 'Class 2', 'Class 2'])
    >>> d1 = density.Gaussian([0.], [[1.]])
    >>> d2 = density.Gaussian([0.], [[2.]])
    >>> npt.assert_equal(optcriterion([np.array([[0.96, 0.9]]), np.array([[1.8]]),
    ... np.array([[2]]), np.array([[2.3]]), np.array([[3]])],
    ... ["Class 1", "Class 2"], [d1.negloglik, d2.negloglik], _max=False),
    ... ['Class 1', 'Class 2', 'Class 2', 'Class 2', 'Class 2'])

    @param points: List of n points to be classified, each element a d x N NumPy array
    @type points: list of 2-dimensional NumPy arrays
    @param classes: Class labels for the classes/models specified by the corresponding criteria (k)
    @type classes: list
    @param criteria: List of criteria functions among which an optimum is sought (k)
    @type criteria: list of functions
    @param kwparams: List of dictionaries containing the keyword arguments of the corresponding elements of C{criteria} (k)
    @type kwparams: list of dictionaries
    @keyword _max: Indicates if a class maximizing (C{_max = True}) or minimizing (C{_max = False}) the C{criteria} is sought. 
    @type _max: boolean
     
    @return: classification of each data point (n)
    @rtype: list
    '''
    clas = []
    for p in points:
        x = [(criteria[i](p,**(kwparams[i])) if kwparams else criteria[i](p)) for i in range(len(criteria))]
        idx = np.argmax(x) if _max else np.argmin(x)
        clas.append(classes[idx])
    return clas

def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()
