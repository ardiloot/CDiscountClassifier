cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def RepeatAndLabel(np.ndarray[long long, ndim = 1] arr, np.ndarray[long long, ndim = 1] ns):
    """RepeatAndLabel(arr, ns)
    
    This is a helper function that repeats the every element of `arr` number of
    times specified by `ns`. Implemented in Cython for performance reasons.
    
    """
    cdef int newRows = ns.sum()
    
    res = np.zeros((newRows, 2), dtype = arr.dtype)
    
    cdef int i, j, n, count = 0
    for i in range(arr.shape[0]):
        n = ns[i]
        for j in range(n):
            res[count, 0] = arr[i]
            res[count, 1] = j
            count += 1

    return res
    