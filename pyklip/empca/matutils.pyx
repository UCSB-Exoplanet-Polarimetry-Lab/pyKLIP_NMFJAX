
import cython
from cython.parallel import prange, parallel
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

def multmats(double [:, :] A, float [:, :] b, float [:, :] w, 
             int maxproc=4, int blocksize=1000):
    
    """
    Multiply the input matrices A, b, and w together, equivalent 
    to the following numpy code:
    
    all_A[i] = np.dot(A, (A*w[i]).T)
    all_b[i] = np.dot(A, w[i]*b[i])

    The resulting matrices all_A and all_b are then input to a
    SVD-based routine to find the least-squares solution to 
    all_A[i]*x = all_b[i].  The routine is fully parallelized 
    using openMP.

    Inputs:
    1. A       : 2D double-precision numpy.ndarray
    2. b       : 2D single-precision numpy.ndarray
    3. w       : 2D single-precision numpy.ndarray

    Optional input:
    1. maxproc   : integer, maximum number of threads.  Default 4.
    2. blocksize : integer, size of array to process at once.  This
                   is a small hack to maximize performance by 
                   increasing the cache hit rate.  Default 1000.

    Returns [all_A, all_b], double-precision ndarrays.

    Notes: b and w must match in shape.  For the expectation-
    maximization low-rank approximation, b is generally the data
    to be approximated and w the weights.
    
    """

    cdef int nvec, di, npts, i, j, k, ii
    cdef int i1, i2, n
    cdef double x

    assert b.shape[0] == w.shape[0] and b.shape[1] == w.shape[1]
    assert A.shape[1] == w.shape[1]

    nvec = A.shape[0]
    npts = A.shape[1]
    di = b.shape[0]

    ###############################################################
    # Don't initialize to zero: this will not be parallelized.
    # We'll assign values to all entries within the loop.
    ###############################################################

    all_A_np = np.empty((di, nvec, nvec))
    cdef double [:, :, :] all_A = all_A_np
    all_b_np = np.empty((di, nvec))
    cdef double [:, :] all_b = all_b_np
        
    ###############################################################
    # The arrays below break up the large arrays into smaller 
    # arrays.  This has several positive effects.  The first is 
    # that the smaller arrays give higher cache hit rates, while
    # the second is that the summations are done in smaller groups,
    # which lessens the impact of round-off error.  Finally, there
    # is no float-to-double cast in the multiplications that
    # follow; that is done earlier (and fewer times overall).
    ###############################################################

    if blocksize > npts:
        n = blocksize
    else:
        n = npts
    A_tmp_np = np.empty((nvec, n))
    cdef double [:, :] A_tmp = A_tmp_np
    w_tmp_np = np.empty((di, n))
    cdef double [:, :] w_tmp = w_tmp_np
    b_tmp_np = np.empty((di, n))
    cdef double [:, :] b_tmp = b_tmp_np

    if True: #npts > blocksize:        
        i1 = 0
        while True:
            i2 = i1 + blocksize
            if i2 > npts:
                i2 = npts
        
            with nogil, parallel(num_threads=maxproc):
                for i in prange(nvec, schedule='dynamic'):
                    for k in range(i2 - i1):
                        A_tmp[i, k] = A[i, i1 + k]

                for i in prange(di, schedule='dynamic'):

                    for k in range(i2 - i1):
                        w_tmp[i, k] = w[i, k + i1]
                        b_tmp[i, k] = b[i, k + i1]

                    for j in range(nvec):
                        for ii in range(nvec):
                            if i1 == 0:
                                all_A[i, j, ii] = 0
                            x = 0
                            for k in range(i2 - i1):
                                x = x + w_tmp[i, k]*A_tmp[j, k]*A_tmp[ii, k]
                            all_A[i, j, ii] = all_A[i, j, ii] + x
    
                        if i1 == 0:
                            all_b[i, j] = 0
                        x = 0
                        for k in range(i2 - i1):
                            x = x + A_tmp[j, k]*w_tmp[i, k]*b_tmp[i, k]
                        all_b[i, j] = all_b[i, j] + x
      
            i1 = i2
            if i2 == npts:
                break        

    ##################################################################
    # The code below is inefficient with large arrays because the
    # cache hit rate becomes low inside the parallel block.
    ##################################################################

    else:
        with nogil, parallel(num_threads=maxproc):
            for i in prange(di, schedule='dynamic'):
                for j in range(nvec):
                    for ii in range(nvec):
                        x = 0
                        for k in range(npts):
                            x = x + w[i, k]*A[j, k]*A[ii, k]
                        all_A[i, j, ii] = x
                    x = 0
                    for k in range(npts):
                        x = x + A[j, k]*w[i, k]*b[i, k]
                    all_b[i, j] = x

    return [all_A_np, all_b_np]


@cython.boundscheck(False)
@cython.wraparound(False)

def calc_chisq(float [:, :] im, double [:, :] b, float [:, :] w, 
             double [:, :] coef, int maxproc=4):
    
    """
    Calculate chi squared, equivalent to 
    np.sum((im[i] - np.sum(coef[i]*b.T, axis=1))**2*w[i])
    summed over i, where i runs over the number of images.

    Returns the total chi squared summed over all points and all
    images.  nvec is the number of basis images.

    Inputs:
    1. im    : nim x npts, single-precision numpy.ndarray.  Data 
               to be fit by the basis images.
    2. b     : nvec x npts, double precision numpy.ndarray.  The
               nvec basis images.
    3. w     : nim x npts, single-precision numpy.ndarray.  
               Weights (inverse variances) of the data.
    4. coef  : nvec x npts, double precision numpy.ndarray.  The
               coefficients of the basis image fits.

    Output: the total chi squared (double precision float).

    Notes: for the best performance, im, w, and coef should be in C
    order (last axis fast), while b should be in

    """

    cdef int nvec, npts, n_im, i, j, k
    cdef double x, y

    n_im = im.shape[0]
    npts = im.shape[1]
    nvec = b.shape[0]

    chisq_np = np.empty((n_im))
    cdef double [:] chisq = chisq_np

    with nogil, parallel(num_threads=maxproc):
        for i in prange(n_im, schedule='dynamic'):
            y = 0
            for j in range(npts):
                x = im[i, j]
                for k in range(nvec):
                    x = x - coef[i, k]*b[k, j]
                y = y + x*x*w[i, j]
            chisq[i] = y

    return np.sum(chisq)


@cython.boundscheck(False)
@cython.wraparound(False)

def dot(double [:, :] A, double [:, :] B, int maxproc=4):
    
    """
    Compute and return the simple dot product of the input matrices
    A and B, performing operations in parallel.  Return the product
    A*B.  If A is n x m and B is m x l, the output will be n x l.

    The input matrices should be double precision, the output will
    also be double precision.
    """

    cdef int i, j, k, n1, n2, n3
    cdef double x

    n1 = A.shape[0]
    n2 = A.shape[1]
    n3 = B.shape[1]
    assert A.shape[1] == B.shape[0]

    result_np = np.empty((n1, n3))
    cdef double [:, :] result = result_np

    with nogil, parallel(num_threads=maxproc):
        for i in prange(n1, schedule='dynamic'):
            for j in range(n3):
                x = 0
                for k in range(n2):
                    x = x + A[i, k]*B[k, j]
                result[i, j] = x

    return result_np


@cython.wraparound(False)
@cython.boundscheck(False)

def lstsq(double [:, :, :] A, double [:, :] b, int maxproc=4):

    """

    """

    cdef int flag, its, jj, j, ii, i, l, k, nm, n, m, inc, di
    cdef double c, f, h, s, x, y, z, tmp, tmp1, tmp2, sw, eps, tsh
    cdef double anorm, g, scale

    cdef extern from "math.h" nogil:
        double sqrt(double _x)
        double fabs(double _x)

    m = A.shape[1]
    n = A.shape[2]
    inc = 1
    eps = 2.3e-16
    di = A.shape[0]

    ###############################################################
    # None of these arrays will be visible outside of this routine.
    # They are used to construct the SVD.
    ###############################################################

    tmparr_np = np.empty((di, n))
    cdef double [:, :] tmparr = tmparr_np
    su_np = np.empty((di, m))
    cdef double [:, :] su = su_np
    sv_np = np.empty((di, n))
    cdef double [:, :] sv = sv_np
    w_np = np.empty((di, n))
    cdef double [:, :] w = w_np
    rv1_np = np.empty((di, n))
    cdef double [:, :] rv1 = rv1_np

    v_np = np.empty((di, n, n))
    cdef double [:, :, :] v = v_np
    
    ###############################################################
    # coef is the array that will hold the answer.
    # It will be returned by the function.
    ###############################################################

    coef_np = np.zeros((di, n))
    cdef double [:, :] coef = coef_np

    ###############################################################
    # The code below is largely copied from Numerical Recipes and from
    # http://www.public.iastate.edu/~dicook/JSS/paper/code/svd.c
    ###############################################################

    with nogil, parallel(num_threads=maxproc):

        for ii in prange(di, schedule='dynamic'):
            scale = 0.
            g = 0.
            anorm = 0.
            for i in range(n):
                l = i + 1
                rv1[ii, i] = scale*g
                g = 0.
                s = 0.
                scale = 0.
                if i < m:
                    for k in range(i, m):
                        scale = scale + fabs(A[ii, k, i])
                    if scale != 0:
                        for k in range(i, m):
                            A[ii, k, i] = A[ii, k, i]/scale
                            s = s + A[ii, k, i]*A[ii, k, i]
                        f = A[ii, i, i]
                        g = -1*sqrt(s)
                        if f < 0:
                            g = -1*g
                        h = f*g - s
                        A[ii, i, i] = f - g
                        if i != n - 1:
                            for j in range(l, n):
                                s = 0
                                for k in range(i, m):
                                    s = s + A[ii, k, i]*A[ii, k, j]
                                f = s/h
                                for k in range(i, m):
                                    A[ii, k, j] = A[ii, k, j] + f*A[ii, k, i]
                        for k in range(i, m):
                            A[ii, k, i] = A[ii, k, i]*scale

                w[ii, i] = scale*g
                g = 0.
                s = 0.
                scale = 0.
                if i < m and i != n - 1:
                    for k in range(l, n):
                        scale = scale + fabs(A[ii, i, k])
                    if scale != 0:
                        for k in range(l, n):
                            A[ii, i, k] = A[ii, i, k]/scale
                            s = s + A[ii, i, k]*A[ii, i, k]
                        f = A[ii, i, l]
                        g = -1*sqrt(s)
                        if f < 0:
                            g = -1*g
                        h = f*g - s
                        A[ii, i, l] = f - g
                        for k in range(l, n):
                            rv1[ii, k] = A[ii, i, k]/h
                        if i != m - 1:
                            for j in range(l, m):
                                s = 0
                                for k in range(l, n):
                                    s = s + A[ii, j, k]*A[ii, i, k]
                                for k in range(l, n):
                                    A[ii, j, k] = A[ii, j, k] + s*rv1[ii, k]
    
                        for k in range(l, n):
                            A[ii, i, k] = A[ii, i, k]*scale
    
                if fabs(w[ii, i]) + fabs(rv1[ii, i]) > anorm:
                    anorm = fabs(w[ii, i]) + fabs(rv1[ii, i])
            
            for i in range(n - 1, -1, -1):
                if i < n - 1:
                    if g != 0:
                        for j in range(l, n):
                            v[ii, j, i] = A[ii, i, j]/A[ii, i, l]/g
                        for j in range(l, n):
                            s = 0
                            for k in range(l, n):
                                s = s + A[ii, i, k]*v[ii, k, j]
                            for k in range(l, n):
                                v[ii, k, j] = v[ii, k, j] + s*v[ii, k, i]
                    for j in range(l, n):
                        v[ii, i, j] = 0.
                        v[ii, j, i] = 0.
                v[ii, i, i] = 1.
                g = rv1[ii, i]
                l = i
    
            for i in range(n - 1, -1, -1):
                l = i + 1
                g = w[ii, i]
                if i < n - 1:
                    for j in range(l, n):
                        A[ii, i, j] = 0.
                if g != 0:
                    g = 1./g
                    if i != n - 1:
                        for j in range(l, n):
                            s = 0
                            for k in range(l, m):
                                s = s + A[ii, k, i]*A[ii, k, j]
                            f = (s/A[ii, i, i])*g
                            for k in range(i, m):
                                A[ii, k, j] = A[ii, k, j] + f*A[ii, k, i]
                    for j in range(i, m):
                        A[ii, j, i] = A[ii, j, i]*g
                else:
                    for j in range(i, m):
                        A[ii, j, i] = 0.
                A[ii, i, i] = A[ii, i, i] + 1.
    
            for k in range(n - 1, -1, -1):
                for its in range(30):
                    flag = 1
                    for l in range(k, -1, -1):
                        nm = l - 1
                        if fabs(rv1[ii, l]) + anorm == anorm:
                            flag = 0
                            break
                        if fabs(w[ii, nm]) + anorm == anorm:
                            break
                    if flag != 0:
                        c = 0.
                        s = 1.
                        for i in range(l, k + 1):
                            f = s*rv1[ii, i]
                            if fabs(f) + anorm != anorm:
                                g = fabs(w[ii, i])
                                h = sqrt(f*f + g*g)
                                w[ii, i] = h
                                h = 1./h
                                c = g*h
                                s = -1.*f*h
                                for j in range(m):
                                    y = A[ii, j, nm]
                                    z = A[ii, j, i]
                                    A[ii, j, nm] = y*c + z*s
                                    A[ii, j, i] = z*c - y*s
                    z = w[ii, k]
                    if l == k:
                        if z < 0.:
                            w[ii, k] = -1.*z
                            for j in range(n):
                                v[ii, j, k] = -1.*v[ii, j, k]
                        break
                    #if its >= 30:
                    # no convergence
                    
                    x = w[ii, l]
                    nm = k - 1
                    y = w[ii, nm]
                    g = rv1[ii, nm]
                    h = rv1[ii, k]
                    f = ((y - z)*(y + z) + (g - h)*(g + h))/(2.*h*y)
    
                    g = sqrt(1. + f*f)
                    tmp = g
                    if f < 0:
                        tmp = -1*tmp
                    
                    f = ((x - z)*(x + z) + h*((y/(f + tmp)) - h))/x
                    
                    c = 1.
                    s = 1.
                    for j in range(l, nm + 1):
                        i = j + 1
                        g = rv1[ii, i]
                        y = w[ii, i]
                        h = s*g
                        g = c*g
                                               
                        z = sqrt(f*f + h*h)

                        rv1[ii, j] = z
                        c = f/z
                        s = h/z
                        f = x*c + g*s
                        g = g*c - x*s
                        h = y*s
                        y = y*c
                        for jj in range(n):
                            x = v[ii, jj, j]
                            z = v[ii, jj, i]
                            v[ii, jj, j] = x*c + z*s
                            v[ii, jj, i] = z*c - x*s
                        
                        z = sqrt(f*f + h*h)

                        w[ii, j] = z
                        if z != 0:
                            z = 1./z
                            c = f*z
                            s = h*z
                        f = c*g + s*y
                        x = c*y - s*g
                        for jj in range(m):
                            y = A[ii, jj, j]
                            z = A[ii, jj, i]
                            A[ii, jj, j] = y*c + z*s
                            A[ii, jj, i] = z*c - y*s
    
                    rv1[ii, l] = 0.
                    rv1[ii, k] = f
                    w[ii, k] = x
    
            inc = 1
            while True:
                inc = inc*3 + 1
                if inc > n:
                    break
            while True:
                inc = inc/3
                for i in range(inc, n):
                    sw = w[ii, i]
                    for k in range(m):
                        su[ii, k] = A[ii, k, i]
                    for k in range(n):
                        sv[ii, k] = v[ii, k, i]
                    j = i
                    while w[ii, j - inc] < sw:
                        w[ii, j] = w[ii, j - inc]
                        for k in range(m):
                            A[ii, k, j] = A[ii, k, j - inc] 
                        for k in range(n):
                            v[ii, k, j] = v[ii, k, j - inc]
                        j = j - inc
                        if j < inc:
                            break
                    w[ii, j] = sw
                    for k in range(m):
                        A[ii, k, j] = su[ii, k]
                    for k in range(n):
                        v[ii, k, j] = sv[ii, k]
                if inc <= 1:
                    break
            for k in range(n):
                jj = 0
                for i in range(m):
                    if A[ii, i, k] < 0:
                        jj = jj + 1
                for j in range(n):
                    if v[ii, j, k] < 0:
                        jj = jj + 1
                if jj > (m + n)/2:
                    for i in range(m):
                        A[ii, i, k] = -1.*A[ii, i, k]
                    for j in range(n):
                        v[ii, j, k] = -1.*v[ii, j, k]


            #eps = 2.3e-16
            tsh = 0.5*sqrt(m + n + 1.)*w[ii, 0]*eps
    
            for j in range(n):
                s = 0.
                if w[ii, j] > tsh:
                    for i in range(m):
                        s = s + A[ii, i, j]*b[ii, i]
                    s = s/w[ii, j]
                tmparr[ii, j] = s*1.
            for j in range(n):
                s = 0.
                for jj in range(n):
                    s = s + v[ii, j, jj]*tmparr[ii, jj]
                coef[ii, j] = s*1.
    
    return coef_np
    
    
