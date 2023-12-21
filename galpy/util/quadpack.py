###############################################################################
#   galpy.util.quadpack: some variations on scipy's quadpack/quad
###############################################################################
import warnings

import numpy
from scipy.integrate import fixed_quad, quad
from scipy.integrate._quadrature import vectorize1


def _infunc(x, func, gfun, hfun, more_args, epsrel, epsabs):
    a = gfun(x)
    b = hfun(x)
    myargs = (x,) + more_args
    retval = quad(func, a, b, args=myargs, epsrel=epsrel, epsabs=epsabs)
    # print x, a, b, retval
    return retval[0]


def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
    return quad(
        _infunc,
        a,
        b,
        (func, gfun, hfun, args, epsrel, epsabs),
        epsabs=epsabs,
        epsrel=epsrel,
    )


# scipy adaptive Gaussian quadrature that was deprecated in 1.12.0 and associated
# functions; same BSD-3 license as galpy
class AccuracyWarning(Warning):
    pass


def vectorize1(func, args=(), vec_func=False):
    """Vectorize the call to a function.

    This is an internal utility function used by `romberg` and
    `quadrature` to create a vectorized version of a function.

    If `vec_func` is True, the function `func` is assumed to take vector
    arguments.

    Parameters
    ----------
    func : callable
        User defined function.
    args : tuple, optional
        Extra arguments for the function.
    vec_func : bool, optional
        True if the function func takes vector arguments.

    Returns
    -------
    vfunc : callable
        A function that will take a vector argument and return the
        result.

    """
    if vec_func:

        def vfunc(x):
            return func(x, *args)

    else:  # pragma: no cover
        # Don't cover because not used in galpy

        def vfunc(x):
            if numpy.isscalar(x):
                return func(x, *args)
            x = numpy.asarray(x)
            # call with first point to get output type
            y0 = func(x[0], *args)
            n = len(x)
            dtype = getattr(y0, "dtype", type(y0))
            output = numpy.empty((n,), dtype=dtype)
            output[0] = y0
            for i in range(1, n):
                output[i] = func(x[i], *args)
            return output

    return vfunc


def quadrature(
    func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=50, vec_func=True, miniter=1
):
    """
    Compute a definite integral using fixed-tolerance Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature
    with absolute tolerance `tol`.

    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function.
    tol, rtol : float, optional
        Iteration stops when error between last two iterates is less than
        `tol` OR the relative change is less than `rtol`.
    maxiter : int, optional
        Maximum order of Gaussian quadrature.
    vec_func : bool, optional
        True or False if func handles arrays as arguments (is
        a "vector" function). Default is True.
    miniter : int, optional
        Minimum order of Gaussian quadrature.

    Returns
    -------
    val : float
        Gaussian quadrature approximation (within tolerance) to integral.
    err : float
        Difference between last two estimates of the integral.
    """
    vfunc = vectorize1(func, args, vec_func=vec_func)
    val = numpy.inf
    err = numpy.inf
    maxiter = max(miniter + 1, maxiter)
    for n in range(miniter, maxiter + 1):
        newval = fixed_quad(vfunc, a, b, (), n)[0]
        err = abs(newval - val)
        val = newval

        if err < tol or err < rtol * abs(val):
            break
    else:
        warnings.warn(
            "maxiter (%d) exceeded. Latest difference = %e" % (maxiter, err),
            AccuracyWarning,
        )
    return val, err
