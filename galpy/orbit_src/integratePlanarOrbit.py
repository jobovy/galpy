def integratePlanarOrbit_leapfrog(pot,yo,t,rtol=None,atol=None):
    """
    NAME:
       integratePlanarOrbit_leapfrog
    PURPOSE:
       leapfrog integrate an ode for a planarOrbit
    INPUT:
       pot - Potential or list of such instances
       yo - initial condition [q,p]
       t - set of times at which one wants the result
       rtol, atol
    OUTPUT:
       y : array, shape (len(y0), len(t))
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
    HISTORY:
       2011-10-03 - Written - Bovy (NYU)
    """
    
