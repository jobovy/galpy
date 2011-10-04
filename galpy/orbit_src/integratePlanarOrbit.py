import numpy
import ctypes
from galpy import potential
#Find and load the library
_lib = None
_libname = ctypes.util.find_library('galpy_integrate_c')
if _libname:
    _lib = ctypes.CDLL(_libname)
if _lib is None:
    import sys
for path in sys.path:
    try:
        _lib = ctypes.CDLL(os.path.join(path,'galpy_integrate_c.so'))
    except OSError:
        _lib = None
    else:
        break
if _lib is None:
    raise IOError('galpy integration module not found')

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
    #Figure out what's in pot
    if not isinstance(pot,list):
        pot= [pot]
    #Initialize everythin
    logp= chr(False)
    nlpargs= 0
    lpargs= numpy.zeros(1)
    for p in pot:
        if isinstance(p,potential.LogarithmicHaloPotential):
            logp= chr(True)
            nlpargs= 1
            lpargs= numpy.zeros(1)+p._q

    #Set up the C code
    ndarrayFlags= ('C_CONTIGUOUS','WRITEABLE')
    integrationFunc= _lib.integratePlanarOrbit
    integrationFunc.argtypes= [ctypes.c_int,
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_int,                             
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_char_p,
                               ctypes.c_int,
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags),
                               ctypes.c_double,
                               ctypes.c_double,
                               ndpointer(dtype=nu.float64,flags=ndarrayFlags)]

    #Array requirements, first store old order
    f_cont= [yo.flags['F_CONTIGUOUS'],
             t.flags['F_CONTIGUOUS'],
             lpargs.flags['F_CONTIGUOUS'],
             result.flags['F_CONTIGUOUS']]
    yo= nu.require(yo,dtype=nu.float64,requirements=['C','W'])
    t= nu.require(t,dtype=nu.float64,requirements=['C','W'])
    lpargs= nu.require(lpargs,dtype=nu.float64,requirements=['C','W'])
    result= nu.require(result,dtype=nu.float64,requirements=['C','W'])

    #Run the C code
    integrationFunc(yo,
                    ctypes.c_int(len(t)),
                    t,
                    ctypes.byref(ctypes.c_char(logp)),
                    ctypes.c_int(nlpargs),lpargs,
                    ctypes.c_double(rtol),ctypes.c_double(atol),
                    result)

    #Reset input arrays
    if f_cont[0]: yo= nu.asfortranarray(yo)
    if f_cont[1]: t= nu.asfortranarray(t)
    if f_cont[2]: lpargs= nu.asfortranarray(lpargs)
    if f_cont[3]: result= nu.asfortranarray(result)

                    
