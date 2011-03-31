###############################################################################
#   bovy_quadpack: some variations on scipy's quadpack
###############################################################################
from scipy.integrate import quad
def _infunc(x,func,gfun,hfun,more_args,epsrel,epsabs):
    a = gfun(x)
    b = hfun(x)
    myargs = (x,) + more_args
    retval= quad(func,a,b,args=myargs,epsrel=epsrel,epsabs=epsabs)
    #print x, a, b, retval
    return retval[0]

def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
    return quad(_infunc,a,b,(func,gfun,hfun,args,epsrel,epsabs),
                epsabs=epsabs,epsrel=epsrel)
