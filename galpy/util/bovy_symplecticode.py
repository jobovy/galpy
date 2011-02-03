#Symplectic ODE integrators
#Follows scipy.integrate.odeint inputs
import numpy as nu
from numpy import linalg
def leapfrog(func,yo,t,args=(),rtol=None,atol=None):
    """
    NAME:
       leapfrog
    PURPOSE:
       leapfrog integrate an ode
    INPUT:
       func - force function of (y,*args)
       yo - initial condition [q,p]
       t - set of times at which one wants the result
       rtol, atol
    OUTPUT:
       y : array, shape (len(y0), len(t))
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
    HISTORY:
       2011-02-02 - Written - Bovy (NYU)
    """
    if rtol is None:
        rtol= 1.49012e-8
    if atol is None:
        atol= 0.       
    #Initialize
    qo= yo[0:len(yo)/2]
    po= yo[len(yo)/2:len(yo)]
    out= nu.zeros((len(yo),len(t)))
    out[:,0]= yo
    #Estimate necessary step size
    dt= t[1]-t[0] #assumes that the steps are equally spaced
    init_dt= dt
    dt= _leapfrog_estimate_step(func,qo,po,dt,t[0],args,rtol,atol)
    ndt= int(init_dt/dt)
    print dt, ndt
    #Integrate
    to= t[0]
    for ii in range(1,len(t)):
        for jj in range(ndt): #loop over number of sub-intervals
            #This could be made faster by combining the drifts
            #drift
            q12= leapfrog_leapq(qo,po,dt/2.)
            #kick
            force= func(q12,*args,t=to+dt/2)
            po= leapfrog_leapp(po,dt,force)
            #drift
            qo= leapfrog_leapq(q12,po,dt/2.)
            #Get ready for next
            to+= dt
        out[0:len(yo)/2,ii]= qo
        out[len(yo)/2:len(yo),ii]= po
    return out.T

def leapfrog_leapq(q,p,dt):
    return q+dt*p

def leapfrog_leapp(p,dt,force):
    return p+dt*force

def _leapfrog_estimate_step(func,qo,po,dt,to,args,rtol,atol):
    qmax= nu.amax(nu.fabs(qo))+nu.zeros(len(qo))
    pmax= nu.amax(nu.fabs(po))+nu.zeros(len(po))
    scale= atol+rtol*nu.array([qmax,pmax]).flatten()
    print scale
    err= 2.
    dt*= 2.
    while err > 1.:
        #Do one leapfrog step with step dt and one with dt/2.
        #dt
        q12= leapfrog_leapq(qo,po,dt/2.)
        force= func(q12,*args,t=to+dt/2)
        p11= leapfrog_leapp(po,dt,force)
        q11= leapfrog_leapq(q12,p11,dt/2.)
        #dt/2.
        q12= leapfrog_leapq(qo,po,dt/4.)
        force= func(q12,*args,t=to+dt/4)
        ptmp= leapfrog_leapp(po,dt/2.,force)
        qtmp= leapfrog_leapq(q12,ptmp,dt/2.)#Take full step combining two half
        force= func(qtmp,*args,t=to+3.*dt/4)
        p12= leapfrog_leapp(ptmp,dt/2.,force)
        q12= leapfrog_leapq(qtmp,p12,dt/4.)
        #Norm
        delta= nu.array([nu.fabs(q11-q12),nu.fabs(p11-p12)]).flatten()
        err= nu.sqrt(nu.mean((delta/scale)**2.))
        print err
        dt/= 2.
    return dt

