#Symplectic ODE integrators
#Follows scipy.integrate.odeint inputs
import numpy as nu
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
    if rtol is None and atol is None:
        rtol= 1.49012e-8
        atol= 0.
    dt= t[1]-t[0] #FOR NOW BOVY
    ndt= 1
    #Initialize
    qo= yo[0:len(yo)/2]
    po= yo[len(yo)/2:len(yo)]
    out= nu.zeros((len(yo),len(t)))
    out[:,0]= yo
    #Integrate
    to= t[0]
    for ii in range(1,len(t)):
        for jj in range(ndt): #loop over number of sub-intervals
            #This could be made faster by combining the drifts
            #drift
            q12= leapfrog_leapq(qo,po,dt/2.)
            #kick
            force= func(q12,to+dt/2.,*args)
            p1= leapfrog_leapp(po,dt,force)
            #drift
            q1= leapfrog_leapq(qo,p1,dt/2.)
            #Get ready for next
            qo= q1
            po= p1
            to+= dt
        out[0:len(yo)/2,ii]= qo
        out[len(yo)/2:len(yo),ii]= po
    return out

def leapfrog_leapq(q,p,dt):
    return q+dt*p

def leapfrog_leapp(p,dt,force):
    return p+dt*force
