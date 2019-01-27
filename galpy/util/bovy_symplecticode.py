#############################################################################
#Symplectic ODE integrators
#Follows scipy.integrate.odeint inputs as much as possible
#############################################################################
#############################################################################
#Copyright (c) 2011, Jo Bovy
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without 
#modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products 
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import numpy as nu
_MAX_DT_REDUCE= 10000.
def leapfrog(func,yo,t,args=(),rtol=1.49012e-12,atol=1.49012e-12):
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
    #Initialize
    qo= yo[0:len(yo)//2]
    po= yo[len(yo)//2:len(yo)]
    out= nu.zeros((len(t),len(yo)))
    out[0,:]= yo
    #Estimate necessary step size
    dt= t[1]-t[0] #assumes that the steps are equally spaced
    init_dt= dt
    dt= _leapfrog_estimate_step(func,qo,po,dt,t[0],args,rtol,atol)
    ndt= int(init_dt/dt)
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
        out[ii,0:len(yo)//2]= qo
        out[ii,len(yo)//2:len(yo)]= po
    return out

def leapfrog_leapq(q,p,dt):
    return q+dt*p

def leapfrog_leapp(p,dt,force):
    return p+dt*force

def _leapfrog_estimate_step(func,qo,po,dt,to,args,rtol,atol):
    init_dt= dt
    qmax= nu.amax(nu.fabs(qo))+nu.zeros(len(qo))
    pmax= nu.amax(nu.fabs(po))+nu.zeros(len(po))
    scale= atol+rtol*nu.array([qmax,pmax]).flatten()
    err= 2.
    dt*= 2.
    while err > 1. and init_dt/dt < _MAX_DT_REDUCE:
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
        dt/= 2.
    return dt

