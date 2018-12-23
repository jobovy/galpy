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
       leapfrog integrate in rectangular coordinates
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
       2018-12-22 - Re-implemented using general leapfrog integrator
    """
    dim= len(yo)//2
    pdim= len(yo)
    def drift(x):
        out= nu.zeros(pdim)
        out[:dim]= x[dim:]
        return out
    def kick(x,*args,**kwargs):
        out= nu.zeros(pdim)
        out[dim:]= func(x[:dim],*args,**kwargs)
        return out
    qmax= nu.amax(nu.fabs(yo[:dim]))+nu.zeros(dim)
    pmax= nu.amax(nu.fabs(yo[dim:]))+nu.zeros(dim)
    scaling= nu.array([qmax,pmax]).flatten()
    return leapfrog_general(drift,kick,yo,t,args=args,
                            rtol=rtol,atol=atol,
                            scaling=scaling,metric=lambda x,y: nu.fabs(x-y))

def leapfrog_general(drift,kick,yo,t,args=(),rtol=1.49012e-12,atol=1.49012e-12,
                     scaling=None,metric=None,construct_Lz=False):
    """
    NAME:
       leapfrog_general
    PURPOSE:
       leapfrog integrate a Hamiltonian ODE in arbitrary coordinates (primarily used in galpy to integrate in cylindrical coordinates in 2,3,4,5,6 phase-space dimensions)
    INPUT:
       drift - Delta y in drift step, function of y
       kick -  Delta y in kick step, function of (y,*args,t=)
       yo - initial condition [R,vR,vT,z,vz,phi] or galpy subsets
       t - set of times at which one wants the result
       rtol, atol- relative and absolute tolerance parameters
       scaling- scaling to use in relative/absolute tolerance combination: scale= atol+rtol*scaling
       construct_Lz= (False) if True, input is in cylindrical galpy coordinates [R,vR,vT...] and we want to transform vT --> Lz
    OUTPUT:
       y : array, shape (len(y0), len(t))
       Array containing the value of y for each desired time in t, \
       with the initial value y0 in the first row.
    HISTORY:
       2018-12-21 - Written - Bovy (UofT)
    """
    #Initialize
    y= nu.copy(yo)
    if construct_Lz:
        # vT --> Lz
        y[2]*= y[0]
    out= nu.zeros((len(t),len(yo)))
    out[0,:]= y
    #Estimate necessary step size
    dt= t[1]-t[0] #assumes that the steps are equally spaced
    init_dt= dt
    dt= _leapfrog_cyl_estimate_step(drift,kick,y,dt,t[0],args,rtol,atol,
                                    scaling,metric)
    dt2= dt/2.
    ndt= int(init_dt/dt)
    #Integrate
    to= t[0]
    for ii in range(1,len(t)):
        #drift half
        y12= y+dt2*drift(y)
        for jj in range(ndt): #loop over number of sub-intervals
            #kick
            y12+= dt*kick(y12,*args,t=to+dt/2)
            #drift full
            y12+= dt*drift(y12)
            #Get ready for next
            to+= dt
        #drift half back to correct overshoot
        y= y12-dt2*drift(y12)
        out[ii,:]= y
    if construct_Lz:
        # Lz --> vT
        out[:,2]/= out[:,0]
    return out

def _leapfrog_cyl_estimate_step(drift,kick,yo,dt,to,args,rtol,atol,
                                scaling,metric):
    init_dt= dt
    scale= atol+rtol*scaling
    err= 2.
    dt*= 2.
    while err > 1. and init_dt/dt < _MAX_DT_REDUCE:
        #Do one leapfrog step with step dt and one with dt/2.
        #dt
        y12= yo+dt/2.*drift(yo)
        y12+= dt*kick(y12,*args,t=to+dt/2)
        y11= y12+dt/2.*drift(y12)
        #dt/2.
        y12= yo+dt/4.*drift(yo)
        y12+= dt/2.*kick(y12,*args,t=to+dt/4.)
        y12+= dt/2.*drift(y12)#Take full step combining two half
        y12+= dt/2.*kick(y12,*args,t=to+3.*dt/4.)
        y12+= dt/4.*drift(y12)
        #Norm
        delta= metric(y11,y12)
        err= nu.sqrt(nu.mean((delta/scale)**2.))
        dt/= 2.
    return dt

