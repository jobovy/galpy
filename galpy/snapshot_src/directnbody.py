#Direct force summation N-body code
from __future__ import print_function
import numpy as nu
from numpy import linalg
import galpy.util.bovy_symplecticode as symplecticode
from galpy.potential.Potential import evaluateRforces, evaluatezforces,\
    evaluatephiforces
from galpy.potential.planarPotential import evaluateplanarRforces,\
    evaluateplanarphiforces
from galpy.potential.linearPotential import evaluatelinearForces
def direct_nbody(q,p,m,t,pot=None,softening_model='plummer',
                 softening_length=None,
                 atol=None,rtol=None):
    """
    NAME:
       direct_nbody
    PURPOSE:
       N-body code using direct summation for force evaluation
    INPUT:
       q - list of initial positions (numpy.ndarrays)
       p - list of initial momenta (numpy.ndarrays)
       m - list of masses
       t - times at which output is desired
       pot= external potential (galpy.potential or list of galpy.potentials)
       softening_model=  type of softening to use ('plummer')
       softening_length= (optional)
    OUTPUT:
       list of [q,p] at times t
    HISTORY:
       2011-02-03 - Written - Bovy (NYU)
    """
    #Set up everything
    if softening_model.lower() == 'plummer':
        softening= _plummer_soft
    out= []
    out.append([q,p])
    qo= q
    po= p
    #Determine appropriate stepsize
    dt= t[1]-t[0]
    ndt= 1
    to= t[0]
    #determine appropriate softening length if not given
    if softening_length is None:
        softening_length= 0.01
    #Run simulation
    for ii in range(1,len(t)):
        print(ii)
        for jj in range(ndt): #loop over number of sub-intervals
            (qo,po)= _direct_nbody_step(qo,po,m,to,dt,pot,
                                        softening,(softening_length,))
            #print(qo)
            to+= dt
        out.append([qo,po])
    #Return output
    return out

def _direct_nbody_step(q,p,m,t,dt,pot,softening,softening_args):
    """One N-body step: drift-kick-drift"""
    #drift
    q12= [symplecticode.leapfrog_leapq(q[ii],p[ii],dt/2.) \
              for ii in range(len(q))]
    #kick
    force= _direct_nbody_force(q12,m,t+dt/2.,pot,softening,softening_args)
    #print(force)
    p= [symplecticode.leapfrog_leapp(p[ii],dt,force[ii]) \
            for ii in range(len(p))]
    #drift
    q= [symplecticode.leapfrog_leapq(q12[ii],p[ii],dt/2.) \
            for ii in range(len(q12))]
    return (q,p)

def _direct_nbody_force(q,m,t,pot,softening,softening_args):
    """Calculate the force"""
    #First do the particles
    #Calculate all the distances
    nq= len(q)
    dim= len(q[0])
    dist_vec= nu.zeros((nq,nq,dim))
    dist= nu.zeros((nq,nq))
    for ii in range(nq):
        for jj in range(ii+1,nq):
            dist_vec[ii,jj,:]= q[jj]-q[ii]
            dist_vec[jj,ii,:]= -dist_vec[ii,jj,:]
            dist[ii,jj]= linalg.norm(dist_vec[ii,jj,:])
            dist[jj,ii]= dist[ii,jj]
    #Calculate all the forces
    force= []
    for ii in range(nq):
        thisforce= nu.zeros(dim)
        for jj in range(nq):
            if ii == jj: continue
            thisforce+= m[jj]*softening(dist[ii,jj],*softening_args)\
                /dist[ii,jj]*dist_vec[ii,jj,:]
        force.append(thisforce)
    #Then add the external force
    if pot is None: return force
    for ii in range(nq):
        force[ii]+= _external_force(q[ii],t,pot)
    return force

def _external_force(x,t,pot):
    dim= len(x)
    if dim == 3:
        #x is rectangular so calculate R and phi
        R= nu.sqrt(x[0]**2.+x[1]**2.)
        phi= nu.arccos(x[0]/R)
        sinphi= x[1]/R
        cosphi= x[0]/R
        if x[1] < 0.: phi= 2.*nu.pi-phi
        #calculate forces
        Rforce= evaluateRforces(R,x[2],pot,phi=phi,t=t)
        phiforce= evaluatephiforces(R,x[2],pot,phi=phi,t=t)
        return nu.array([cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce,
                     evaluatezforces(R,x[2],pot,phi=phi,t=t)])
    elif dim == 2:
        #x is rectangular so calculate R and phi
        R= nu.sqrt(x[0]**2.+x[1]**2.)
        phi= nu.arccos(x[0]/R)
        sinphi= x[1]/R
        cosphi= x[0]/R
        if x[1] < 0.: phi= 2.*nu.pi-phi
        #calculate forces
        Rforce= evaluateplanarRforces(R,pot,phi=phi,t=t)
        phiforce= evaluateplanarphiforces(R,pot,phi=phi,t=t)
        return nu.array([cosphi*Rforce-1./R*sinphi*phiforce,
                         sinphi*Rforce+1./R*cosphi*phiforce])
    elif dim == 1:
        return evaluatelinearForces(x,pot,t=t)

def _plummer_soft(d,eps):
    return d/(d**2.+eps**2.)**1.5

