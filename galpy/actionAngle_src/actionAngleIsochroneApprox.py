###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochroneApprox
#
#             Calculate actions-angle coordinates for any potential by using 
#             an isochrone potential as an approximate potential and using 
#             a Fox & Binney (2013?) + torus machinery-like algorithm 
#             (angle-fit)
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import copy
import math
import numpy as nu
import numpy.linalg as linalg
from scipy import optimize
from galpy.potential import dvcircdR, vcirc
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleIsochrone
from actionAngle import actionAngle
from galpy.potential import IsochronePotential
class actionAngleIsochroneApprox():
    """Action-angle formalism using an isochrone potential as an approximate potential and using a Fox & Binney (2013?) like algorithm to calculate the actions using orbit integrations and a torus-machinery-like angle-fit to get the angles and frequencies"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleIsochroneApprox object
        INPUT:
           Either:
              b= scale parameter of the isochrone parameter
              ip= instance of a IsochronePotential
              aAI= instance of an actionAngleIsochrone
           pot= potential to calculate action-angle variables for
           tintJ= (default: 100) time to integrate orbits for to estimate 
                  actions
           tintA= (default: 20) time to integrate orbits for to estimate angles
           ntintJ= (default: 10000) number of time-integration points
                  actions
           ntintA= (default: 100) number of time-integration points
           integrate_method= (default: 'dopr54_c') integration method to use
        OUTPUT:
        HISTORY:
           2013-09-10 - Written - Bovy (IAS)
        """
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleStaeckel")
        self._pot= kwargs['pot']
        if not kwargs.has_key('b') and not kwargs.has_key('ip') \
                and not kwargs.has_key('aAI'):
            raise IOError("Must specify b=, ip=, or aAI= for actionAngleIsochroneApprox")
        if kwargs.has_key('aAI'):
            if not isinstance(kwargs['aAI'],actionAngleIsochrone):
                raise IOError("'Provided aAI= does not appear to be an instance of an actionAngleIsochrone")
            self._aAI= kwargs['aAI']
        elif kwargs.has_key('ip'):
            ip= kwargs['ip']
            if not isinstance(ip,IsochronePotential):
                raise IOError("'Provided ip= does not appear to be an instance of an IsochronePotential")
            self._aAI= actionAngleIsochrone(ip=ip)
        else:
            self._aAI= actionAngleIsochrone(ip=IsochronePotential(b=kwargs['b'],
                                                                  normalize=1.))
        if kwargs.has_key('tintJ'):
            self._tintJ= kwargs['tintJ']
        else:
            self._tintJ= 100.
        if kwargs.has_key('ntintJ'):
            self._ntintJ= kwargs['ntintJ']
        else:
            self._ntintJ= 10000
        self._tsJ= nu.linspace(0.,self._tintJ,self._ntintJ)
        if kwargs.has_key('tintA'):
            self._tintA= kwargs['tintA']
        else:
            self._tintA= 20.
        if kwargs.has_key('ntintA'):
            self._ntintA= kwargs['ntintA']
        else:
            self._ntintA= 100
        self._tsA= nu.linspace(0.,self._tintA,self._ntintA)
        if kwargs.has_key('integrate_method'):
            self._integrate_method= kwargs['integrate_method']
        else:
            self._integrate_method= 'dopr54_c'
        self._c= False
        ext_loaded= False
        if ext_loaded and ((kwargs.has_key('c') and kwargs['c'])
                           or not kwargs.has_key('c')):
            self._c= True
        else:
            self._c= False
        return None
    
    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz:
                 1) floats: phase-space value for single object
                 2) numpy.ndarray: [N] phase-space values for N objects 
                 3) numpy.ndarray: [N,M] phase-space values for N objects at M
                    times
              b) Orbit instance or list thereof; can be integrated already
           nonaxi= set to True to also calculate Lz using the isochrone 
                   approximation for non-axisymmetric potentials
           cumul= if True, return the cumulative average actions (to look 
                  at convergence)
        OUTPUT:
           (jr,lz,jz)
        HISTORY:
           2013-09-10 - Written - Bovy (IAS)
        """
        R,vR,vT,z,vz,phi= self._parse_args(*args)
        if self._c:
            pass
        else:
            #Use self._aAI to calculate the actions and angles in the isochrone potential
            acfs= self._aAI.actionsFreqsAngles(R.flatten(),
                                               vR.flatten(),
                                               vT.flatten(),
                                               z.flatten(),
                                               vz.flatten(),
                                               phi.flatten())
            jrI= nu.reshape(acfs[0],R.shape)[:,:-1]
            jzI= nu.reshape(acfs[2],R.shape)[:,:-1]
            anglerI= nu.reshape(acfs[6],R.shape)
            anglezI= nu.reshape(acfs[8],R.shape)
            danglerI= ((nu.roll(anglerI,-1)-anglerI) % (2.*nu.pi))[:,:-1]
            danglezI= ((nu.roll(anglezI,-1)-anglezI) % (2.*nu.pi))[:,:-1]
            if kwargs.has_key('cumul') and kwargs['cumul']:
                sumFunc= nu.cumsum
            else:
                sumFunc= nu.sum
            jr= sumFunc(jrI*danglerI,axis=1)/sumFunc(danglerI,axis=1)
            jz= sumFunc(jzI*danglezI,axis=1)/sumFunc(danglezI,axis=1)
            if kwargs.has_key('nonaxi') and kwargs['nonaxi']:
                lzI= nu.reshape(acfs[1],R.shape)[:,:-1]
                anglephiI= nu.reshape(acfs[7],R.shape)
                danglephiI= ((nu.roll(anglephiI,-1)-anglephiI) % (2.*nu.pi))[:,:-1]
                lz= sumFunc(lzI*danglephiI,axis=1)/sumFunc(danglephiI,axis=1)
            else:
                lz= R[:,0]*vT[:,0]
            return (jr,lz,jz)

    def actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           actionsFreqs
        PURPOSE:
           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz:
                 1) floats: phase-space value for single object
                 2) numpy.ndarray: [N] phase-space values for N objects 
                 3) numpy.ndarray: [N,M] phase-space values for N objects at M
                    times
              b) Orbit instance or list thereof; can be integrated already
           nonaxi= set to True to also calculate Lz using the isochrone 
                   approximation for non-axisymmetric potentials
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        HISTORY:
           2013-09-10 - Written - Bovy (IAS)
        """
        acfs= self.actionsFreqsAngles(*args,**kwargs)
        return (acfs[0],acfs[1],acfs[2],acfs[3],acfs[4],acfs[5])

    def actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           actionsFreqsAngles
        PURPOSE:
           evaluate the actions, frequencies, and angles 
           (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        INPUT:
           Either:
              a) R,vR,vT,z,vz:
                 1) floats: phase-space value for single object
                 2) numpy.ndarray: [N] phase-space values for N objects 
                 3) numpy.ndarray: [N,M] phase-space values for N objects at M
                    times
              b) Orbit instance or list thereof; can be integrated already
           maxn= (default: 2) Use a grid in vec(n) up to this n (zero-based)
           nonaxi= set to True to also calculate Lz using the isochrone 
                   approximation for non-axisymmetric potentials
           ts= if set, the phase-space points correspond to these times (IF NOT SET, WE ASSUME THAT ts IS THAT THAT WAS SETUP WHEN SETTING UP THE OBJECT)
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        HISTORY:
           2013-09-10 - Written - Bovy (IAS)
        """
        R,vR,vT,z,vz,phi= self._parse_args(*args)
        if kwargs.has_key('ts'):
            ts= kwargs['ts']
        else:
            ts= self._tsJ
        if kwargs.has_key('maxn'):
            maxn= kwargs['maxn']
        else:
            maxn= 2
        if self._c:
            pass
        else:
            #Use self._aAI to calculate the actions and angles in the isochrone potential
            acfs= self._aAI.actionsFreqsAngles(R.flatten(),
                                               vR.flatten(),
                                               vT.flatten(),
                                               z.flatten(),
                                               vz.flatten(),
                                               phi.flatten())
            jrI= nu.reshape(acfs[0],R.shape)[:,:-1]
            jzI= nu.reshape(acfs[2],R.shape)[:,:-1]
            anglerI= nu.reshape(acfs[6],R.shape)
            anglezI= nu.reshape(acfs[8],R.shape)
            danglerI= ((nu.roll(anglerI,-1)-anglerI) % (2.*nu.pi))[:,:-1]
            danglezI= ((nu.roll(anglezI,-1)-anglezI) % (2.*nu.pi))[:,:-1]
            if kwargs.has_key('cumul') and kwargs['cumul']:
                sumFunc= nu.cumsum
            else:
                sumFunc= nu.sum
            jr= sumFunc(jrI*danglerI,axis=1)/sumFunc(danglerI,axis=1)
            jz= sumFunc(jzI*danglezI,axis=1)/sumFunc(danglezI,axis=1)
            if kwargs.has_key('nonaxi') and kwargs['nonaxi']:
                lzI= nu.reshape(acfs[1],R.shape)[:,:-1]
                anglephiI= nu.reshape(acfs[7],R.shape)
                danglephiI= ((nu.roll(anglephiI,-1)-anglephiI) % (2.*nu.pi))[:,:-1]
                lz= sumFunc(lzI*danglephiI,axis=1)/sumFunc(danglephiI,axis=1)
            else:
                lz= R[:,0]*vT[:,0]
            #Now do an 'angle-fit'
            angleRT= dePeriod(acfs[6])
            anglephiT= dePeriod(acfs[7])
            angleZT= dePeriod(acfs[8])
            #Write the angle-fit as Y=AX, build A and Y
            nt= len(angleRT)
            nn= maxn*(2*maxn-1)-maxn #remove 0,0,0
            A= nu.zeros((nt,2+nn))
            A[:,0]= 1.
            A[:,1]= ts
            #sorting the phi and Z grids this way makes it easy to exclude the origin
            phig= list(nu.arange(-maxn+1,maxn,1))
            phig.sort(key = lambda x: abs(x))
            phig= nu.array(phig,dtype='int')
            grid= nu.meshgrid(nu.arange(maxn),
                              phig,
                              indexing='ij')
            gridR= grid[0].flatten()[1:] #remove 0,0,0
            gridZ= grid[1].flatten()[1:]
            mask = nu.ones(len(gridR), dtype=bool)
            mask[:2*maxn-3:2]= False
            gridR= gridR[mask]
            gridZ= gridZ[mask]
            tangleR= nu.tile(angleRT,(nn,1)).T
            tgridR= nu.tile(gridR,(nt,1))
            tangleZ= nu.tile(angleZT,(nn,1)).T
            tgridZ= nu.tile(gridZ,(nt,1))
            sinnR= nu.sin(tgridR*tangleR+tgridZ*tangleZ)
            A[:,2:]= sinnR
            #Matrix magic
            atainv= linalg.inv(nu.dot(A.T,A))
            angleR= nu.sum(atainv[0,:]*nu.dot(A.T,angleRT))
            OmegaR= nu.sum(atainv[1,:]*nu.dot(A.T,angleRT))
            anglephi= nu.sum(atainv[0,:]*nu.dot(A.T,anglephiT))
            Omegaphi= nu.sum(atainv[1,:]*nu.dot(A.T,anglephiT))
            angleZ= nu.sum(atainv[0,:]*nu.dot(A.T,angleZT))
            OmegaZ= nu.sum(atainv[1,:]*nu.dot(A.T,angleZT))
            return (jr,lz,jz,OmegaR,Omegaphi,OmegaZ,angleR,anglephi,angleZ)


    def _parse_args(self,*args):
        """Helper function to parse the arguments to the __call__ and actionsFreqsAngles functions"""
        RasOrbit= False
        if len(args) == 5:
            raise IOError("Must specify phi for actionAngleIsochroneApprox")
        if len(args) == 6:
            R,vR,vT, z, vz, phi= args
            if isinstance(R,float):
                o= Orbit([R,vR,vT,z,vz,phi])
                o.integrate(self._tsJ,pot=self._pot,method=self._integrate_method)
                this_orbit= o.getOrbit()
                R= nu.reshape(this_orbit[:,0],(1,self._ntintJ))
                vR= nu.reshape(this_orbit[:,1],(1,self._ntintJ))
                vT= nu.reshape(this_orbit[:,2],(1,self._ntintJ))
                z= nu.reshape(this_orbit[:,3],(1,self._ntintJ))
                vz= nu.reshape(this_orbit[:,4],(1,self._ntintJ))           
                phi= nu.reshape(this_orbit[:,5],(1,self._ntintJ))           
            if len(R.shape) == 1: #not integrated yet
                os= [Orbit([R[ii],vR[ii],vT[ii],z[ii],vz[ii],phi[ii]]) for ii in range(R.shape[0])]
                RasOrbit= True
        if isinstance(args[0],Orbit) \
                or (isinstance(args[0],list) and isinstance(args[0][0],Orbit)) \
                or RasOrbit:
            if RasOrbit:
                pass
            elif not isinstance(args[0],list):
                os= [args[0]]
            else:
                os= args[0]
            if not hasattr(os[0],'orbit'): #not integrated yet
                [o.integrate(self._tsJ,pot=self._pot,
                             method=self._integrate_method) for o in os]
            ntJ= os[0].getOrbit().shape[0]
            no= len(os)
            R= nu.empty((no,ntJ))
            vR= nu.empty((no,ntJ))
            vT= nu.empty((no,ntJ))
            z= nu.empty((no,ntJ))
            vz= nu.empty((no,ntJ))
            phi= nu.empty((no,ntJ))
            for ii in range(len(os)):
                this_orbit= os[ii].getOrbit()
                R[ii,:]= this_orbit[:,0]
                vR[ii,:]= this_orbit[:,1]
                vT[ii,:]= this_orbit[:,2]
                z[ii,:]= this_orbit[:,3]
                vz[ii,:]= this_orbit[:,4]
                phi[ii,:]= this_orbit[:,5]
        return (R,vR,vT,z,vz,phi)

def estimateBIsochrone(R,z,pot=None):
    """
    NAME:
       estimateBIsochrone
    PURPOSE:
       Estimate a good value for the scale of the isochrone potential by matching the slope of the rotation curve
    INPUT:
       R,z = coordinates (if these are arrays, the median estimated delta is returned, i.e., if this is an orbit)
       pot= Potential instance or list thereof
    OUTPUT:
       b if 1 R,Z given
       bmin,bmedian,bmax if multiple R given       
    HISTORY:
       2013-09-12 - Written - Bovy (IAS)
    """
    if pot is None:
        raise IOError("pot= needs to be set to a Potential instance or list thereof")
    if isinstance(R,nu.ndarray):
        bs= nu.array([estimateBIsochrone(R[ii],z[ii],pot=pot) for ii in range(len(R))])
        return (nu.amin(bs[True-nu.isnan(bs)]),
                nu.median(bs[True-nu.isnan(bs)]),
                nu.amax(bs[True-nu.isnan(bs)]))
    else:
        r2= R**2.+z**2
        r= math.sqrt(r2)
        dlvcdlr= dvcircdR(pot,r)/vcirc(pot,r)*r
        try:
            b= optimize.brentq(lambda x: dlvcdlr-(x/math.sqrt(r2+x**2.)-0.5*r2/(r2+x**2.)),
                               0.01,100.)
        except:
            b= nu.nan
        return b

def dePeriod(arr):
    """make an array of periodic angles increase linearly"""
    diff= arr-nu.roll(arr,1)
    w= diff < -6.
    addto= nu.cumsum(w.astype(int))
    return arr+2.*nu.pi*addto
