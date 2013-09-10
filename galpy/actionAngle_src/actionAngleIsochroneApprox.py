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
import numpy as nu
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
           nonaxi= set to True to also calculate Lz using the isochrone 
                   approximation for non-axisymmetric potentials
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
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
