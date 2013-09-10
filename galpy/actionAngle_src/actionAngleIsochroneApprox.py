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
           ntintJ= (default: 1000) number of time-integration points
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
            self._ntintJ= 1000
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
                lz= nu.sumFunc(lzI*danglephiI,axis=1)/sumFunc(danglephiI,axis=1)
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
        if len(args) == 5: #R,vR.vT, z, vz
            R,vR,vT, z, vz= args
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
        else:
            meta= actionAngle(*args)
            R= meta._R
            vR= meta._vR
            vT= meta._vT
            z= meta._z
            vz= meta._vz
            phi= meta._phi
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
            phi= nu.array([phi])
        if self._c:
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            #Actions
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            Jr= self.amp/nu.sqrt(-2.*E)\
                -0.5*(L+nu.sqrt((L2+4.*self.amp*self.b)))
            #Frequencies
            Omegar= (-2.*E)**1.5/self.amp
            Omegaz= 0.5*(1.+L/nu.sqrt(L2+4.*self.amp*self.b))*Omegar
            Omegaphi= copy.copy(Omegaz)
            indx= Lz < 0.
            Omegaphi[indx]*= -1.
            #Angles
            c= -self.amp/2./E-self.b
            e2= 1.-L2/self.amp/c*(1.+self.b/c)
            e= nu.sqrt(e2)
            s= 1.+nu.sqrt(1.+(R**2.+z**2.)/self.b**2.)
            coseta= 1/e*(1.-self.b/c*(s-2.))
            eta= nu.arccos(coseta)
            costheta= z/nu.sqrt(R**2.+z**2.)
            sintheta= R/nu.sqrt(R**2.+z**2.)
            vrindx= (vR*sintheta+vz*costheta) < 0.
            eta[vrindx]= 2.*nu.pi-eta[vrindx]
            angler= eta-e*c/(c+self.b)*nu.sin(eta)
            tan11= nu.arctan(nu.sqrt((1.+e)/(1.-e))*nu.tan(0.5*eta))
            tan12= nu.arctan(nu.sqrt((1.+e+2.*self.b/c)/(1.-e+2.*self.b/c))*nu.tan(0.5*eta))
            vzindx= (-vz*sintheta+vR*costheta) > 0.
            tan11[tan11 < 0.]+= nu.pi
            tan12[tan12 < 0.]+= nu.pi
            i= nu.arccos(Lz/L)
            sinpsi= costheta/nu.sin(i)
            psi= nu.arcsin(sinpsi)
            psi[vzindx]= nu.pi-psi[vzindx]
            psi= psi % (2.*nu.pi)
            anglez= psi+Omegaz/Omegar*angler\
                -tan11-1./nu.sqrt(1.+4*self.amp*self.b/L2)*tan12
            sinu= z/R/nu.tan(i)
            u= nu.arcsin(sinu)
            u[vzindx]= nu.pi-u[vzindx]
            Omega= phi-u
            anglephi= Omega
            anglephi[indx]-= anglez[indx]
            anglephi[True-indx]+= anglez[True-indx]
            angler= angler % (2.*nu.pi)
            anglephi= anglephi % (2.*nu.pi)
            anglez= anglez % (2.*nu.pi)
            return (Jr,Jphi,Jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

    def _parse_args(self,*args):
        """Helper function to parse the arguments to the __call__ and actionsFreqsAngles functions"""
        if len(args) == 5:
            raise IOError("Must specify phi for actionAngleIsochroneApprox")
        if len(args) == 6:
            R,vR,vT, z, vz, phi= args
            RasOrbit= False
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
