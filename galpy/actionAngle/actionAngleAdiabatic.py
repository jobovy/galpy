###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabatic
#
#      methods:
#             __call__: returns (jr,lz,jz)
#            _EccZmaxRperiRap: return (e,zmax,rperi,rap)
#
###############################################################################
import copy
import warnings
import numpy as nu
from galpy.util import galpyWarning
from galpy.potential import planarPotential, MWPotential
from galpy.potential.Potential import flatten as flatten_potential
from .actionAngleSpherical import actionAngleSpherical
from .actionAngleVertical import actionAngleVertical
from .actionAngle import actionAngle
from . import actionAngleAdiabatic_c
from .actionAngleAdiabatic_c import _ext_loaded as ext_loaded
from galpy.potential.Potential import _check_c, _dim
class actionAngleAdiabatic(actionAngle):
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleAdiabatic object

        INPUT:

           pot= potential or list of potentials

           gamma= (default=1.) replace Lz by Lz+gamma Jz in effective potential

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

            2012-07-26 - Written - Bovy (IAS@MPIA)

        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleAdiabatic")
        self._pot= flatten_potential(kwargs['pot'])
        if self._pot == MWPotential:
            warnings.warn("Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy",
                          galpyWarning)
        if ext_loaded and 'c' in kwargs and kwargs['c']:
            self._c= _check_c(self._pot)
            if 'c' in kwargs and kwargs['c'] and not self._c:
                warnings.warn("C module not used because potential does not have a C implementation",galpyWarning) #pragma: no cover
        else:
            self._c= False
        self._gamma= kwargs.get('gamma',1.)
        # Setup actionAngleSpherical object for calculations in Python 
        # (if they become necessary)
        if _dim(self._pot) == 3:
            if isinstance(self._pot,list):
                thispot= [p.toPlanar() for p in self._pot]
            else:
                thispot= self._pot.toPlanar()
        else:
            thispot= self._pot
            self._gamma= 0.
        self._aAS= actionAngleSpherical(pot=thispot,_gamma=self._gamma)
        # Check the units
        self._check_consistent_units()
        return None
    
    def _evaluate(self,*args,**kwargs):
        """
        NAME:
           __call__ (_evaluate)
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument 
           c= (object-wide default, bool) True/False to override the object-wide setting for whether or not to use the C implementation
           scipy.integrate.quadrature keywords
           _justjr, _justjz= if True, only calculate the radial or vertical action (internal use)
        OUTPUT:
           (jr,lz,jz)
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        if ((self._c and not ('c' in kwargs and not kwargs['c']))\
                or (ext_loaded and (('c' in kwargs and kwargs['c'])))) \
                and _check_c(self._pot):
            if len(args) == 5: #R,vR.vT, z, vz
                R,vR,vT, z, vz= args
            elif len(args) == 6: #R,vR.vT, z, vz, phi
                R,vR,vT, z, vz, phi= args
            else:
                self._parse_eval_args(*args)
                R= self._eval_R
                vR= self._eval_vR
                vT= self._eval_vT
                z= self._eval_z
                vz= self._eval_vz
            if isinstance(R,float):
                R= nu.array([R])
                vR= nu.array([vR])
                vT= nu.array([vT])
                z= nu.array([z])
                vz= nu.array([vz])
            Lz= R*vT
            jr, jz, err= actionAngleAdiabatic_c.actionAngleAdiabatic_c(\
                self._pot,self._gamma,R,vR,vT,z,vz)
            if err == 0:
                return (jr,Lz,jz)
            else: #pragma: no cover
                raise RuntimeError("C-code for calculation actions failed; try with c=False")
        else:
            if 'c' in kwargs and kwargs['c'] and not self._c:
                warnings.warn("C module not used because potential does not have a C implementation",galpyWarning) #pragma: no cover
            kwargs.pop('c',None)
            if (len(args) == 5 or len(args) == 6) \
                    and isinstance(args[0],nu.ndarray):
                ojr= nu.zeros((len(args[0])))
                olz= nu.zeros((len(args[0])))
                ojz= nu.zeros((len(args[0])))
                for ii in range(len(args[0])):
                    if len(args) == 5:
                        targs= (args[0][ii],args[1][ii],args[2][ii],
                                args[3][ii],args[4][ii])
                    elif len(args) == 6:
                        targs= (args[0][ii],args[1][ii],args[2][ii],
                                args[3][ii],args[4][ii],args[5][ii])
                    tjr,tlz,tjz= self(*targs,**copy.copy(kwargs))
                    ojr[ii]= tjr
                    ojz[ii]= tjz
                    olz[ii]= tlz
                return (ojr,olz,ojz)
            else:
                self._parse_eval_args(*args)
                if kwargs.get('_justjr',False):
                    kwargs.pop('_justjr')
                    return (self._aAS(self._eval_R,self._eval_vR,self._eval_vT,
                                      0.,0.,_Jz=0.)[0],
                            nu.nan,nu.nan)
                #Set up the actionAngleVertical object
                if _dim(self._pot) == 3:
                    if isinstance(self._pot,list):
                        thisverticalpot= [p.toVertical(self._eval_R) 
                                          for p in self._pot]
                    else:
                        thisverticalpot= self._pot.toVertical(self._eval_R)
                    aAV= actionAngleVertical(pot=thisverticalpot)
                    Jz= aAV(self._eval_z,self._eval_vz)
                else: #2D in-plane
                    Jz= 0.
                if kwargs.get('_justjz',False):
                    kwargs.pop('_justjz')
                    return (nu.nan,nu.nan,Jz)
                else:
                    axiJ= self._aAS(self._eval_R,self._eval_vR,self._eval_vT,
                                    0.,0.,_Jz=Jz)
                    return (axiJ[0],axiJ[1],Jz)

    def _EccZmaxRperiRap(self,*args,**kwargs):
        """
        NAME:
           EccZmaxRperiRap (_EccZmaxRperiRap)
        PURPOSE:
           evaluate the eccentricity, maximum height above the plane, peri- and apocenter in the adiabatic approximation
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument 
           c= (object-wide default, bool) True/False to override the object-wide setting for whether or not to use the C implementation
        OUTPUT:
           (e,zmax,rperi,rap)
        HISTORY:
           2017-12-21 - Written - Bovy (UofT)
        """
        if ((self._c and not ('c' in kwargs and not kwargs['c']))\
                or (ext_loaded and (('c' in kwargs and kwargs['c'])))) \
                and _check_c(self._pot):
            if len(args) == 5: #R,vR.vT, z, vz
                R,vR,vT, z, vz= args
            elif len(args) == 6: #R,vR.vT, z, vz, phi
                R,vR,vT, z, vz, phi= args
            else:
                self._parse_eval_args(*args)
                R= self._eval_R
                vR= self._eval_vR
                vT= self._eval_vT
                z= self._eval_z
                vz= self._eval_vz
            if isinstance(R,float):
                R= nu.array([R])
                vR= nu.array([vR])
                vT= nu.array([vT])
                z= nu.array([z])
                vz= nu.array([vz])
            rperi,Rap,zmax, err= actionAngleAdiabatic_c.actionAngleRperiRapZmaxAdiabatic_c(\
                self._pot,self._gamma,R,vR,vT,z,vz)
            if err == 0:
                rap= nu.sqrt(Rap**2.+zmax**2.)
                ecc= (rap-rperi)/(rap+rperi)
                return (ecc,zmax,rperi,rap)
            else: #pragma: no cover
                raise RuntimeError("C-code for calculation actions failed; try with c=False")
        else:
            if 'c' in kwargs and kwargs['c'] and not self._c:
                warnings.warn("C module not used because potential does not have a C implementation",galpyWarning) #pragma: no cover
            kwargs.pop('c',None)
            if (len(args) == 5 or len(args) == 6) \
                    and isinstance(args[0],nu.ndarray):
                oecc= nu.zeros((len(args[0])))
                orperi= nu.zeros((len(args[0])))
                orap= nu.zeros((len(args[0])))
                ozmax= nu.zeros((len(args[0])))
                for ii in range(len(args[0])):
                    if len(args) == 5:
                        targs= (args[0][ii],args[1][ii],args[2][ii],
                                args[3][ii],args[4][ii])
                    elif len(args) == 6:
                        targs= (args[0][ii],args[1][ii],args[2][ii],
                                args[3][ii],args[4][ii],args[5][ii])
                    tecc, tzmax, trperi,trap= self._EccZmaxRperiRap(\
                        *targs,**copy.copy(kwargs))
                    oecc[ii]= tecc
                    ozmax[ii]= tzmax
                    orperi[ii]= trperi
                    orap[ii]= trap
                return (oecc,ozmax,orperi,orap)
            else:
                self._parse_eval_args(*args)
                if _dim(self._pot) == 3:
                    if isinstance(self._pot,list):
                        thisverticalpot= [p.toVertical(self._eval_R) for p in self._pot]
                    else:
                        thisverticalpot= self._pot.toVertical(self._eval_R)
                    aAV= actionAngleVertical(pot=thisverticalpot)
                    zmax= aAV.calcxmax(self._eval_z,self._eval_vz,**kwargs)
                    if self._gamma != 0.:
                        Jz= aAV(self._eval_z,self._eval_vz)
                    else:
                        Jz= 0.
                else:
                    zmax= 0.
                    Jz= 0.
                _,_,rperi,Rap= self._aAS.EccZmaxRperiRap(\
                    self._eval_R,self._eval_vR,self._eval_vT,0.,0.,_Jz=Jz)
                rap= nu.sqrt(Rap**2.+zmax**2.)
                return ((rap-rperi)/(rap+rperi),zmax,rperi,rap)
