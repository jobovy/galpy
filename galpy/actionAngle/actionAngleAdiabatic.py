###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabatic
#
#             wrapper around actionAngleAxi (adiabatic approximation) to do
#             this for any (x,v)
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import copy
import warnings
import numpy
from ..util import galpyWarning
from ..potential import MWPotential
from ..potential.Potential import flatten as flatten_potential
from ..potential import toPlanarPotential, toVerticalPotential
from .actionAngleAxi import actionAngleAxi
from .actionAngle import actionAngle
from . import actionAngleAdiabatic_c
from .actionAngleAdiabatic_c import _ext_loaded as ext_loaded
from ..potential.Potential import _check_c
class actionAngleAdiabatic(actionAngle):
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleAdiabatic object

        INPUT:

           pot= potential or list of potentials (planarPotentials)

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
            raise IOError("Must specify pot= for actionAngleAxi")
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
            R= numpy.array([R])
            vR= numpy.array([vR])
            vT= numpy.array([vT])
            z= numpy.array([z])
            vz= numpy.array([vz])
        if ((self._c and not ('c' in kwargs and not kwargs['c']))\
                or (ext_loaded and (('c' in kwargs and kwargs['c'])))) \
                and _check_c(self._pot):
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
            if len(R) > 1:
                ojr= numpy.zeros((len(R)))
                olz= numpy.zeros((len(R)))
                ojz= numpy.zeros((len(R)))
                for ii in range(len(R)):
                    targs= (R[ii],vR[ii],vT[ii],z[ii],vz[ii])
                    tjr,tlz,tjz= self(*targs,**copy.copy(kwargs))
                    ojr[ii]= tjr
                    ojz[ii]= tjz
                    olz[ii]= tlz
                return (ojr,olz,ojz)
            else:
                #Set up the actionAngleAxi object
                thispot= toPlanarPotential(self._pot)
                thisverticalpot= toVerticalPotential(self._pot,R[0])
                aAAxi= actionAngleAxi(R[0],vR[0],vT[0],z[0],vz[0],
                                      pot=thispot,
                                       verticalPot=thisverticalpot,
                                       gamma=self._gamma)
                if kwargs.get('_justjr',False):
                    kwargs.pop('_justjr')
                    return (aAAxi.JR(**kwargs),numpy.nan,numpy.nan)
                elif kwargs.get('_justjz',False):
                    kwargs.pop('_justjz')
                    return (numpy.atleast_1d(numpy.nan),
                            numpy.atleast_1d(numpy.nan),
                            numpy.atleast_1d(aAAxi.Jz(**kwargs)))
                else:
                    return (numpy.atleast_1d(aAAxi.JR(**kwargs)),
                            numpy.atleast_1d(aAAxi._R*aAAxi._vT),
                            numpy.atleast_1d(aAAxi.Jz(**kwargs)))

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
            R= numpy.array([R])
            vR= numpy.array([vR])
            vT= numpy.array([vT])
            z= numpy.array([z])
            vz= numpy.array([vz])
        if ((self._c and not ('c' in kwargs and not kwargs['c']))\
                or (ext_loaded and (('c' in kwargs and kwargs['c'])))) \
                and _check_c(self._pot):
            rperi,Rap,zmax, err= actionAngleAdiabatic_c.actionAngleRperiRapZmaxAdiabatic_c(\
                self._pot,self._gamma,R,vR,vT,z,vz)
            if err == 0:
                rap= numpy.sqrt(Rap**2.+zmax**2.)
                ecc= (rap-rperi)/(rap+rperi)
                return (ecc,zmax,rperi,rap)
            else: #pragma: no cover
                raise RuntimeError("C-code for calculation actions failed; try with c=False")
        else:
            if 'c' in kwargs and kwargs['c'] and not self._c:
                warnings.warn("C module not used because potential does not have a C implementation",galpyWarning) #pragma: no cover
            kwargs.pop('c',None)
            if len(R) > 1:
                oecc= numpy.zeros((len(R)))
                orperi= numpy.zeros((len(R)))
                orap= numpy.zeros((len(R)))
                ozmax= numpy.zeros((len(R)))
                for ii in range(len(R)):
                    targs= (R[ii],vR[ii],vT[ii],z[ii],vz[ii])
                    tecc, tzmax, trperi,trap= self._EccZmaxRperiRap(\
                        *targs,**copy.copy(kwargs))
                    oecc[ii]= tecc
                    ozmax[ii]= tzmax
                    orperi[ii]= trperi
                    orap[ii]= trap
                return (oecc,ozmax,orperi,orap)
            else:
                #Set up the actionAngleAxi object
                thispot= toPlanarPotential(self._pot)
                thisverticalpot= toVerticalPotential(self._pot,R[0])
                aAAxi= actionAngleAxi(R[0],vR[0],vT[0],z[0],vz[0],
                                      pot=thispot,
                                       verticalPot=thisverticalpot,
                                       gamma=self._gamma)
                rperi,Rap= aAAxi.calcRapRperi(**kwargs)
                zmax= aAAxi.calczmax(**kwargs)
                rap= numpy.sqrt(Rap**2.+zmax**2.)
                return (numpy.atleast_1d((rap-rperi)/(rap+rperi)),
                        numpy.atleast_1d(zmax),numpy.atleast_1d(rperi),
                        numpy.atleast_1d(rap))

    def calcRapRperi(self,*args,**kwargs):
        """
        NAME:
           calcRapRperi
        PURPOSE:
           calculate the apocenter and pericenter radii
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
        OUTPUT:
           (rperi,rap)
        HISTORY:
           2013-11-27 - Written - Bovy (IAS)
        """
        #Set up the actionAngleAxi object
        thispot= toPlanarPotential(self._pot)
        aAAxi= actionAngleAxi(*args,pot=thispot,amma=self._gamma)
        return aAAxi.calcRapRperi(**kwargs)

    def calczmax(self,*args,**kwargs): #pragma: no cover
        """
        NAME:
           calczmax
        PURPOSE:
           calculate the maximum height
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
        OUTPUT:
           zmax
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
        """
        warnings.warn("actionAngleAdiabatic.calczmax function will soon be deprecated; please contact galpy's maintainer if you require this function")
        #Set up the actionAngleAxi object
        self._parse_eval_args(*args)
        thispot= toPlanarPotential(self._pot)
        thisverticalpot= toVerticalPotential(self._pot,self._eval_R)
        aAAxi= actionAngleAxi(*args,pot=thispot,
                               verticalPot=thisverticalpot,
                               gamma=self._gamma)
        return aAAxi.calczmax(**kwargs)
