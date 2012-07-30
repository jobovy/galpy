###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAdiabaticGrid
#
#             build grid in integrals of motion to quickly evaluate 
#             actionAngleAdiabatic
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import math
import numpy
from scipy import interpolate
from actionAngleAdiabatic import actionAngleAdiabatic
from actionAngle import actionAngle
from galpy.potential import evaluatePotentials
from matplotlib import pyplot
class actionAngleAdiabaticGrid():
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation, grid-based interpolation"""
    def __init__(self,pot=None,zmax=3./8.,gamma=1.,Rmax=3.,
                 nR=25,nEz=25,nEr=25,nLz=25,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleAdiabaticGrid object
        INPUT:
           pot= potential or list of potentials (planarPotentials)
           zmax= zmax for building Ez grid
           Rmax = Rmax for building grids
           gamma= (default=1.) replace Lz by Lz+gamma Jz in effective potential
           nEz=, nEr=, nLz, nR= grid size
           +scipy.integrate.quad keywords
        OUTPUT:
        HISTORY:
            2012-07-27 - Written - Bovy (IAS@MPIA)
        """
        if pot is None:
            raise IOError("Must specify pot= for actionAngleAxi")
        self._gamma= gamma
        self._pot= pot
        self._zmax= zmax
        self._Rmax= Rmax
        #Set up the actionAngleAdiabatic object that we will use to interpolate
        self._aA= actionAngleAdiabatic(pot=self._pot,gamma=self._gamma)
        #Build grid for Ez, first calculate Ez(zmax;R) function
        self._Rs= numpy.linspace(0.01,self._Rmax,nR)
        self._EzZmaxs= numpy.array([evaluatePotentials(r,self._zmax,self._pot)-
                              evaluatePotentials(r,0.,self._pot) for r in self._Rs])
        self._EzZmaxsInterp= interpolate.InterpolatedUnivariateSpline(self._Rs,numpy.log(self._EzZmaxs),k=3)
        y= numpy.linspace(0.,1.,nEz)
        jz= numpy.zeros((nR,nEz))
        jzEzzmax= numpy.zeros(nR)
        for ii in range(nR):
            for jj in range(nEz):
                #Calculate Jz
                jz[ii,jj]= self._aA.Jz(self._Rs[ii],0.,1.,#these two r dummies
                                       0.,math.sqrt(2.*y[jj]*self._EzZmaxs[ii]),
                                       **kwargs)[0]
                if jj == nEz-1: 
                    jzEzzmax[ii]= jz[ii,jj]
        for ii in range(nR): jz[ii,:]/= jzEzzmax[ii]
        #First interpolate Ez=Ezmax
        self._jzEzmaxInterp= interpolate.InterpolatedUnivariateSpline(self._Rs,numpy.log(jzEzzmax+10.**-5.),k=3)
        self._jzInterp= interpolate.RectBivariateSpline(self._Rs,
                                                        y,
                                                        jz,
                                                        kx=3,ky=3,s=0.)
        return None

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           (jr,lz,jz)
        HISTORY:
           2012-07-27 - Written - Bovy (IAS@MPIA)
        NOTE:
           For a Miyamoto-Nagai potential, this seems accurate to 0.1% and takes ~0.13 ms
           For a MWPotential, this takes ~ 0.17 ms
        """
        meta= actionAngle(*args)
        #First work on the vertical action
        Phi= evaluatePotentials(meta._R,meta._z,self._pot)
        Phio= evaluatePotentials(meta._R,0.,self._pot)
        Ez= Phi-Phio+meta._vz**2./2.
        #Bigger than Ezzmax?
        thisEzZmax= numpy.exp(self._EzZmaxsInterp(meta._R))
        if numpy.log(Ez) > thisEzZmax: #Outside of the grid
            print "Outside of grid"
            jz= self._aA.Jz(meta._R,0.,1.,#these two r dummies
                            0.,math.sqrt(2.*Ez),
                            **kwargs)[0]
        else:
            jz= self._jzInterp(meta._R,Ez/thisEzZmax)\
                *numpy.exp(self._jzEzmaxInterp(meta._R))
        return jz
