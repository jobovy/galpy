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
        Rs= numpy.linspace(0.01,self._Rmax,nR)
        EzZmaxs= numpy.array([evaluatePotentials(r,self._zmax,self._pot)-
                              evaluatePotentials(r,0.,self._pot) for r in Rs])
        self._EzZmaxsInterp= interpolate.InterpolatedUnivariateSpline(Rs,numpy.log(EzZmaxs),k=3)
        x= numpy.zeros(nR*nEz)
        y= numpy.zeros(nR*nEz)
        jz= numpy.zeros(nR*nEz)
        jzEzzmax= numpy.zeros(nR)
        jzEzzmax2d= numpy.zeros(nR*nEz)
        for ii in range(nR):
            Ezs= numpy.linspace(EzZmaxs[ii],0.,nEz)
            for jj in range(nEz):
                x[ii*nEz+jj]= Rs[ii]
                y[ii*nEz+jj]= Ezs[jj]
                #Calculate Jz
                jz[ii*nEz+jj]= self._aA.Jz(Rs[ii],0.,1.,#these two r dummies
                                           0.,math.sqrt(2.*Ezs[jj]),
                                           **kwargs)[0]
                if jj == 0: 
                    jzEzzmax[ii]= jz[ii*nEz+jj]
                jz[ii*nEz+jj]/= jzEzzmax[ii]
        #First interpolate Ez=Ezmax
        self._jzEzmaxInterp= interpolate.InterpolatedUnivariateSpline(Rs,numpy.log(jzEzzmax+10.**-5.),k=3)
        #Divide this out
        self._jzInterp= interpolate.bisplrep(x,
                                             y,
                                             jz,
                                             kx=3,ky=3)
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
        """
        meta= actionAngle(*args)
        #First work on the vertical action
        Phi= self._pot(meta._R,meta._z)
        Phio= self._pot(meta._R,0.)
        Ez= Phi-Phio+meta._vz**2./2.
        #Bigger than Ezzmax?
        thisEzZmax= self._EzZmaxsInterp(meta._R)
        if numpy.log(Ez) > thisEzZmax: #Outside of the grid
            print "Outside of grid"
            jz= self._aA.Jz(meta._R,0.,1.,#these two r dummies
                            0.,math.sqrt(2.*Ez),
                            **kwargs)[0]
        else:
            jz= interpolate.bisplev(meta._R,Ez,self._jzInterp)\
                *numpy.exp(self._jzEzmaxInterp(meta._R))
        return jz
