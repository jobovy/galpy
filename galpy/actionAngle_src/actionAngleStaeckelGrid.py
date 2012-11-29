###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleStaeckelGrid
#
#             build grid in integrals of motion to quickly evaluate 
#             actionAngleStaeckel
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import math
import numpy
from scipy import interpolate, optimize, ndimage
import actionAngleStaeckel
from galpy.actionAngle import actionAngle, UnboundError
import galpy.potential
from galpy.util import multi
from matplotlib import pyplot
_PRINTOUTSIDEGRID= False
class actionAngleStaeckelGrid():
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation, grid-based interpolation"""
    def __init__(self,pot=None,delta=None,Rmax=5.,
                 nE=25,npsi=25,nLz=25,numcores=1,
                 **kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleStaeckelGrid object
        INPUT:
           pot= potential or list of potentials
           delta= focus of prolate confocal coordinate system
           Rmax = Rmax for building grids
           nE=, npsi=, nLz= grid size
           numcores= number of cpus to use to parallellize
           +scipy.integrate.quad keywords
        OUTPUT:
        HISTORY:
            2012-11-29 - Written - Bovy (IAS)
        """
        if pot is None:
            raise IOError("Must specify pot= for actionAngleStaeckelGrid")
        self._pot= pot
        if delta is None:
            raise IOError("Must specify delta= for actionAngleStaeckelGrid")
        self._delta= delta
        self._Rmax= Rmax
        self._Rmin= 0.01
        #Set up the actionAngleStaeckel object that we will use to interpolate
        self._aA= actionAngleStaeckel.actionAngleStaeckel(pot=self._pot,delta=self._delta)
        #Build grid
        self._Lzmin= 0.01
        self._Lzs= numpy.linspace(self._Lzmin,
                                  self._Rmax\
                                      *galpy.potential.vcirc(self._pot,
                                                             self._Rmax),
                                  nLz)
        self._Lzmax= self._Lzs[-1]
        #Calculate E_c(R=RL), energy of circular orbit
        self._RL= numpy.array([galpy.potential.rl(self._pot,l) for l in self._Lzs])
        self._RLInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                 self._RL,k=3)
        self._ERL= numpy.array([galpy.potential.evaluatePotentials(self._RL[ii],0.,self._pot) +self._Lzs[ii]**2./2./self._RL[ii]**2. for ii in range(nLz)])
        self._ERLmax= numpy.amax(self._ERL)+1.
        self._ERLInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                  numpy.log(-(self._ERL-self._ERLmax)),k=3)
        self._Ramax= 99.
        self._ERa= numpy.array([galpy.potential.evaluatePotentials(self._Ramax,0.,self._pot) +self._Lzs[ii]**2./2./self._Ramax**2. for ii in range(nLz)])
        self._ERamax= numpy.amax(self._ERa)+1.
        self._ERaInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                  numpy.log(-(self._ERa-self._ERamax)),k=3)
        y= numpy.linspace(0.,1.,nE)
        psis= numpy.linspace(0.,1.,npsi)*numpy.pi/2.
        jr= numpy.zeros((nLz,nE,npsi))
        jz= numpy.zeros((nLz,nE,npsi))
        u0= numpy.zeros((nLz,nE))
        jrLz= numpy.zeros(nLz)
        jzLz= numpy.zeros(nLz)
        if numcores > 1:
            raise NotImplementedError("'numcores > 1' not yet supported...")
            thisRL= (numpy.tile(self._RL,(nEr-1,1)).T).flatten()
            thisLzs= (numpy.tile(self._Lzs,(nEr-1,1)).T).flatten()
            thisERRL= (numpy.tile(self._ERRL,(nEr-1,1)).T).flatten()
            thisERRa= (numpy.tile(self._ERRa,(nEr-1,1)).T).flatten()
            thisy= (numpy.tile(y[0:-1],(nLz,1))).flatten()
            mjr= multi.parallel_map((lambda x: self._aA.JR(thisRL[x],
                                                          numpy.sqrt(2.*(thisERRa[x]+thisy[x]*(thisERRL[x]-thisERRa[x])-galpy.potential.evaluatePotentials(thisRL[x],0.,self._pot))-thisLzs[x]**2./thisRL[x]**2.),
                                                          thisLzs[x]/thisRL[x],
                                                          0.,0.,
                                                          **kwargs)[0]),
                                   range((nEr-1)*nLz),
                                   numcores=numcores)
            jr[:,0:-1]= numpy.reshape(mjr,(nLz,nEr-1))
            jrERRa[0:nLz]= jr[:,0]
        else:
            for ii in range(nLz):
                for jj in range(nE):
                    thisLz= self._Lzs[ii]
                    thisE= self._ERa[ii]+y[jj]*(self._ERL[ii]-self._ERa[ii])
                    u0[ii,jj]= self.calcu0(thisE,thisLz)
                    thisR= self._delta*numpy.sinh(u0[ii,jj])
                    thisv= self.vatu0(thisE,thisLz,u0[ii,jj],thisR)
                    print u0[ii,jj], thisR, thisv, thisE
                    for kk in range(npsi):
                        try:
                            jr[ii,jj,kk]= self._aA.JR(thisR, #R
                                                      thisv*numpy.cos(psis[kk]), #vR
                                                      thisLz/thisR, #vT
                                                      0., #z
                                                      thisv*numpy.sin(psis[kk]), #vz
self._RL[ii],
                                                      **kwargs)[0]
                            print jr[ii,jj,kk]
                        except UnboundError:
                            raise
                        #I know that calculating them independently is not 
                        # completely efficient
                        try:
                            jz[ii,jj,kk]= self._aA.Jz(thisR, #R
                                                      thisv*numpy.cos(psis[kk]), #vR
                                                      thisLz/thisR, #vT
                                                      0., #z
                                                      thisv*numpy.sin(psis[kk]), #vz
                                                      self._RL[ii],
                                                      **kwargs)[0]
                            print "jz", jz[ii,jj,kk]
                        except UnboundError:
                            raise
                #Normalize
                jrLz[ii]= numpy.amax(jr[ii,:,:])
                jr[ii,:,:]/= jrLz[ii]
                jzLz[ii]= numpy.amax(jz[ii,:,:])
                jz[ii,:,:]/= jzLz[ii]
        #First interpolate the maxima
        self._jr= jr
        self._jz= jz
        self._u0= u0
        self._jrLzInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                   numpy.log(jrLz+10.**-5.),k=3)
        self._jzLzInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                   numpy.log(jzLz+10.**-5.),k=3)
        #Interpolate u0
        self._u0Interp= interpolate.RectBivariateSpline(self._Lzs,
                                                        y,
                                                        u0,
                                                        kx=3,ky=3,s=0.)
        #spline filter jr and jz, such that they can be used with ndimage.map_coordinates
        self._jrFiltered= ndimage.spline_filter(self._jr)
        self._jzFiltered= ndimage.spline_filter(self._jz)
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
        #First work on the vertical action
        Phi= galpy.potential.evaluatePotentials(R,z,self._pot)
        Phio= galpy.potential.evaluatePotentials(R,0.,self._pot)
        Ez= Phi-Phio+vz**2./2.
        #Bigger than Ezzmax?
        thisEzZmax= numpy.exp(self._EzZmaxsInterp(R))
        if isinstance(R,numpy.ndarray):
            indx= (R > self._Rmax)
            indx+= (R < self._Rmin)
            indx+= (Ez != 0.)*(numpy.log(Ez) > thisEzZmax)
            indxc= True-indx
            jz= numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                jz[indxc]= (self._jzInterp.ev(R[indxc],Ez[indxc]/thisEzZmax[indxc])\
                                *(numpy.exp(self._jzEzmaxInterp(R[indxc]))-10.**-5.))
            if numpy.sum(indx) > 0:
                jzindiv= numpy.empty(numpy.sum(indx))
                for ii in range(numpy.sum(indx)):
                    try:
                        jzindiv[ii]= self._aA.Jz(R[indx][ii],0.,1.,#these two r dummies
                                                 0.,numpy.sqrt(2.*Ez[indx][ii]),
                                                 **kwargs)[0]
                    except UnboundError:
                        jzindiv[ii]= numpy.nan
                jz[indx]= jzindiv
        else:
            if R > self._Rmax or R < self._Rmin or (Ez != 0 and numpy.log(Ez) > thisEzZmax): #Outside of the grid
                if _PRINTOUTSIDEGRID:
                    print "Outside of grid in Ez", R > self._Rmax , R < self._Rmin , (Ez != 0 and numpy.log(Ez) > thisEzZmax)
                jz= self._aA.Jz(R,0.,1.,#these two r dummies
                                    0.,math.sqrt(2.*Ez),
                                    **kwargs)[0]
            else:
                jz= (self._jzInterp(R,Ez/thisEzZmax)\
                         *(numpy.exp(self._jzEzmaxInterp(R))-10.**-5.))[0][0]
        #Radial action
        ERLz= numpy.fabs(R*vT)+self._gamma*jz
        ER= Phio+vR**2./2.+ERLz**2./2./R**2.
        thisRL= self._RLInterp(ERLz)
        thisERRL= -numpy.exp(self._ERRLInterp(ERLz))+self._ERRLmax
        thisERRa= -numpy.exp(self._ERRaInterp(ERLz))+self._ERRamax
        if isinstance(R,numpy.ndarray):
            indx= ((ER-thisERRa)/(thisERRL-thisERRa) > 1.)\
                *(((ER-thisERRa)/(thisERRL-thisERRa)-1.) < 10.**-2.)
            ER[indx]= thisERRL[indx]
            indx= ((ER-thisERRa)/(thisERRL-thisERRa) < 0.)\
                *((ER-thisERRa)/(thisERRL-thisERRa) > -10.**-2.)
            ER[indx]= thisERRa[indx]
            indx= (ERLz < self._Lzmin)
            indx+= (ERLz > self._Lzmax)
            indx+= ((ER-thisERRa)/(thisERRL-thisERRa) > 1.)
            indx+= ((ER-thisERRa)/(thisERRL-thisERRa) < 0.)
            indxc= True-indx
            jr= numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                jr[indxc]= (self._jrInterp.ev(ERLz[indxc],
                                              (ER[indxc]-thisERRa[indxc])/(thisERRL[indxc]-thisERRa[indxc]))\
                                *(numpy.exp(self._jrERRaInterp(ERLz[indxc]))-10.**-5.))
            if numpy.sum(indx) > 0:
                jrindiv= numpy.empty(numpy.sum(indx))
                for ii in range(numpy.sum(indx)):
                    try:
                        jrindiv[ii]= self._aA.JR(thisRL[indx][ii],
                                                 numpy.sqrt(2.*(ER[indx][ii]-galpy.potential.evaluatePotentials(thisRL[indx][ii],0.,self._pot))-ERLz[indx][ii]**2./thisRL[indx][ii]**2.),
                                                 ERLz[indx][ii]/thisRL[indx][ii],
                                                 0.,0.,
                                                 **kwargs)[0]
                    except (UnboundError,OverflowError):
                        jrindiv[ii]= numpy.nan
                jr[indx]= jrindiv
        else:
            if (ER-thisERRa)/(thisERRL-thisERRa) > 1. \
                    and ((ER-thisERRa)/(thisERRL-thisERRa)-1.) < 10.**-2.:
                ER= thisERRL
            elif (ER-thisERRa)/(thisERRL-thisERRa) < 0. \
                    and (ER-thisERRa)/(thisERRL-thisERRa) > -10.**-2.:
                ER= thisERRa
            #Outside of grid?
            if ERLz < self._Lzmin or ERLz > self._Lzmax \
                    or (ER-thisERRa)/(thisERRL-thisERRa) > 1. \
                    or (ER-thisERRa)/(thisERRL-thisERRa) < 0.:
                if _PRINTOUTSIDEGRID:
                    print "Outside of grid in ER/Lz", ERLz < self._Lzmin , ERLz > self._Lzmax \
                        , (ER-thisERRa)/(thisERRL-thisERRa) > 1. \
                        , (ER-thisERRa)/(thisERRL-thisERRa) < 0., ER, thisERRL, thisERRa, (ER-thisERRa)/(thisERRL-thisERRa)
                jr= self._aA.JR(thisRL,
                                numpy.sqrt(2.*(ER-galpy.potential.evaluatePotentials(thisRL,0.,self._pot))-ERLz**2./thisRL**2.),
                                ERLz/thisRL,
                                0.,0.,
                                **kwargs)[0]
            else:
                jr= (self._jrInterp(ERLz,
                                    (ER-thisERRa)/(thisERRL-thisERRa))\
                         *(numpy.exp(self._jrERRaInterp(ERLz))-10.**-5.))[0][0]
        return (jr,R*vT,jz)

    def Jz(self,*args,**kwargs):
        """
        NAME:
           Jz
        PURPOSE:
           evaluate the action jz
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           jz
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        meta= actionAngle(*args)
        Phi= galpy.potential.evaluatePotentials(meta._R,meta._z,self._pot)
        Phio= galpy.potential.evaluatePotentials(meta._R,0.,self._pot)
        Ez= Phi-Phio+meta._vz**2./2.
        #Bigger than Ezzmax?
        thisEzZmax= numpy.exp(self._EzZmaxsInterp(meta._R))
        if meta._R > self._Rmax or meta._R < self._Rmin or (Ez != 0. and numpy.log(Ez) > thisEzZmax): #Outside of the grid
            if _PRINTOUTSIDEGRID:
                print "Outside of grid in Ez"
            jz= self._aA.Jz(meta._R,0.,1.,#these two r dummies
                            0.,math.sqrt(2.*Ez),
                            **kwargs)[0]
        else:
            jz= (self._jzInterp(meta._R,Ez/thisEzZmax)\
                *(numpy.exp(self._jzEzmaxInterp(meta._R))-10.**-5.))[0][0]
        return jz

    def vatu0(self,E,Lz,u0,R):
        """
        NAME:
           vatu0
        PURPOSE:
           calculate the velocity at u0
        INPUT:
           E - energy
           Lz - angular momentum
           u0 - u0
        OUTPUT:
           velocity
        HISTORY:
           2012-11-29 - Written - Bovy (IAS)
        """                        
        v2= (2.*(E-actionAngleStaeckel.potentialStaeckel(u0,numpy.pi/2.,
                                                         self._pot,
                                                         self._delta))
             -Lz**2./R**2.)
        if isinstance(E,float) and v2 < 0. and v2 > -10.**-7.: 
            return 0. #rounding errors
        elif isinstance(E,float):
            return numpy.sqrt(v2)
        elif isinstance(v2,numpy.ndarray):
            v2[(v2 < 0.)*(v2 > -10.**-7.)]= 0.
            return numpy.sqrt(v2)
    
    def calcu0(self,E,Lz):
        """
        NAME:
           calcu0
        PURPOSE:
           calculate the minimum of the u potential
        INPUT:
           E - energy
           Lz - angular momentum
        OUTPUT:
           u0
        HISTORY:
           2012-11-29 - Written - Bovy (IAS)
        """                           
        logu0= optimize.brent(_u0Eq,
                              args=(self._delta,self._pot,
                                    E,Lz**2./2.))
        return numpy.exp(logu0)

def _u0Eq(logu,delta,pot,E,Lz22):
    """The equation that needs to be minimized to find u0"""
    u= numpy.exp(logu)
    sinh2u= numpy.sinh(u)**2.
    cosh2u= numpy.cosh(u)**2.
    dU= cosh2u*actionAngleStaeckel.potentialStaeckel(u,numpy.pi/2.,pot,delta)
    return -(E*sinh2u-dU-Lz22/delta**2./sinh2u)

