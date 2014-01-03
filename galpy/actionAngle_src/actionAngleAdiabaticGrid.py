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
from galpy.actionAngle_src.actionAngle import actionAngle, UnboundError
import galpy.potential
from galpy.util import multi
from matplotlib import pyplot
_PRINTOUTSIDEGRID= False
class actionAngleAdiabaticGrid():
    """Action-angle formalism for axisymmetric potentials using the adiabatic approximation, grid-based interpolation"""
    def __init__(self,pot=None,zmax=1.,gamma=1.,Rmax=5.,
                 nR=16,nEz=16,nEr=31,nLz=31,numcores=1,
                 **kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleAdiabaticGrid object
        INPUT:

           pot= potential or list of potentials

           zmax= zmax for building Ez grid

           Rmax = Rmax for building grids

           gamma= (default=1.) replace Lz by Lz+gamma Jz in effective potential

           nEz=, nEr=, nLz, nR= grid size

           numcores= number of cpus to use to parallellize

           c= if True, use C to calculate actions

           +scipy.integrate.quad keywords
        OUTPUT:
        HISTORY:
            2012-07-27 - Written - Bovy (IAS@MPIA)
        """
        if pot is None:
            raise IOError("Must specify pot= for actionAngleAxi")
        if kwargs.has_key('c') and kwargs['c']:
            self._c= True
            kwargs.pop('c')
        else:
            self._c= False
            if kwargs.has_key('c'): kwargs.pop('c')
        self._gamma= gamma
        self._pot= pot
        self._zmax= zmax
        self._Rmax= Rmax
        self._Rmin= 0.01
        #Set up the actionAngleAdiabatic object that we will use to interpolate
        self._aA= actionAngleAdiabatic(pot=self._pot,gamma=self._gamma,
                                       c=self._c)
        #Build grid for Ez, first calculate Ez(zmax;R) function
        self._Rs= numpy.linspace(self._Rmin,self._Rmax,nR)
        try:
            self._EzZmaxs= galpy.potential.evaluatePotentials(self._Rs,self._zmax*numpy.ones(nR),self._pot)\
                -galpy.potential.evaluatePotentials(self._Rs,numpy.zeros(nR),self._pot)
        except TypeError:
            self._EzZmaxs= numpy.array([galpy.potential.evaluatePotentials(r,self._zmax,self._pot)-
                                        galpy.potential.evaluatePotentials(r,0.,self._pot) for r in self._Rs])
        self._EzZmaxsInterp= interpolate.InterpolatedUnivariateSpline(self._Rs,numpy.log(self._EzZmaxs),k=3)
        y= numpy.linspace(0.,1.,nEz)
        jz= numpy.zeros((nR,nEz))
        jzEzzmax= numpy.zeros(nR)
        thisRs= (numpy.tile(self._Rs,(nEz,1)).T).flatten()
        thisEzZmaxs= (numpy.tile(self._EzZmaxs,(nEz,1)).T).flatten()
        thisy= (numpy.tile(y,(nR,1))).flatten()
        if self._c:
            jz= self._aA(thisRs,
                         numpy.zeros(len(thisRs)),
                         numpy.ones(len(thisRs)),#these two r dummies
                         numpy.zeros(len(thisRs)),
                         numpy.sqrt(2.*thisy*thisEzZmaxs),
                         **kwargs)[2]
            jz= numpy.reshape(jz,(nR,nEz))
            jzEzzmax[0:nR]= jz[:,nEz-1]
        else:
            if numcores > 1:
                jz= multi.parallel_map((lambda x: self._aA.Jz(thisRs[x],0.,1.,#these two r dummies
                                                              0.,math.sqrt(2.*thisy[x]*thisEzZmaxs[x]),
                                                              **kwargs)[0]),
                                       range(nR*nEz),numcores=numcores)
                jz= numpy.reshape(jz,(nR,nEz))
                jzEzzmax[0:nR]= jz[:,nEz-1]
            else:
                for ii in range(nR):
                    for jj in range(nEz):
                    #Calculate Jz
                        jz[ii,jj]= self._aA.Jz(self._Rs[ii],0.,1.,#these two r dummies
                                               0.,numpy.sqrt(2.*y[jj]*self._EzZmaxs[ii]),
                                               **kwargs)[0]
                        if jj == nEz-1: 
                            jzEzzmax[ii]= jz[ii,jj]
        for ii in range(nR): jz[ii,:]/= jzEzzmax[ii]
        #First interpolate Ez=Ezmax
        self._jzEzmaxInterp= interpolate.InterpolatedUnivariateSpline(self._Rs,numpy.log(jzEzzmax+10.**-5.),k=3)
        self._jz= jz
        self._jzInterp= interpolate.RectBivariateSpline(self._Rs,
                                                        y,
                                                        jz,
                                                        kx=3,ky=3,s=0.)
        #JR grid
        self._Lzmin= 0.01
        self._Lzs= numpy.linspace(self._Lzmin,
                                  self._Rmax\
                                      *galpy.potential.vcirc(self._pot,
                                                             self._Rmax),
                                  nLz)
        self._Lzmax= self._Lzs[-1]
        #Calculate ER(vr=0,R=RL)
        self._RL= numpy.array([galpy.potential.rl(self._pot,l) for l in self._Lzs])
        self._RLInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                 self._RL,k=3)
        try:
            self._ERRL= galpy.potential.evaluatePotentials(self._RL,numpy.zeros(nLz),self._pot) +self._Lzs**2./2./self._RL**2.
        except TypeError:
            self._ERRL= numpy.array([galpy.potential.evaluatePotentials(self._RL[ii],0.,self._pot) +self._Lzs[ii]**2./2./self._RL[ii]**2. for ii in range(nLz)])
        self._ERRLmax= numpy.amax(self._ERRL)+1.
        self._ERRLInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                   numpy.log(-(self._ERRL-self._ERRLmax)),k=3)
        self._Ramax= 99.
        try:
            self._ERRa= galpy.potential.evaluatePotentials(self._Ramax,0.,self._pot) +self._Lzs**2./2./self._Ramax**2.
        except TypeError:
            self._ERRa= numpy.array([galpy.potential.evaluatePotentials(self._Ramax,0.,self._pot) +self._Lzs[ii]**2./2./self._Ramax**2. for ii in range(nLz)])
        self._ERRamax= numpy.amax(self._ERRa)+1.
        self._ERRaInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                   numpy.log(-(self._ERRa-self._ERRamax)),k=3)
        y= numpy.linspace(0.,1.,nEr)
        jr= numpy.zeros((nLz,nEr))
        jrERRa= numpy.zeros(nLz)
        thisRL= (numpy.tile(self._RL,(nEr-1,1)).T).flatten()
        thisLzs= (numpy.tile(self._Lzs,(nEr-1,1)).T).flatten()
        thisERRL= (numpy.tile(self._ERRL,(nEr-1,1)).T).flatten()
        thisERRa= (numpy.tile(self._ERRa,(nEr-1,1)).T).flatten()
        thisy= (numpy.tile(y[0:-1],(nLz,1))).flatten()
        if self._c:
            mjr= self._aA(thisRL,
                          numpy.sqrt(2.*(thisERRa+thisy*(thisERRL-thisERRa)-galpy.potential.evaluatePotentials(thisRL,numpy.zeros((nEr-1)*nLz),self._pot))-thisLzs**2./thisRL**2.),
                          thisLzs/thisRL,
                          numpy.zeros(len(thisRL)),
                          numpy.zeros(len(thisRL)),
                          **kwargs)[0]
            jr[:,0:-1]= numpy.reshape(mjr,(nLz,nEr-1))
            jrERRa[0:nLz]= jr[:,0]
        else:
            if numcores > 1:
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
                    for jj in range(nEr-1): #Last one is zero by construction
                        try:
                            jr[ii,jj]= self._aA.JR(self._RL[ii],
                                                   numpy.sqrt(2.*(self._ERRa[ii]+y[jj]*(self._ERRL[ii]-self._ERRa[ii])-galpy.potential.evaluatePotentials(self._RL[ii],0.,self._pot))-self._Lzs[ii]**2./self._RL[ii]**2.),
                                                   self._Lzs[ii]/self._RL[ii],
                                                   0.,0.,
                                                   **kwargs)[0]
                        except UnboundError:
                            raise
                        if jj == 0: 
                            jrERRa[ii]= jr[ii,jj]
        for ii in range(nLz): jr[ii,:]/= jrERRa[ii]
        #First interpolate Ez=Ezmax
        self._jr= jr
        self._jrERRaInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                     numpy.log(jrERRa+10.**-5.),k=3)
        self._jrInterp= interpolate.RectBivariateSpline(self._Lzs,
                                                        y,
                                                        jr,
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
        try:
            Phio= galpy.potential.evaluatePotentials(R,numpy.zeros(len(R)),self._pot)
        except TypeError:
            Phio= galpy.potential.evaluatePotentials(R,0.,self._pot)
        Ez= Phi-Phio+vz**2./2.
        #Bigger than Ezzmax?
        thisEzZmax= numpy.exp(self._EzZmaxsInterp(R))
        if isinstance(R,numpy.ndarray):
            if len(R) == 1 and not isinstance(thisEzZmax,numpy.ndarray):
                thisEzZmax= numpy.array([thisEzZmax])
            indx= (R > self._Rmax)
            indx+= (R < self._Rmin)
            indx+= (Ez != 0.)*(numpy.log(Ez) > thisEzZmax)
            indxc= True-indx
            jz= numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                jz[indxc]= (self._jzInterp.ev(R[indxc],Ez[indxc]/thisEzZmax[indxc])\
                                *(numpy.exp(self._jzEzmaxInterp(R[indxc]))-10.**-5.))
            if numpy.sum(indx) > 0:
                jz[indx]= self._aA(R[indx],
                                   numpy.zeros(numpy.sum(indx)),
                                   numpy.ones(numpy.sum(indx)),#these two r dummies
                                   numpy.zeros(numpy.sum(indx)),
                                   numpy.sqrt(2.*Ez[indx]),
                                   **kwargs)[2]
            """
                for ii in range(numpy.sum(indx)):
                    try:
                        jzindiv[ii]= self._aA.Jz(R[indx][ii],0.,1.,#these two r dummies
                                                 0.,numpy.sqrt(2.*Ez[indx][ii]),
                                                 **kwargs)[0]
                    except UnboundError:
                        jzindiv[ii]= numpy.nan
            jz[indx]= jzindiv
            """
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
            if len(R) == 1 and not isinstance(thisRL,numpy.ndarray):
                thisRL= numpy.array([thisRL])
                thisERRL= numpy.array([thisERRL])
                thisERRa= numpy.array([thisERRa])
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
                jr[indx]= self._aA(thisRL[indx],
                                   numpy.sqrt(2.*(ER[indx]-galpy.potential.evaluatePotentials(thisRL[indx],0.,self._pot))-ERLz[indx]**2./thisRL[indx]**2.),
                                   ERLz[indx]/thisRL[indx],
                                   numpy.zeros(len(thisRL)),
                                   numpy.zeros(len(thisRL)),
                                   **kwargs)[0]                
                """
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
                """
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
