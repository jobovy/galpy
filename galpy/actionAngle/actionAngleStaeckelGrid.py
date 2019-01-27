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
import numpy
from scipy import interpolate, optimize, ndimage
from . import actionAngleStaeckel
from .actionAngle import actionAngle
from . import actionAngleStaeckel_c
from .actionAngleStaeckel_c import _ext_loaded as ext_loaded
import galpy.potential
from galpy.potential.Potential import _evaluatePotentials
from galpy.potential.Potential import flatten as flatten_potential
from galpy.util import multi, bovy_coords
from galpy.util.bovy_conversion import physical_conversion_actionAngle, \
    actionAngle_physical_input
_PRINTOUTSIDEGRID= False
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class actionAngleStaeckelGrid(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation, grid-based interpolation"""
    def __init__(self,pot=None,delta=None,Rmax=5.,
                 nE=25,npsi=25,nLz=30,numcores=1,
                 interpecc=False,
                 **kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleStaeckelGrid object
        INPUT:
           pot= potential or list of potentials

           delta= focus of prolate confocal coordinate system (can be Quantity)

           Rmax = Rmax for building grids (natural units)

           nE=, npsi=, nLz= grid size

           interpecc= (False) if True, also interpolate the approximate eccentricity, zmax, rperi, and rapo

           numcores= number of cpus to use to parallellize

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
         
           instance

        HISTORY:

            2012-11-29 - Written - Bovy (IAS)

            2017-12-15 - Written - Bovy (UofT)

        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if pot is None:
            raise IOError("Must specify pot= for actionAngleStaeckelGrid")
        self._pot= flatten_potential(pot)
        if delta is None:
            raise IOError("Must specify delta= for actionAngleStaeckelGrid")
        if ext_loaded and 'c' in kwargs and kwargs['c']:
            self._c= True
        else:
            self._c= False
        self._delta= delta
        if _APY_LOADED and isinstance(self._delta,units.Quantity):
            self._delta= self._delta.to(units.kpc).value/self._ro
        self._Rmax= Rmax
        self._Rmin= 0.01
        #Set up the actionAngleStaeckel object that we will use to interpolate
        self._aA= actionAngleStaeckel.actionAngleStaeckel(pot=self._pot,delta=self._delta,c=self._c)
        #Build grid
        self._Lzmin= 0.01
        self._Lzs= numpy.linspace(self._Lzmin,
                                  self._Rmax\
                                      *galpy.potential.vcirc(self._pot,
                                                             self._Rmax),
                                  nLz)
        self._Lzmax= self._Lzs[-1]
        self._nLz= nLz
        #Calculate E_c(R=RL), energy of circular orbit
        self._RL= numpy.array([galpy.potential.rl(self._pot,l) for l in self._Lzs])
        self._RLInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                 self._RL,k=3)
        self._ERL= _evaluatePotentials(self._pot,self._RL,
                                       numpy.zeros(self._nLz))\
                                       +self._Lzs**2./2./self._RL**2.
        self._ERLmax= numpy.amax(self._ERL)+1.
        self._ERLInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                  numpy.log(-(self._ERL-self._ERLmax)),k=3)
        self._Ramax= 200./8.
        self._ERa= _evaluatePotentials(self._pot,self._Ramax,0.) +self._Lzs**2./2./self._Ramax**2.
        #self._EEsc= numpy.array([self._ERL[ii]+galpy.potential.vesc(self._pot,self._RL[ii])**2./4. for ii in range(nLz)])
        self._ERamax= numpy.amax(self._ERa)+1.
        self._ERaInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                  numpy.log(-(self._ERa-self._ERamax)),k=3)
        y= numpy.linspace(0.,1.,nE)
        self._nE= nE
        psis= numpy.linspace(0.,1.,npsi)*numpy.pi/2.
        self._npsi= npsi
        jr= numpy.zeros((nLz,nE,npsi))
        jz= numpy.zeros((nLz,nE,npsi))
        u0= numpy.zeros((nLz,nE))
        jrLzE= numpy.zeros((nLz))
        jzLzE= numpy.zeros((nLz))
        #First calculate u0
        thisLzs= (numpy.tile(self._Lzs,(nE,1)).T).flatten()
        thisERL= (numpy.tile(self._ERL,(nE,1)).T).flatten()
        thisERa= (numpy.tile(self._ERa,(nE,1)).T).flatten()
        thisy= (numpy.tile(y,(nLz,1))).flatten()
        thisE= _invEfunc(_Efunc(thisERa,thisERL)+thisy*(_Efunc(thisERL,thisERL)-_Efunc(thisERa,thisERL)),thisERL)
        if isinstance(self._pot,galpy.potential.interpRZPotential) and hasattr(self._pot,'_origPot'):
            u0pot= self._pot._origPot
        else:
            u0pot= self._pot
        if self._c:
            mu0= actionAngleStaeckel_c.actionAngleStaeckel_calcu0(thisE,thisLzs,
                                                                  u0pot,
                                                                  self._delta)[0]
        else:
            if numcores > 1:
                mu0= multi.parallel_map((lambda x: self.calcu0(thisE[x],
                                                               thisLzs[x])),
                                        range(nE*nLz),
                                        numcores=numcores)
            else:
                mu0= list(map((lambda x: self.calcu0(thisE[x],
                                                     thisLzs[x])),
                              range(nE*nLz)))
        u0= numpy.reshape(mu0,(nLz,nE))
        thisR= self._delta*numpy.sinh(u0)
        thisv= numpy.reshape(self.vatu0(thisE.flatten(),thisLzs.flatten(),
                                        u0.flatten(),
                                        thisR.flatten()),(nLz,nE))
        self.thisv= thisv
        #reshape
        thisLzs= numpy.reshape(thisLzs,(nLz,nE))
        thispsi= numpy.tile(psis,(nLz,nE,1)).flatten()
        thisLzs= numpy.tile(thisLzs.T,(npsi,1,1)).T.flatten()
        thisR= numpy.tile(thisR.T,(npsi,1,1)).T.flatten()
        thisv= numpy.tile(thisv.T,(npsi,1,1)).T.flatten()
        mjr, mlz, mjz= self._aA(thisR, #R
                                thisv*numpy.cos(thispsi), #vR
                                thisLzs/thisR, #vT
                                numpy.zeros(len(thisR)), #z
                                thisv*numpy.sin(thispsi), #vz
                                fixed_quad=True) 
        if interpecc:
            mecc, mzmax, mrperi, mrap=\
                self._aA.EccZmaxRperiRap(thisR, #R
                                         thisv*numpy.cos(thispsi), #vR
                                         thisLzs/thisR, #vT
                                         numpy.zeros(len(thisR)), #z
                                         thisv*numpy.sin(thispsi)) #vz
        if isinstance(self._pot,galpy.potential.interpRZPotential) and hasattr(self._pot,'_origPot'):
            #Interpolated potentials have problems with extreme orbits
            indx= (mjr == 9999.99)
            indx+= (mjz == 9999.99)
            #Re-calculate these using the original potential, hopefully not too slow
            tmpaA= actionAngleStaeckel.actionAngleStaeckel(pot=self._pot._origPot,delta=self._delta,c=self._c)
            mjr[indx], dum, mjz[indx]= tmpaA(thisR[indx], #R
                                             thisv[indx]*numpy.cos(thispsi[indx]), #vR
                                             thisLzs[indx]/thisR[indx], #vT
                                             numpy.zeros(numpy.sum(indx)), #z
                                             thisv[indx]*numpy.sin(thispsi[indx]), #vz
                                             fixed_quad=True)
            if interpecc:
                mecc[indx], mzmax[indx], mrperi[indx], mrap[indx]=\
                    self._aA.EccZmaxRperiRap(thisR[indx], #R
                                             thisv[indx]*numpy.cos(thispsi[indx]), #vR
                                             thisLzs[indx]/thisR[indx], #vT
                                             numpy.zeros(numpy.sum(indx)), #z
                                             thisv[indx]*numpy.sin(thispsi[indx])) #vz
        jr= numpy.reshape(mjr,(nLz,nE,npsi))
        jz= numpy.reshape(mjz,(nLz,nE,npsi))
        if interpecc:
            ecc= numpy.reshape(mecc,(nLz,nE,npsi))
            zmax= numpy.reshape(mzmax,(nLz,nE,npsi))
            rperi= numpy.reshape(mrperi,(nLz,nE,npsi))
            rap= numpy.reshape(mrap,(nLz,nE,npsi))
            zmaxLzE= numpy.zeros((nLz))
            rperiLzE= numpy.zeros((nLz))
            rapLzE= numpy.zeros((nLz))
        for ii in range(nLz):
            jrLzE[ii]= numpy.nanmax(jr[ii,(jr[ii,:,:] != 9999.99)])#:,:])
            jzLzE[ii]= numpy.nanmax(jz[ii,(jz[ii,:,:] != 9999.99)])#:,:])
            if interpecc:
                zmaxLzE[ii]= numpy.amax(zmax[ii,numpy.isfinite(zmax[ii])])
                rperiLzE[ii]= numpy.amax(rperi[ii,numpy.isfinite(rperi[ii])])
                rapLzE[ii]= numpy.amax(rap[ii,numpy.isfinite(rap[ii])])
        jrLzE[(jrLzE == 0.)]= numpy.nanmin(jrLzE[(jrLzE > 0.)])
        jzLzE[(jzLzE == 0.)]= numpy.nanmin(jzLzE[(jzLzE > 0.)])
        if interpecc:
            zmaxLzE[(zmaxLzE == 0.)]= numpy.nanmin(zmaxLzE[(zmaxLzE > 0.)])
            rperiLzE[(rperiLzE == 0.)]= numpy.nanmin(rperiLzE[(rperiLzE > 0.)])
            rapLzE[(rapLzE == 0.)]= numpy.nanmin(rapLzE[(rapLzE > 0.)])
        for ii in range(nLz):
            jr[ii,:,:]/= jrLzE[ii]
            jz[ii,:,:]/= jzLzE[ii]
            if interpecc:
                zmax[ii,:,:]/= zmaxLzE[ii]
                rperi[ii,:,:]/= rperiLzE[ii]
                rap[ii,:,:]/= rapLzE[ii]
        #Deal w/ 9999.99
        jr[(jr > 1.)]= 1.
        jz[(jz > 1.)]= 1.
        #Deal w/ NaN
        jr[numpy.isnan(jr)]= 0.
        jz[numpy.isnan(jz)]= 0.
        if interpecc:
            ecc[(ecc < 0.)]= 0.
            ecc[(ecc > 1.)]= 1.
            ecc[numpy.isnan(ecc)]= 0.
            ecc[numpy.isinf(ecc)]= 1.
            zmax[(zmax > 1.)]= 1.
            zmax[numpy.isnan(zmax)]= 0.
            zmax[numpy.isinf(zmax)]= 1.
            rperi[(rperi > 1.)]= 1.
            rperi[numpy.isnan(rperi)]= 0.
            rperi[numpy.isinf(rperi)]= 0. # typically orbits that can reach 0
            rap[(rap > 1.)]= 1.
            rap[numpy.isnan(rap)]= 0.
            rap[numpy.isinf(rap)]= 1.
        #First interpolate the maxima
        self._jr= jr
        self._jz= jz
        self._u0= u0
        self._jrLzE= jrLzE
        self._jzLzE= jzLzE
        self._jrLzInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                   numpy.log(jrLzE+10.**-5.),k=3)
        self._jzLzInterp= interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                                   numpy.log(jzLzE+10.**-5.),k=3)
        if interpecc:
            self._ecc= ecc
            self._zmax= zmax
            self._rperi= rperi
            self._rap= rap
            self._zmaxLzE= zmaxLzE
            self._rperiLzE= rperiLzE
            self._rapLzE= rapLzE
            self._zmaxLzInterp=\
                interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                         numpy.log(zmaxLzE+10.**-5.),k=3)
            self._rperiLzInterp=\
                interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                         numpy.log(rperiLzE+10.**-5.),k=3)
            self._rapLzInterp=\
                interpolate.InterpolatedUnivariateSpline(self._Lzs,
                                                         numpy.log(rapLzE+10.**-5.),k=3)
        #Interpolate u0
        self._logu0Interp= interpolate.RectBivariateSpline(self._Lzs,
                                                           y,
                                                           numpy.log(u0),
                                                           kx=3,ky=3,s=0.)
        #spline filter jr and jz, such that they can be used with ndimage.map_coordinates
        self._jrFiltered= ndimage.spline_filter(numpy.log(self._jr+10.**-10.),order=3)
        self._jzFiltered= ndimage.spline_filter(numpy.log(self._jz+10.**-10.),order=3)
        if interpecc:
            self._eccFiltered= ndimage.spline_filter(numpy.log(self._ecc+10.**-10.),order=3)
            self._zmaxFiltered= ndimage.spline_filter(numpy.log(self._zmax+10.**-10.),order=3)
            self._rperiFiltered= ndimage.spline_filter(numpy.log(self._rperi+10.**-10.),order=3)
            self._rapFiltered= ndimage.spline_filter(numpy.log(self._rap+10.**-10.),order=3)
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
           Keywords for actionAngleStaeckel.__call__ for off-the-grid evaluations
        OUTPUT:
           (jr,lz,jz)
        HISTORY:
           2012-11-29 - Written - Bovy (IAS)
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
        Lz= R*vT
        Phi= _evaluatePotentials(self._pot,R,z)
        E= Phi+vR**2./2.+vT**2./2.+vz**2./2.
        thisERL= -numpy.exp(self._ERLInterp(Lz))+self._ERLmax
        thisERa= -numpy.exp(self._ERaInterp(Lz))+self._ERamax
        if isinstance(R,numpy.ndarray):
            indx= ((E-thisERa)/(thisERL-thisERa) > 1.)\
                *(((E-thisERa)/(thisERL-thisERa)-1.) < 10.**-2.)
            E[indx]= thisERL[indx]
            indx= ((E-thisERa)/(thisERL-thisERa) < 0.)\
                *((E-thisERa)/(thisERL-thisERa) > -10.**-2.)
            E[indx]= thisERa[indx]
            indx= (Lz < self._Lzmin)
            indx+= (Lz > self._Lzmax)
            indx+= ((E-thisERa)/(thisERL-thisERa) > 1.)
            indx+= ((E-thisERa)/(thisERL-thisERa) < 0.)
            indxc= True^indx
            jr= numpy.empty(R.shape)
            jz= numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                u0= numpy.exp(self._logu0Interp.ev(Lz[indxc],
                                                   (_Efunc(E[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))/(_Efunc(thisERL[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))))
                sinh2u0= numpy.sinh(u0)**2.
                thisEr= self.Er(R[indxc],z[indxc],vR[indxc],vz[indxc],
                                E[indxc],Lz[indxc],sinh2u0,u0)
                thisEz= self.Ez(R[indxc],z[indxc],vR[indxc],vz[indxc],
                                E[indxc],Lz[indxc],sinh2u0,u0)
                thisv2= self.vatu0(E[indxc],Lz[indxc],u0,self._delta*numpy.sinh(u0),retv2=True)
                cos2psi= 2.*thisEr/thisv2/(1.+sinh2u0) #latter is cosh2u0
                cos2psi[(cos2psi > 1.)*(cos2psi < 1.+10.**-5.)]= 1.
                indxCos2psi= (cos2psi > 1.)
                indxCos2psi+= (cos2psi < 0.)
                indxc[indxc]= True^indxCos2psi#Handle these two cases as off-grid
                indx= True^indxc
                psi= numpy.arccos(numpy.sqrt(cos2psi[True^indxCos2psi]))
                coords= numpy.empty((3,numpy.sum(indxc)))
                coords[0,:]= (Lz[indxc]-self._Lzmin)/(self._Lzmax-self._Lzmin)*(self._nLz-1.)
                y= (_Efunc(E[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))/(_Efunc(thisERL[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))
                coords[1,:]= y*(self._nE-1.)
                coords[2,:]= psi/numpy.pi*2.*(self._npsi-1.)
                jr[indxc]= (numpy.exp(ndimage.interpolation.map_coordinates(self._jrFiltered,
                                                                            coords,
                                                                            order=3,
                                                                            prefilter=False))-10.**-10.)*(numpy.exp(self._jrLzInterp(Lz[indxc]))-10.**-5.)
                #Switch to Ez-calculated psi
                sin2psi= 2.*thisEz[True^indxCos2psi]/thisv2[True^indxCos2psi]/(1.+sinh2u0[True^indxCos2psi]) #latter is cosh2u0
                sin2psi[(sin2psi > 1.)*(sin2psi < 1.+10.**-5.)]= 1.
                indxSin2psi= (sin2psi > 1.)
                indxSin2psi+= (sin2psi < 0.)
                indxc[indxc]= True^indxSin2psi#Handle these two cases as off-grid
                indx= True^indxc
                psiz= numpy.arcsin(numpy.sqrt(sin2psi[True^indxSin2psi]))
                newcoords= numpy.empty((3,numpy.sum(indxc)))
                newcoords[0:2,:]= coords[0:2,True^indxSin2psi]
                newcoords[2,:]= psiz/numpy.pi*2.*(self._npsi-1.)
                jz[indxc]= (numpy.exp(ndimage.interpolation.map_coordinates(self._jzFiltered,
                                                                           newcoords,
                                                                           order=3,
                                                                           prefilter=False))-10.**-10.)*(numpy.exp(self._jzLzInterp(Lz[indxc]))-10.**-5.)
            if numpy.sum(indx) > 0:
                jrindiv, lzindiv, jzindiv= self._aA(R[indx],
                                                    vR[indx],
                                                    vT[indx],
                                                    z[indx],
                                                    vz[indx],
                                                    **kwargs)
                jr[indx]= jrindiv
                jz[indx]= jzindiv
                """
                jrindiv= numpy.empty(numpy.sum(indx))
                jzindiv= numpy.empty(numpy.sum(indx))
                for ii in range(numpy.sum(indx)):
                    try:
                        thisaA= actionAngleStaeckel.actionAngleStaeckelSingle(\
                            R[indx][ii], #R
                            vR[indx][ii], #vR
                            vT[indx][ii], #vT
                            z[indx][ii], #z
                            vz[indx][ii], #vz
                            pot=self._pot,delta=self._delta)
                        jrindiv[ii]= thisaA.JR(fixed_quad=True)[0]
                        jzindiv[ii]= thisaA.Jz(fixed_quad=True)[0]
                    except (UnboundError,OverflowError):
                        jrindiv[ii]= numpy.nan
                        jzindiv[ii]= numpy.nan
                jr[indx]= jrindiv
                jz[indx]= jzindiv
                """
        else:
            jr,Lz, jz= self(numpy.array([R]),
                            numpy.array([vR]),
                            numpy.array([vT]),
                            numpy.array([z]),
                            numpy.array([vz]),
                            **kwargs)
            return (jr[0],Lz[0],jz[0])
        jr[jr < 0.]= 0.
        jz[jz < 0.]= 0.
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
        return self(*args,**kwargs)[2]

    def JR(self,*args,**kwargs):
        """
        NAME:
           JR
        PURPOSE:
           evaluate the action jr
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           jr
        HISTORY:
           2012-07-30 - Written - Bovy (IAS@MPIA)
        """
        return self(*args,**kwargs)[0]

    def _EccZmaxRperiRap(self,*args,**kwargs):
        """
        NAME:
           EccZmaxRperiRap (_EccZmaxRperiRap)
        PURPOSE:
           evaluate the eccentricity, maximum height above the plane, peri- and apocenter in the Staeckel approximation
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
           (e,zmax,rperi,rap)
        HISTORY:
           2017-12-15 - Written - Bovy (UofT)
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
        Lz= R*vT
        Phi= _evaluatePotentials(self._pot,R,z)
        E= Phi+vR**2./2.+vT**2./2.+vz**2./2.
        thisERL= -numpy.exp(self._ERLInterp(Lz))+self._ERLmax
        thisERa= -numpy.exp(self._ERaInterp(Lz))+self._ERamax
        if isinstance(R,numpy.ndarray):
            indx= ((E-thisERa)/(thisERL-thisERa) > 1.)\
                *(((E-thisERa)/(thisERL-thisERa)-1.) < 10.**-2.)
            E[indx]= thisERL[indx]
            indx= ((E-thisERa)/(thisERL-thisERa) < 0.)\
                *((E-thisERa)/(thisERL-thisERa) > -10.**-2.)
            E[indx]= thisERa[indx]
            indx= (Lz < self._Lzmin)
            indx+= (Lz > self._Lzmax)
            indx+= ((E-thisERa)/(thisERL-thisERa) > 1.)
            indx+= ((E-thisERa)/(thisERL-thisERa) < 0.)
            indxc= True^indx
            ecc= numpy.empty(R.shape)
            zmax= numpy.empty(R.shape)
            rperi= numpy.empty(R.shape)
            rap= numpy.empty(R.shape)
            if numpy.sum(indxc) > 0:
                u0= numpy.exp(self._logu0Interp.ev(Lz[indxc],
                                                   (_Efunc(E[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))/(_Efunc(thisERL[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))))
                sinh2u0= numpy.sinh(u0)**2.
                thisEr= self.Er(R[indxc],z[indxc],vR[indxc],vz[indxc],
                                E[indxc],Lz[indxc],sinh2u0,u0)
                thisEz= self.Ez(R[indxc],z[indxc],vR[indxc],vz[indxc],
                                E[indxc],Lz[indxc],sinh2u0,u0)
                thisv2= self.vatu0(E[indxc],Lz[indxc],u0,self._delta*numpy.sinh(u0),retv2=True)
                cos2psi= 2.*thisEr/thisv2/(1.+sinh2u0) #latter is cosh2u0
                cos2psi[(cos2psi > 1.)*(cos2psi < 1.+10.**-5.)]= 1.
                indxCos2psi= (cos2psi > 1.)
                indxCos2psi+= (cos2psi < 0.)
                indxc[indxc]= True^indxCos2psi#Handle these two cases as off-grid
                indx= True^indxc
                psi= numpy.arccos(numpy.sqrt(cos2psi[True^indxCos2psi]))
                coords= numpy.empty((3,numpy.sum(indxc)))
                coords[0,:]= (Lz[indxc]-self._Lzmin)/(self._Lzmax-self._Lzmin)*(self._nLz-1.)
                y= (_Efunc(E[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))/(_Efunc(thisERL[indxc],thisERL[indxc])-_Efunc(thisERa[indxc],thisERL[indxc]))
                coords[1,:]= y*(self._nE-1.)
                coords[2,:]= psi/numpy.pi*2.*(self._npsi-1.)
                ecc[indxc]= (numpy.exp(ndimage.interpolation.map_coordinates(self._eccFiltered,
                                                                            coords,
                                                                            order=3,
                                                                            prefilter=False))-10.**-10.)
                rperi[indxc]= (numpy.exp(ndimage.interpolation.map_coordinates(self._rperiFiltered,
                                                                            coords,
                                                                            order=3,
                                                                            prefilter=False))-10.**-10.)*(numpy.exp(self._rperiLzInterp(Lz[indxc]))-10.**-5.)
                # We do rap below with zmax
                #Switch to Ez-calculated psi
                sin2psi= 2.*thisEz[True^indxCos2psi]/thisv2[True^indxCos2psi]/(1.+sinh2u0[True^indxCos2psi]) #latter is cosh2u0
                sin2psi[(sin2psi > 1.)*(sin2psi < 1.+10.**-5.)]= 1.
                indxSin2psi= (sin2psi > 1.)
                indxSin2psi+= (sin2psi < 0.)
                indxc[indxc]= True^indxSin2psi#Handle these two cases as off-grid
                indx= True^indxc
                psiz= numpy.arcsin(numpy.sqrt(sin2psi[True^indxSin2psi]))
                newcoords= numpy.empty((3,numpy.sum(indxc)))
                newcoords[0:2,:]= coords[0:2,True^indxSin2psi]
                newcoords[2,:]= psiz/numpy.pi*2.*(self._npsi-1.)
                zmax[indxc]= (numpy.exp(ndimage.interpolation.map_coordinates(self._zmaxFiltered,
                                                                           newcoords,
                                                                           order=3,
                                                                           prefilter=False))-10.**-10.)*(numpy.exp(self._zmaxLzInterp(Lz[indxc]))-10.**-5.)
                rap[indxc]= (numpy.exp(ndimage.interpolation.map_coordinates(self._rapFiltered,
                                                                           newcoords,
                                                                           order=3,
                                                                           prefilter=False))-10.**-10.)*(numpy.exp(self._rapLzInterp(Lz[indxc]))-10.**-5.)
            if numpy.sum(indx) > 0:
                eccindiv, zmaxindiv, rperiindiv, rapindiv=\
                    self._aA.EccZmaxRperiRap(R[indx],
                                             vR[indx],
                                             vT[indx],
                                             z[indx],
                                             vz[indx],
                                             **kwargs)
                ecc[indx]= eccindiv
                zmax[indx]= zmaxindiv
                rperi[indx]= rperiindiv
                rap[indx]= rapindiv
        else:
            ecc,zmax,rperi,rap= self.EccZmaxRperiRap(numpy.array([R]),
                                                     numpy.array([vR]),
                                                     numpy.array([vT]),
                                                     numpy.array([z]),
                                                     numpy.array([vz]),
                                                     **kwargs)
            return (ecc[0],zmax[0],rperi[0],rap[0])
        ecc[ecc < 0.]= 0.
        zmax[zmax < 0.]= 0.
        rperi[rperi < 0.]= 0.
        rap[rap < 0.]= 0.
        return (ecc,zmax,rperi,rap)

    def vatu0(self,E,Lz,u0,R,retv2=False):
        """
        NAME:
           vatu0
        PURPOSE:
           calculate the velocity at u0
        INPUT:
           E - energy
           Lz - angular momentum
           u0 - u0
           R - radius corresponding to u0,pi/2.
           retv2= (False), if True return v^2
        OUTPUT:
           velocity
        HISTORY:
           2012-11-29 - Written - Bovy (IAS)
        """                        
        v2= (2.*(E-actionAngleStaeckel.potentialStaeckel(u0,numpy.pi/2.,
                                                         self._pot,
                                                         self._delta))
             -Lz**2./R**2.)
        if retv2: return v2
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

    def Er(self,R,z,vR,vz,E,Lz,sinh2u0,u0):
        """
        NAME:
           Er
        PURPOSE:
           calculate the 'radial energy'
        INPUT:
           R, z, vR, vz - coordinates
           E - energy
           Lz - angular momentum
           sinh2u0, u0 - sinh^2 and u0
        OUTPUT:
           Er
        HISTORY:
           2012-11-29 - Written - Bovy (IAS)
        """                           
        u,v= bovy_coords.Rz_to_uv(R,z,self._delta)
        pu= (vR*numpy.cosh(u)*numpy.sin(v)
             +vz*numpy.sinh(u)*numpy.cos(v)) #no delta, bc we will divide it out
        out= (pu**2./2.+Lz**2./2./self._delta**2.*(1./numpy.sinh(u)**2.-1./sinh2u0)
              -E*(numpy.sinh(u)**2.-sinh2u0)
              +(numpy.sinh(u)**2.+1.)*actionAngleStaeckel.potentialStaeckel(u,numpy.pi/2.,self._pot,self._delta)
              -(sinh2u0+1.)*actionAngleStaeckel.potentialStaeckel(u0,numpy.pi/2.,self._pot,self._delta))
#              +(numpy.sinh(u)**2.+numpy.sin(v)**2.)*actionAngleStaeckel.potentialStaeckel(u,v,self._pot,self._delta)
#              -(sinh2u0+numpy.sin(v)**2.)*actionAngleStaeckel.potentialStaeckel(u0,v,self._pot,self._delta))
        return out

    def Ez(self,R,z,vR,vz,E,Lz,sinh2u0,u0):
        """
        NAME:
           Ez
        PURPOSE:
           calculate the 'vertical energy'
        INPUT:
           R, z, vR, vz - coordinates
           E - energy
           Lz - angular momentum
           sinh2u0, u0 - sinh^2 and u0
        OUTPUT:
           Ez
        HISTORY:
           2012-12-23 - Written - Bovy (IAS)
        """                           
        u,v= bovy_coords.Rz_to_uv(R,z,self._delta)
        pv= (vR*numpy.sinh(u)*numpy.cos(v)
             -vz*numpy.cosh(u)*numpy.sin(v)) #no delta, bc we will divide it out
        out= (pv**2./2.+Lz**2./2./self._delta**2.*(1./numpy.sin(v)**2.-1.)
              -E*(numpy.sin(v)**2.-1.)
              -(sinh2u0+1.)*actionAngleStaeckel.potentialStaeckel(u0,numpy.pi/2.,self._pot,self._delta)
              +(sinh2u0+numpy.sin(v)**2.)*actionAngleStaeckel.potentialStaeckel(u0,v,self._pot,self._delta))
        return out


def _u0Eq(logu,delta,pot,E,Lz22):
    """The equation that needs to be minimized to find u0"""
    u= numpy.exp(logu)
    sinh2u= numpy.sinh(u)**2.
    cosh2u= numpy.cosh(u)**2.
    dU= cosh2u*actionAngleStaeckel.potentialStaeckel(u,numpy.pi/2.,pot,delta)
    return -(E*sinh2u-dU-Lz22/delta**2./sinh2u)

def _Efunc(E,*args):
    """Function to apply to the energy in building the grid (e.g., if this is a log, then the grid will be logarithmic"""
#    return ((E-args[0]))**0.5
    return numpy.log((E-args[0]+10.**-10.))
def _invEfunc(Ef,*args):
    """Inverse of Efunc"""
#    return Ef**2.+args[0]
    return numpy.exp(Ef)+args[0]-10.**-10.
