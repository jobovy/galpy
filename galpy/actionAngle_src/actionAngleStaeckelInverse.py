###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleStaeckelInverse
#
#             Calculate (x,v) coordinates for any Staeckel potential from 
#             given action-angle coordinates
#
###############################################################################
import numpy
from scipy import optimize
from scipy.spatial import cKDTree
from galpy.potential import IsochronePotential, vcirc, dvcircdR, \
    evaluatePotentials, evaluateRforces, toPlanarPotential
from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneInverse
from galpy.actionAngle import actionAngleSpherical
from galpy.actionAngle_src.actionAngleAxi import actionAngleAxi
from galpy.actionAngle_src.actionAngleInverse import actionAngleInverse
from galpy.actionAngle_src.actionAngleIsochrone import _actionAngleIsochroneHelper
from galpy.util import bovy_coords
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class actionAngleStaeckelInverse(actionAngleInverse):
    """Inverse action-angle formalism for Staeckel potentials"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleStaeckelInverse object

        INPUT:

           pot= a Staeckel potential

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2017-12-04 - Started - Bovy (UofT)

        """
        actionAngleInverse.__init__(self,*args,**kwargs)
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleStaeckelInverse")
        self._pot= kwargs['pot']
        #The following for if we ever implement this code in C
        self._c= False
        ext_loaded= False
        if ext_loaded and (('c' in kwargs and kwargs['c'])
                           or not 'c' in kwargs):
            self._c= True #pragma: no cover
        else:
            self._c= False
        # Check the units
        self._check_consistent_units()
        return None
    
    def _evaluate(self,jr,jphi,jz,angler,anglephi,anglez,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])

           tol= (object-wide value) goal for |dJ|/|J| along the torus

        OUTPUT:

           [R,vR,vT,z,vz,phi]

        HISTORY:

           2017-12-04 - Written - Bovy (UofT)

        """
        return self._xvFreqs(jr,jphi,jz,angler,anglephi,anglez,**kwargs)[:6]
        
    def _xvFreqs(self,jr,jphi,jz,angler,anglephi,anglez,**kwargs):
        """
        NAME:

           xvFreqs

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequencies

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])


        OUTPUT:

           ([R,vR,vT,z,vz,phi],OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2017-12-04 - Written - Bovy (UofT)

        """
        raise NotImplementedError("Method not implemented yet")

    def _Freqs(self,jr,jphi,jz,**kwargs):
        """
        NAME:

           Freqs

        PURPOSE:

           return the frequencies corresponding to a torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

        OUTPUT:

           (OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2017-12-04 - Written - Bovy (UofT)

        """
        raise NotImplementedError("Method not implemented yet")

class actionAngleStaeckelInverseSingle(actionAngleInverse):
    """Invert the action-angle formalism for a Staeckel potential for a single torus"""
    def __init__(self,E,Lz,I3,delta,
                 jr=None,jphi=None,jz=None,
                 Omegar=None,Omegaphi=None,Omegaz=None,
                 umin=None,umax=None,vmin=None,
                 pot=None,ntr='auto',ntz='auto',max_ntrz=512):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleSphericalInverseSingle object

        INPUT:

           E- energy

           Lz- z-component of the angular momentum

           I3 - third integral
           
           delta - delta defining the spheroidal coordinate system

           pot= Staeckel potential instance or list of such instances

           Optionally, also supply the following:
           
               jr,jphi,jz= actions

               Omegar, Omegaphi, Omegaz= frequencies

               umin, umax, vmin= 'peri' and 'apo'center radii in u and v

           ntr= ('auto')

           ntz= ('auto')

           max_ntrz= (512) Maximum number of radial and vertical angles to use to determine the torus mapping when using ntr='auto'

        OUTPUT:

           instance

        HISTORY:

           2017-11-21 - Written - Bovy (UofT)

        """
        if pot is None: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleSphericalInverse")
        self._pot= pot
        self._E= E
        self._Lz= Lz
        self._Lz2= self._Lz**2.
        self._I3= I3
        self._delta= delta
        delta2= self._delta**2.
        # Calculate jr if not given
        if jr is None or umin is None or umax is None or vmin is None \
                or Omegar is None or Omegaphi is None or Omegaz is None:
            #NEEDS EDITING

            # Just setup an actionAngleAxi instance for all of this
            # Setup the orbit at R_L s.t. R_L vc(R_L) = L
            rl= optimize.newton(lambda x: x*vcirc(self._pot,x)-L,1.,
                                lambda x: dvcircdR(self._pot,x)+x)
            aAS= actionAngleAxi(rl,
                                numpy.sqrt(2.*(E
                                               -evaluatePotentials(self._pot,
                                                                   rl,0.))
                                           -L2/rl**2.),
                                L/rl,pot=toPlanarPotential(self._pot))
            self._jr= aAS.JR()
            # Assume that the orbit is in the plane unless otherwise specified
            if jphi is None and jz is None:
                self._jphi= L
                self._jz= 0.
            else:
                # Should probably check that jphi and jz are consistent with L
                self._jphi= jphi
                self._jz= jz
            self._rperi,self._rap= aAS.calcRapRperi()
            # Also compute the frequencies, use internal actionAngleSpherical
            aAS= actionAngleSpherical(pot=self._pot)
            Rmean= numpy.exp((numpy.log(self._rperi)+numpy.log(self._rap))/2.)
            self._Omegar= aAS._calc_or(Rmean,self._rperi,self._rap,
                                       self._E,self._L,False)
            self._Omegaz= aAS._calc_op(self._Omegar,Rmean,
                                       self._rperi,self._rap,
                                       self._E,self._L,False)
        else:
            # Store everything
            self._jr= jr
            self._jphi= jphi
            self._jz= jz
            self._Omegar= Omegar
            self._Omegaphi= Omegaphi
            self._Omegaz= Omegaz
            self._umin= umin
            self._umax= umax
            self._vmin= vmin
        self._OmegazoverOmegar= self._Omegaz/self._Omegar
        # First need to determine an appropriate IsochronePotential
        ampb= self._Lz2*self._Omegaphi*(self._Omegar-self._Omegaphi)\
            /(2.*self._Omegaphi-self._Omegar)**2.
        if ampb < 0.:
            raise NotImplementedError('actionAngleStaeckelInverse not implemented for the case where Omegaz > Omegar or Omegaz < Omegar/2')
        amp= numpy.sqrt(self._Omegar
                        *(self._jr+numpy.sqrt(self._Lz2+4.*ampb)
                          *self._Omegaphi/self._Omegar)**3.)
        self._ip= IsochronePotential(amp=amp,b=ampb/amp)
        self._isoaa= actionAngleIsochrone(ip=self._ip)
        self._isoaa_helper= _actionAngleIsochroneHelper(ip=self._ip)
        self._isoaainv= actionAngleIsochroneInverse(ip=self._ip)
        # Now need to determine the Sn and dSn mapping       
        if ntr == 'auto': ntr= 128 # BOVY: IMPLEMENT CORRECTLY
        thetara= numpy.linspace(0.,2.*numpy.pi*(1.-1./ntr),ntr)
        self._thetar_offset= (thetara[1]-thetara[0])*1e-4
        thetara+= self._thetar_offset
        thetaza= numpy.linspace(0.,2.*numpy.pi*(1.-1./ntz),ntz)
        self._thetaz_offset= (thetaza[1]-thetaza[0])*1e-4
        thetaza+= self._thetaz_offset
        self._thetara= thetara
        self._thetaza= thetaza
        self._ugrid, self._vgrid= self._create_uvgrid(ntr,ntz,thetara,thetaza)
        self._jra, self._jza=\
            _jrz(numpy.fabs(self._ugrid),numpy.fabs(self._vgrid),
                 self._E,self._Lz2,self._I3,self._pot,self._delta,
                 self._isoaa_helper,
                 sgnu=numpy.sign(self._ugrid),
                 sgnv=numpy.sign(self._vgrid))
        return None
        # Compute Sn and dSn/dJr, remove n=0
        self._nforSn= numpy.arange(len(self._ora)//2+1)
        #self._nSn= numpy.real(numpy.fft.rfft(self._jra-self._jr))[1:]/len(self._jra)
        #self._dSndJr= (numpy.real(numpy.fft.rfft(self._Omegar/self._ora-1.))/self._nforSn)[1:]/len(self._ora)
        #self._dSndLish= (numpy.real(numpy.fft.rfft(self._dEdL))/self._nforSn)[1:]/len(self._ora)
        #self._nforSn= self._nforSn[1:]
        return None

    def __call__(self,angler,anglephi,anglez,jphi=None):
        """
        NAME:
           __call__
        PURPOSE:
           convert angles --> (x,v) for this torus
        INPUT:
           angler, anglephi, anglez - angles on the torus
           jphi= (object-wide default) z-component of the angular momentum
        OUTPUT:
           (R,vR,vT,z,vz,phi)
        HISTORY:
           2017-11-21 - Written - Bovy (UofT)
        """
        # First we need to solve for anglera
        anglera= optimize.newton(\
            lambda ar: ar+2.*numpy.sum(self._dSndJr*numpy.sin(self._nforSn*ar))-angler,
            0.,
            lambda ar: 1.+2.*numpy.sum(self._nforSn*self._dSndJr
                                       *numpy.cos(self._nforSn*ar)))
        # Then compute the auxiliary action
        jra= self._jr+2.*numpy.sum(self._nSn*numpy.cos(self._nforSn*anglera))
        angleza= anglez+self._OmegazoverOmegar*(anglera-angler)\
            -2.*numpy.sum(self._dSndLish*numpy.sin(self._nforSn*anglera))
        if jphi is None: 
            jphi= self._jphi
            jz= self._jz
        else:
            jz= self._L-numpy.fabs(jphi)
        anglephia= anglephi+numpy.sign(jphi)*(angleza-anglez)
        return self._isoaainv(jra,jphi,jz,anglera,anglephia,angleza)

    def _create_uvgrid(self,ntr,ntz,thetara,thetaza):
        # Find (u,v) for a regular grid in (thetara,thetaza)
        # To efficiently start the search, we compute (thetara,thetaza) for
        # a dense grid in (u,v)
        eps= 1e-6
        us= numpy.linspace(self._umin+eps,self._umax-eps,ntr)
        vs= numpy.linspace(self._vmin+eps,numpy.pi-self._vmin-eps,ntz)
        maxdu= 1.*(us[1]-us[0]) # For later changes
        maxdv= 1.*(vs[1]-vs[0])
        us= numpy.tile(us,(ntz,1)).T
        vs= numpy.tile(vs,(ntr,1))
        uvtra= numpy.empty((2*ntr,2*ntz))
        uvtza= numpy.empty((2*ntr,2*ntz))
        ut= numpy.empty((2*ntr,2*ntz))
        vt= numpy.empty((2*ntr,2*ntz))
        # Do each quadrant separately
        for ii,sgnu in enumerate([1.,-1.]):
            for jj,sgnv in enumerate([1.,-1.]):
                tuvtra, tuvtza= _anglerz(us,vs,
                                         self._E,self._Lz2,self._I3,self._pot,
                                         self._delta,self._isoaa_helper,
                                         sgnu=sgnu,sgnv=sgnv)
                uvtra[ii*ntr:(ii+1)*ntr,jj*ntz:(jj+1)*ntz]= tuvtra
                uvtza[ii*ntr:(ii+1)*ntr,jj*ntz:(jj+1)*ntz]= tuvtza
                # Store u,v also encoding the sign info
                ut[ii*ntr:(ii+1)*ntr,jj*ntz:(jj+1)*ntz]= sgnu*us
                vt[ii*ntr:(ii+1)*ntr,jj*ntz:(jj+1)*ntz]= sgnv*vs
        # Now use Newton-Raphson to iterate to a regular grid in 
        # (thetara,thetaza)
        self._uvtra= uvtra
        self._uvtza= uvtza
        flat_uvtra= uvtra.flatten()
        flat_uvtza= uvtza.flatten()
        flat_ut= ut.flatten()
        flat_vt= vt.flatten()
        mtra, mtza= numpy.meshgrid(thetara,thetaza)
        # shape of those: ntz,ntr!!
        mtra= mtra.flatten()
        mtza= mtza.flatten()
        # Find closest matches as starting points
        kdt= cKDTree(numpy.array([flat_uvtra,flat_uvtza]).T)
        dist, cindx= kdt.query(numpy.array([mtra,mtza]).T)
        uout= flat_ut[cindx]
        vout= flat_vt[cindx]
        tra= flat_uvtra[cindx]
        tza= flat_uvtza[cindx]
        # Now iterate
        maxiter= 100
        tol= 1.48e-8
        cntr= 0
        unconv= numpy.ones(len(mtra),dtype='bool')
        dtra= (tra[unconv]-mtra[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
        dtza= (tza[unconv]-mtza[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
        unconv[unconv]= (dtra**2.+dtza**2.) > tol**2.
        cntr= 0
        while True:
            jac= _danglerz(numpy.fabs(uout[unconv]),numpy.fabs(vout[unconv]),
                           self._E,self._Lz2,self._I3,self._pot,
                           self._delta,self._isoaa_helper,
                           sgnu=numpy.sign(uout[unconv]),
                           sgnv=numpy.sign(vout[unconv]))
            detJ= jac[0]*jac[3]-jac[1]*jac[2]
            dtra= (tra[unconv]-mtra[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
            dtza= (tza[unconv]-mtza[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
            du= (jac[3]*dtra-jac[1]*dtza)/detJ
            dv= (-jac[2]*dtra+jac[0]*dtza)/detJ
            du[numpy.fabs(du) > maxdu]= numpy.sign(du[numpy.fabs(du) >maxdu])*maxdu
            dv[numpy.fabs(dv) > maxdv]= numpy.sign(dv[numpy.fabs(dv) >maxdv])*maxdv
            du[True^numpy.isfinite(du)]= maxdu\
                *(numpy.sign(numpy.fabs(uout[unconv][True^numpy.isfinite(du)])-(self._umin+self._umax)/2.))
            dv[True^numpy.isfinite(dv)]= maxdv\
                *(numpy.sign(numpy.fabs(vout[unconv][True^numpy.isfinite(dv)])-numpy.pi/2.))
            newu= numpy.fabs(uout[unconv]-numpy.sign(uout[unconv])*du)
            newv= numpy.fabs(vout[unconv]-numpy.sign(vout[unconv])*dv)
            # Bounce off the edge of the torus, decaying to avoid loops
            cntrpow= 3.
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newu < self._umin
            uout[unconv_off_grid]= -numpy.sign(uout[unconv_off_grid])\
                *(self._umin+maxdu*1e-4/(cntr+1)**cntrpow)
#                *(2.*self._umin-newu[newu < self._umin])
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newu > self._umax
            uout[unconv_off_grid]= -numpy.sign(uout[unconv_off_grid])\
                *(self._umax-maxdu*1e-4/(cntr+1)**cntrpow)
#                *(2.*self._umax-newu[newu > self._umax])
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newv < self._vmin
            vout[unconv_off_grid]= -numpy.sign(vout[unconv_off_grid])\
                *(self._vmin+maxdv*1e-4/(cntr+1)**cntrpow)
#                *(2.*self._vmin-newv[newv < self._vmin])
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newv > numpy.pi-self._vmin
            vout[unconv_off_grid]= -numpy.sign(vout[unconv_off_grid])\
                *(numpy.pi-self._vmin-maxdv*1e-4/(cntr+1)**cntrpow)
#                *(2.*numpy.pi-2.*self._vmin-newv[newv > numpy.pi-self._vmin])
            du[newu < self._umin]= 0.
            du[newu > self._umax]= 0.
            dv[newv < self._vmin]= 0.
            dv[newv > numpy.pi-self._vmin]= 0.
            uout[unconv]-= numpy.sign(uout[unconv])*du
            vout[unconv]-= numpy.sign(vout[unconv])*dv
            unconv[unconv]= (dtra**2.+dtza**2.) > tol**2.
            newtra, newtza= _anglerz(numpy.fabs(uout[unconv]),
                                     numpy.fabs(vout[unconv]),
                                     self._E,self._Lz2,self._I3,self._pot,
                                     self._delta,self._isoaa_helper,
                                     sgnu=numpy.sign(uout[unconv]),
                                     sgnv=numpy.sign(vout[unconv]))
            tra[unconv]= newtra
            tza[unconv]= newtza
            cntr+= 1
            if numpy.sum(unconv) == 0:
                print("Took %i iterations" % cntr)
                break
            if cntr > maxiter:
                break
                raise RuntimeError("Convergence of grid-finding not achieved in %i iterations" % maxiter)
        self._dtra= (tra-mtra+numpy.pi) % (2.*numpy.pi)-numpy.pi
        self._dtza= (tza-mtza+numpy.pi) % (2.*numpy.pi)-numpy.pi
        self._mtra= mtra
        self._mtza= mtza
        return(uout,vout)

def _anglerz(u,v,E,Lz2,I3,pot,delta,isoaa_helper,sgnu=1.,sgnv=1.):
    """
    NAME:
       _anglerz
    PURPOSE:
       compute (thetara,thetaza) angles in the isochrone potential for a grid
       in (u,v)
    INPUT:
       u - u
       v - v
       E - energy
       Lz2 - (z-component of the angular momentum)^2
       I3 - I_3
       pot - potential
       delta - focal length
       isoaa_helper - _actionAngleIsochroneHelper object
       sgnu= sign of p_u to choose
       sgnv= sign of p_v to choose
    OUTPUT:
       (thetara,thetarz)
    HISTORY:
       2017-12-07 - Written - Bovy (UofT)
    """
    # For Kuzmin?!
    v[v==0]= 1e-8
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    # Currently specific to Kuzmin!                             
    pu2= 2.*delta**2.*(E*numpy.sinh(u)**2.
                       +pot._amp/pot._a*numpy.cosh(u)-I3)\
                       -Lz2/numpy.sinh(u)**2.
    pv2= 2.*delta**2.*(E*numpy.sin(v)**2.
                       -pot._amp/pot._a*numpy.fabs(numpy.cos(v))
                       +I3)\
                       -Lz2/numpy.sin(v)**2.
    pu2[pu2 < 0.]= 0.
    pu= sgnu*numpy.sqrt(pu2)
    pv2[pv2 < 0.]= 0.
    pv= sgnv*numpy.sqrt(pv2)
    r= numpy.sqrt(R**2.+z**2.)
    vR,vz= bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)
    vr= vR*R/r+vz*z/r
    vtheta= vR*z/r-vz*R/r
    L2= r**2.*(vtheta**2.+Lz2/R**2.)
    return isoaa_helper.anglerz(r,vr**2.,L2,Lz2,z/r,vr < 0.,vtheta > 0.)

def _danglerz(u,v,E,Lz2,I3,pot,delta,isoaa_helper,sgnu=1.,sgnv=1.):
    # For Kuzmin?!
    v[v==0]= 1e-8
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    # Currently specific to Kuzmin!                             
    pu2= 2.*delta**2.*(E*numpy.sinh(u)**2.
                       +pot._amp/pot._a*numpy.cosh(u)-I3)\
                       -Lz2/numpy.sinh(u)**2.
    pv2= 2.*delta**2.*(E*numpy.sin(v)**2.
                       -pot._amp/pot._a*numpy.fabs(numpy.cos(v))
                       +I3)\
                       -Lz2/numpy.sin(v)**2.
    pu2[pu2 < 0.]= 0.
    pu= sgnu*numpy.sqrt(pu2)
    pv2[pv2 < 0.]= 0.
    pv= sgnv*numpy.sqrt(pv2)
    r= numpy.sqrt(R**2.+z**2.)
    vR,vz= bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)
    vr= vR*R/r+vz*z/r
    vtheta= vR*z/r-vz*R/r
    L2= r**2.*(vtheta**2.+Lz2/R**2.)
    dr2du= 2.*delta**2.*numpy.sinh(u)*numpy.cosh(u)
    dr2dv= -2.*delta**2.*numpy.sin(v)*numpy.cos(v)
    dEdu= pot.Rforce(R,z)*R/numpy.tanh(u)+pot.zforce(R,z)*z*numpy.tanh(u)\
        -isoaa_helper._ip.Rforce(r,0.)*dr2du/2./r
    dEdv= pot.Rforce(R,z)*R/numpy.tan(v)-pot.zforce(R,z)*z*numpy.tan(v)\
        -isoaa_helper._ip.Rforce(r,0.)*dr2dv/2./r  
    # SPECIFIC TO KUZMIN FOR NOW!
    dpudu= (delta**2.*(2.*E*numpy.sinh(u)*numpy.cosh(u)+pot._amp/pot._a*numpy.sinh(u))\
        +Lz2/numpy.sinh(u)**3.*numpy.cosh(u))/pu
    dVdv= pot._amp/pot._a*numpy.sin(v)
    dVdv[numpy.cos(v) < 0.]*= -1.
    dpvdv= (delta**2.*(2.*E*numpy.sin(v)*numpy.cos(v)+dVdv)\
        +Lz2/numpy.sin(v)**3.*numpy.cos(v))/pv
    return isoaa_helper.danglerzduv_constant_ELzI3(r,vr**2.+1e-12,L2,Lz2,z/r,
                                                   vr < 0.,vtheta > 0.,
                                                   u,v,pu,pv,delta**2.,
                                                   r**2.*vtheta**2.,
                                                   dEdu,dEdv,dpudu,dpvdv)

def _jrz(u,v,E,Lz2,I3,pot,delta,isoaa_helper,sgnu=1.,sgnv=1.):
    """
    NAME:
       _jrz
    PURPOSE:
       compute radial and vertical action in the isochrone potential for a 
       grid in (u,v)
    INPUT:
       u - u
       v - v
       E - energy
       Lz2 - (z-component of the angular momentum)^2
       I3 - I_3
       pot - potential
       delta - focal length
       isoaa_helper - _actionAngleIsochroneHelper object
       sgnu= sign of p_u to choose
       sgnv= sign of p_v to choose
    OUTPUT:
       jr,jz
    HISTORY:
       2017-12-08 - Written - Bovy (UofT)
    """
    # For Kuzmin?!
    v[v==0]= 1e-8
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    # Currently specific to Kuzmin!                             
    pu2= 2.*delta**2.*(E*numpy.sinh(u)**2.
                       +pot._amp/pot._a*numpy.cosh(u)-I3)\
                       -Lz2/numpy.sinh(u)**2.
    pv2= 2.*delta**2.*(E*numpy.sin(v)**2.
                       -pot._amp/pot._a*numpy.fabs(numpy.cos(v))
                       +I3)\
                       -Lz2/numpy.sin(v)**2.
    pu= sgnu*numpy.sqrt(pu2)
    pv= sgnv*numpy.sqrt(pv2)
    r= numpy.sqrt(R**2.+z**2.)
    vR,vz= bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)
    vr= vR*R/r+vz*z/r
    vtheta= vR*z/r-vz*R/r
    L2= r**2.*(vtheta**2.+Lz2/R**2.)
    E= isoaa_helper._ip(r,0.)+vr**2./2.+L2/2./r**2.
    L= numpy.sqrt(L2)
    return isoaa_helper.Jr(E,L), L-numpy.fabs(numpy.sqrt(Lz2))

def _anglerzero_eqs(u,v,tza,E,Lz2,I3,pot,delta,isoaa_helper,sgnu=1.,sgnv=1.):
    v[v==0]= 1e-8
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    # Currently specific to Kuzmin!                             
    pu2= 2.*delta**2.*(E*numpy.sinh(u)**2.
                       +pot._amp/pot._a*numpy.cosh(u)-I3)\
                       -Lz2/numpy.sinh(u)**2.
    pv2= 2.*delta**2.*(E*numpy.sin(v)**2.
                       -pot._amp/pot._a*numpy.fabs(numpy.cos(v))
                       +I3)\
                       -Lz2/numpy.sin(v)**2.
    pu= sgnu*numpy.sqrt(pu2)
    pu[pu2 < 0.]= 1e-12
    pv= sgnv*numpy.sqrt(pv2)
    pv[pv2 < 0.]= 1e-12
    r= numpy.sqrt(R**2.+z**2.)
    vR,vz= bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)
    vtheta= vR*z/r-vz*R/r
    L2= r**2.*(vtheta**2.+Lz2/R**2.)
    sini= numpy.sqrt(1.-Lz2/L2) 
    sini[Lz2/L2 > 1.]= 0.
    sinpsi= z/r/sini
    psi= numpy.arcsin(sinpsi)
    psi[vtheta > 0.]= numpy.pi-psi[vtheta > 0.]
    psi[True^numpy.isfinite(psi)]= 0.
    return ((numpy.sinh(u)*numpy.cosh(u)*pu)**2.
            -(numpy.sin(v)*numpy.cos(v)*pv)**2.,
            (psi-tza+numpy.pi) % (2.*numpy.pi)-numpy.pi)

def _danglerzero_eqs(u,v,E,Lz2,I3,pot,delta,isoaa_helper,sgnu=1.,sgnv=1.):
    # For Kuzmin?!
    v[v==0]= 1e-8
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    # Currently specific to Kuzmin!                             
    pu2= 2.*delta**2.*(E*numpy.sinh(u)**2.
                       +pot._amp/pot._a*numpy.cosh(u)-I3)\
                       -Lz2/numpy.sinh(u)**2.
    pv2= 2.*delta**2.*(E*numpy.sin(v)**2.
                       -pot._amp/pot._a*numpy.fabs(numpy.cos(v))
                       +I3)\
                       -Lz2/numpy.sin(v)**2.
    pu= sgnu*numpy.sqrt(pu2)
    pu[pu2 < 0.]= 1e-12
    pv= sgnv*numpy.sqrt(pv2)
    pv[pv2 < 0.]= 1e-12
    r= numpy.sqrt(R**2.+z**2.)
    vR,vz= bovy_coords.pupv_to_vRvz(pu,pv,u,v,delta=delta)
    vr= vR*R/r+vz*z/r
    vtheta= vR*z/r-vz*R/r
    L2= r**2.*(vtheta**2.+Lz2/R**2.)
    dr2du= 2.*delta**2.*numpy.sinh(u)*numpy.cosh(u)
    dr2dv= -2.*delta**2.*numpy.sin(v)*numpy.cos(v)
    dEdu= pot.Rforce(R,z)*R/numpy.tanh(u)+pot.zforce(R,z)*z*numpy.tanh(u)\
        -isoaa_helper._ip.Rforce(r,0.)*dr2du/2./r
    dEdv= pot.Rforce(R,z)*R/numpy.tan(v)-pot.zforce(R,z)*z*numpy.tan(v)\
        -isoaa_helper._ip.Rforce(r,0.)*dr2dv/2./r  
    # SPECIFIC TO KUZMIN FOR NOW!
    dpu2du= 2.*(delta**2.*(2.*E*numpy.sinh(u)*numpy.cosh(u)+pot._amp/pot._a*numpy.sinh(u))\
        +Lz2/numpy.sinh(u)**3.*numpy.cosh(u))
    dVdv= pot._amp/pot._a*numpy.sin(v)
    dVdv[numpy.cos(v) < 0.]*= -1.
    dpv2dv= 2.*(delta**2.*(2.*E*numpy.sin(v)*numpy.cos(v)+dVdv)\
        +Lz2/numpy.sin(v)**3.*numpy.cos(v))
    dpsidu, dpsidv=\
        isoaa_helper.dpsiduv_constant_ELzI3(L2,Lz2,z/r,
                                               vtheta > 0.,
                                               u,v,pu,pv,delta**2.,
                                               r**2.*vtheta**2.,
                                               dpu2du/pu/2.,dpv2dv/pv/2.)
    return (2.*numpy.cosh(2.*u)*numpy.sinh(u)*numpy.cosh(u)*pu2
            +(numpy.sinh(u)*numpy.cosh(u))**2.*dpu2du,
            -2.*numpy.cos(2.*v)*numpy.sin(v)*numpy.cos(v)*pv2
            -(numpy.sin(v)*numpy.cos(v))**2.*dpv2dv,
            dpsidu,dpsidv)
                
def code_to_solve_angler_boundary():
    if True:
        unconv= (mtra == 0.) + (mtra == numpy.pi)

        funval1, funval2= _anglerzero_eqs(numpy.fabs(uout[unconv]),
                                          numpy.fabs(vout[unconv]),
                                          mtza[unconv],
                                          self._E,self._Lz2,self._I3,self._pot,
                                          self._delta,self._isoaa_helper,
                                          sgnu=numpy.sign(uout[unconv]),
                                          sgnv=numpy.sign(vout[unconv]))
        while True:
            jac= _danglerzero_eqs(numpy.fabs(uout[unconv]),
                                  numpy.fabs(vout[unconv]),
                                  self._E,self._Lz2,self._I3,self._pot,
                                  self._delta,self._isoaa_helper,
                                  sgnu=numpy.sign(uout[unconv]),
                                  sgnv=numpy.sign(vout[unconv]))
            detJ= jac[0]*jac[3]-jac[1]*jac[2]
            du= (jac[3]*funval1-jac[1]*funval2)/detJ
            dv= (-jac[2]*funval1+jac[0]*funval2)/detJ
            cntrpow= 3.
            du[True^numpy.isfinite(du)]= maxdu/(cntr+1)**cntrpow\
                *(numpy.sign(numpy.fabs(uout[unconv][True^numpy.isfinite(du)])-(self._umin+self._umax)/2.))
            dv[True^numpy.isfinite(dv)]= maxdv/(cntr+1)**cntrpow\
                *(numpy.sign(numpy.fabs(vout[unconv][True^numpy.isfinite(dv)])-numpy.pi/2.))
            newu= numpy.fabs(uout[unconv]-numpy.sign(uout[unconv])*du)
            newv= numpy.fabs(vout[unconv]-numpy.sign(vout[unconv])*dv)
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newu < self._umin
            #uout[unconv_off_grid]= -uout[unconv][newu < self._umin]
            uout[unconv_off_grid]= -numpy.sign(uout[unconv_off_grid])\
                *(self._umin+maxdu*1e-4/(cntr+1)**cntrpow)
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newu > self._umax
            #uout[unconv_off_grid]= -uout[unconv][newu > self._umax]
            uout[unconv_off_grid]= -numpy.sign(uout[unconv_off_grid])\
                *(self._umax-maxdu*1e-4/(cntr+1)**cntrpow)
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newv < self._vmin
            #vout[unconv_off_grid]= -vout[unconv][newv < self._vmin]
            vout[unconv_off_grid]= -numpy.sign(vout[unconv_off_grid])\
                *(self._vmin+maxdv*1e-4/(cntr+1)**cntrpow)
            unconv_off_grid= numpy.copy(unconv)
            unconv_off_grid[unconv]= newv > numpy.pi-self._vmin
            #vout[unconv_off_grid]= \
            #    -vout[unconv][newv > numpy.pi-self._vmin]
            vout[unconv_off_grid]= -numpy.sign(vout[unconv_off_grid])\
                *(numpy.pi-self._vmin-maxdv*1e-4/(cntr+1)**cntrpow)
            uout[unconv]-= numpy.sign(uout[unconv])*du
            vout[unconv]-= numpy.sign(vout[unconv])*dv
            funval1, funval2= _anglerzero_eqs(numpy.fabs(uout[unconv]),
                                              numpy.fabs(vout[unconv]),
                                              mtza[unconv],
                                              self._E,self._Lz2,self._I3,self._pot,
                                              self._delta,self._isoaa_helper,
                                              sgnu=numpy.sign(uout[unconv]),
                                              sgnv=numpy.sign(vout[unconv]))
            unconv[unconv]= (funval1**2.+funval2**2.) > tol**2.
            funval1, funval2= _anglerzero_eqs(numpy.fabs(uout[unconv]),
                                              numpy.fabs(vout[unconv]),
                                              mtza[unconv],
                                              self._E,self._Lz2,self._I3,self._pot,
                                              self._delta,self._isoaa_helper,
                                              sgnu=numpy.sign(uout[unconv]),
                                              sgnv=numpy.sign(vout[unconv]))
            

            cntr+= 1
            if numpy.sum(unconv) == 0:
                break
            if cntr > maxiter:
                break
                raise RuntimeError("Convergence of grid-finding not achieved in %i iterations" % maxiter)
        # Update angles for these
        unconv= (mtra == 0.) + (mtra == numpy.pi)
        newtra, newtza= _anglerz(numpy.fabs(uout[unconv]),
                                 numpy.fabs(vout[unconv]),
                                 self._E,self._Lz2,self._I3,self._pot,
                                 self._delta,self._isoaa_helper,
                                 sgnu=numpy.sign(uout[unconv]),
                                 sgnv=numpy.sign(vout[unconv]))
        tra[unconv]= newtra
        tza[unconv]= newtza

        print(newtra)
        print(newtza)

        dtra= (tra[unconv]-mtra[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
        dtza= (tza[unconv]-mtza[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi

        print(dtra)
        print(dtza)

