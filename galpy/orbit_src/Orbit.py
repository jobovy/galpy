import numpy as nu
import galpy.util.bovy_coords as coords
from FullOrbit import FullOrbit
from RZOrbit import RZOrbit
from planarOrbit import planarOrbit, planarROrbit
from linearOrbit import linearOrbit
_K=4.74047
def _zEqZeroBC(ar):
    return ar[3]
def _vzEqZeroBC(ar):
    return ar[4]
def _REqZeroBC(ar):
    return ar[0]
def _REqOneBC(ar):
    return ar[0]-1.
def _vREqZeroBC(ar):
    return ar[1]
def _vTEqZeroBC(ar):
    return ar[2]
def _vTEqOneBC(ar):
    return ar[2]-1.
def _phiEqZeroBC(ar):
    if len(ar) > 4: return ar[5]
    else: return ar[3]
class Orbit:
    """General orbit class representing an orbit"""
    def __init__(self,vxvv=None,uvw=False,lb=False,
                 radec=False,vo=235.,ro=8.5,zo=0.025,
                 solarmotion='hogg'):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize an Orbit instance

        INPUT:

           vxvv - initial conditions 
                  3D can be either

              1) in Galactocentric cylindrical coordinates [R,vR,vT(,z,vz,phi)]

              2) [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (all J2000.0; mu_ra = mu_ra * cos dec)

              3) [ra,dec,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]

              4) (l,b,d,mu_l, mu_b, vlos) in [deg,deg,kpc,mas/yr,mas/yr,km/s) (all J2000.0; mu_l = mu_l * cos b)

              5) [l,b,d,U,V,W] in [deg,deg,kpc,km/s,km/s,kms]

           4) and 5) also work when leaving out b and mu_b/W

        OPTIONAL INPUTS:

           radec - if True, input is 2) (or 3) above

           uvw - if True, velocities are UVW

           lb - if True, input is 4) or 5) above

           vo - circular velocity at ro

           ro - distance from vantage point to GC (kpc)

           zo - offset toward the NGP of the Sun wrt the plane (kpc)

           solarmotion - 'hogg' or 'dehnen', or 'schoenrich', or value in 
           [-U,V,W]

        OUTPUT:

           instance

        HISTORY:

           2010-07-20 - Written - Bovy (NYU)

        """
        if isinstance(solarmotion,str) and solarmotion.lower() == 'hogg':
            vsolar= nu.array([-10.1,4.0,6.7])/vo
        elif isinstance(solarmotion,str) and solarmotion.lower() == 'dehnen':
            vsolar= nu.array([-10.,5.25,7.17])/vo
        elif isinstance(solarmotion,str) \
                and solarmotion.lower() == 'schoenrich':
            vsolar= nu.array([-11.1,12.24,7.25])/vo
        else:
            vsolar= nu.array(solarmotion)/vo           
        if radec or lb:
            if radec:
                l,b= coords.radec_to_lb(vxvv[0],vxvv[1],degree=True)
            elif len(vxvv) == 4:
                l, b= vxvv[0], 0.
            else:
                l,b= vxvv[0],vxvv[1]
            if uvw:
                X,Y,Z= coords.lbd_to_XYZ(l,b,vxvv[2],degree=True)
                vx= vxvv[3]
                vy= vxvv[4]
                vz= vxvv[5]
            else:
                if radec:
                    pmll, pmbb= coords.pmrapmdec_to_pmllpmbb(vxvv[3],vxvv[4],
                                                             vxvv[0],vxvv[1],
                                                             degree=True)
                    d, vlos= vxvv[2], vxvv[5]
                elif len(vxvv) == 4:
                    pmll, pmbb= vxvv[2], 0.
                    d, vlos= vxvv[1], vxvv[3]
                else:
                    pmll, pmbb= vxvv[3], vxvv[4]
                    d, vlos= vxvv[2], vxvv[5]
                X,Y,Z,vx,vy,vz= coords.sphergal_to_rectgal(l,b,d,
                                                           vlos,pmll, pmbb,
                                                           degree=True)
            X/= ro
            Y/= ro
            Z/= ro
            vx/= vo
            vy/= vo
            vz/= vo
            vsun= nu.array([0.,1.,0.,])+vsolar
            R, phi, z= coords.XYZ_to_galcencyl(X,Y,Z,Zsun=zo/ro)
            vR, vT,vz= coords.vxvyvz_to_galcencyl(vx,vy,vz,
                                                  R,phi,z,
                                                  vsun=vsun,galcen=True)
            if lb and len(vxvv) == 4: vxvv= [R,vR,vT,phi]
            else: vxvv= [R,vR,vT,z,vz,phi]
        self.vxvv= vxvv
        if len(vxvv) == 2:
            self._orb= linearOrbit(vxvv=vxvv)
        elif len(vxvv) == 3:
            self._orb= planarROrbit(vxvv=vxvv)
        elif len(vxvv) == 4:
            self._orb= planarOrbit(vxvv=vxvv)
        elif len(vxvv) == 5:
            self._orb= RZOrbit(vxvv=vxvv)
        elif len(vxvv) == 6:
            self._orb= FullOrbit(vxvv=vxvv)

    def setphi(self,phi):
        """

        NAME:

           setphi

        PURPOSE:

           set initial azimuth

        INPUT:

           phi - desired azimuth

        OUTPUT:

           (none)

        HISTORY:

           2010-08-01 - Written - Bovy (NYU)

        BUGS:

           Should perform check that this orbit has phi

        """
        if len(self.vxvv) == 2:
            raise AttributeError("One-dimensional orbit has no azimuth")
        elif len(self.vxvv) == 3:
            #Upgrade
            vxvv= [self.vxvv[0],self.vxvv[1],self.vxvv[2],phi]
            self.vxvv= vxvv
            self._orb= planarROrbit(vxvv=vxvv)
        elif len(self.vxvv) == 4:
            self.vxvv[-1]= phi
            self._orb.vxvv[-1]= phi
        elif len(self.vxvv) == 5:
            #Upgrade
            vxvv= [self.vxvv[0],self.vxvv[1],self.vxvv[2],self.vxvv[3],
                   self.vxvv[4],phi]
            self.vxvv= vxvv
            self._orb= FullOrbit(vxvv=vxvv)
        elif len(self.vxvv) == 6:
            self.vxvv[-1]= phi
            self._orb.vxvv[-1]= phi

    def dim(self):
        """
        NAME:

           dim

        PURPOSE:

           return the dimension of the problem

        INPUT:

           (none)

        OUTPUT:

           dimension

        HISTORY:

           2011-02-03 - Written - Bovy (NYU)

        """
        if len(self.vxvv) == 2:
            return 1
        elif len(self.vxvv) == 3 or len(self.vxvv) == 4:
            return 2
        elif len(self.vxvv) == 5 or len(self.vxvv) == 6:
            return 3

    def integrate(self,t,pot,method='leapfrog_c'):
        """
        NAME:

           integrate

        PURPOSE:

           integrate the orbit

        INPUT:

           t - list of times at which to output (0 has to be in this!)

           pot - potential instance or list of instances

           method= 'odeint' for scipy's odeint or 'leapfrog' for a simple leapfrog implementation

        OUTPUT:

           (none) (get the actual orbit using getOrbit()

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.integrate(t,pot,method=method)

    def integrateBC(self,pot,bc=_zEqZeroBC,method='odeint'):
        """
        NAME:

           integrateBC

        PURPOSE:

           integrate the orbit subject to a final boundary condition

        INPUT:

           pot - potential instance or list of instances

           bc= boundary condition, takes array of phase-space position (in the manner that is relevant to the type of Orbit) and outputs the condition that should be zero; default: z=0

           method= 'odeint' for scipy's odeint integrator, 'leapfrog' for
                   a simple symplectic integrator

        OUTPUT:
        
           (Another Orbit instance,time at which BC is reached)

        HISTORY:

           2011-09-30

        """
        o,tout= self._orb.integrateBC(pot,bc=bc,method=method)
        return (Orbit(vxvv=o[1,:]),tout)

    def reverse(self):
        """
        NAME:

           reverse

        PURPOSE:

           reverse an already integrated orbit (that is, make it go from end to beginning in t=0 to tend)

        INPUT:

           (none)

        OUTPUT:

           (none)

        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        sortindx = range(len(self._orb.t))
        sortindx.sort(lambda x,y: cmp(self._orb.t[x],self._orb.t[y]),
                      reverse=True)
        for ii in range(self._orb.orbit.shape[1]):
            self._orb.orbit[:,ii]= self._orb.orbit[sortindx,ii]
        return None

    def getOrbit(self):
        """

        NAME:

           getOrbit

        PURPOSE:

           return a previously calculated orbit

        INPUT:

           (none)

        OUTPUT:

           array orbit[nt,nd]

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        return self._orb.getOrbit()

    def E(self,*args,**kwargs):
        """
        NAME:

           E

        PURPOSE:

           calculate the energy

        INPUT:

           t - (optional) time at which to get the energy

           pot= Potential instance or list of such instances

        OUTPUT:

           energy

        HISTORY:

           2010-09-15 - Written - Bovy (NYU)

        """
        return self._orb.E(*args,**kwargs)

    def L(self,*args,**kwargs):
        """
        NAME:

           L

        PURPOSE:

           calculate the angular momentum at time t

        INPUT:

           t - (optional) time at which to get the angular momentum

        OUTPUT:

           angular momentum

        HISTORY:

           2010-09-15 - Written - Bovy (NYU)

        """
        return self._orb.L(*args,**kwargs)

    def ER(self,*args,**kwargs):
        """
        NAME:

           ER

        PURPOSE:

           calculate the radial energy

        INPUT:

           t - (optional) time at which to get the radial energy

           pot= Potential instance or list of such instances

        OUTPUT:

           radial energy

        HISTORY:

           2013-11-30 - Written - Bovy (IAS)

        """
        return self._orb.ER(*args,**kwargs)

    def Ez(self,*args,**kwargs):
        """
        NAME:

           Ez

        PURPOSE:

           calculate the vertical energy

        INPUT:

           t - (optional) time at which to get the vertical energy

           pot= Potential instance or list of such instances

        OUTPUT:

           vertical energy

        HISTORY:

           2013-11-30 - Written - Bovy (IAS)

        """
        return self._orb.Ez(*args,**kwargs)

    def Jacobi(self,*args,**kwargs):
        """
        NAME:

           Jacobi

        PURPOSE:

           calculate the Jacobi integral E - Omega L

        INPUT:

           t - (optional) time at which to get the Jacobi integral

           OmegaP= pattern speed
           
           pot= potential instance or list of such instances

        OUTPUT:

           Jacobi integral

        HISTORY:

           2011-04-18 - Written - Bovy (NYU)

        """
        return self._orb.Jacobi(*args,**kwargs)

    def e(self,analytic=False,pot=None):
        """
        NAME:

           e

        PURPOSE:

           calculate the eccentricity

        INPUT:

           analytic - compute this analytically

           pot - potential to use for analytical calculation

        OUTPUT:

           eccentricity

        HISTORY:

           2010-09-15 - Written - Bovy (NYU)

        """
        return self._orb.e(analytic=analytic,pot=pot)

    def rap(self,analytic=False,pot=None):
        """
        NAME:

           rap

        PURPOSE:

           calculate the apocenter radius

        INPUT:

           analytic - compute this analytically

           pot - potential to use for analytical calculation

        OUTPUT:

           R_ap

        HISTORY:

           2010-09-20 - Written - Bovy (NYU)

        """
        return self._orb.rap(analytic=analytic,pot=pot)

    def rperi(self,analytic=False,pot=None):
        """
        NAME:

           rperi

        PURPOSE:

           calculate the pericenter radius

        INPUT:

           analytic - compute this analytically

           pot - potential to use for analytical calculation

        OUTPUT:

           R_peri

        HISTORY:

           2010-09-20 - Written - Bovy (NYU)

        """
        return self._orb.rperi(analytic=analytic,pot=pot)

    def zmax(self,analytic=False,pot=None):
        """
        NAME:

           zmax

        PURPOSE:

           calculate the maximum vertical height

        INPUT:

           analytic - compute this analytically

           pot - potential to use for analytical calculation

        OUTPUT:

           Z_max

        HISTORY:

           2010-09-20 - Written - Bovy (NYU)

        """
        return self._orb.zmax(analytic=analytic,pot=pot)

    def resetaA(self,pot=None,type=None):
        """
        NAME:

           resetaA

        PURPOSE:

           re-set up an actionAngle module for this Orbit

        INPUT:

           (none)

        OUTPUT:

           True if reset happened, False otherwise

        HISTORY:

           2014-01-06 - Written - Bovy (IAS)

        """
        try:
            delattr(self._orb,'_aA')
        except AttributeError:
            return False
        else:
            return True

    def jr(self,pot=None,**kwargs):
        """
        NAME:

           jr

        PURPOSE:

           calculate the radial action

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           jr

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA(self)[0]

    def jp(self,pot=None,**kwargs):
        """
        NAME:

           jp

        PURPOSE:

           calculate the azimuthal action

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           jp

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA(self)[1]

    def jz(self,pot=None,**kwargs):
        """
        NAME:

           jz

        PURPOSE:

           calculate the vertical action

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           jz

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA(self)[2]

    def wr(self,pot=None,**kwargs):
        """
        NAME:

           wr

        PURPOSE:

           calculate the radial angle

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           wr

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA.actionsFreqsAngles(self)[6][0]

    def wp(self,pot=None,**kwargs):
        """
        NAME:

           wp

        PURPOSE:

           calculate the azimuthal angle

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           wp

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA.actionsFreqsAngles(self)[7][0]

    def wz(self,pot=None,**kwargs):
        """
        NAME:

           wz

        PURPOSE:

           calculate the vertical angle

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           wz

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA.actionsFreqsAngles(self)[8][0]

    def Tr(self,pot=None,**kwargs):
        """
        NAME:

           Tr

        PURPOSE:

           calculate the radial period

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Tr

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return 2.*nu.pi/self._orb._aA.actionsFreqs(self)[3][0]

    def Tp(self,pot=None,**kwargs):
        """
        NAME:

           Tp

        PURPOSE:

           calculate the azimuthal period

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Tp

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return 2.*nu.pi/self._orb._aA.actionsFreqs(self)[4][0]

    def TrTp(self,pot=None,**kwargs):
        """
        NAME:

           TrTp

        PURPOSE:

           the 'ratio' between the radial and azimuthal period Tr/Tphi*pi

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Tr/Tp*pi

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA.actionsFreqs(self)[4][0]/self._orb._aA.actionsFreqs(self)[3][0]*nu.pi
 
    def Tz(self,pot=None,**kwargs):
        """
        NAME:

           Tz

        PURPOSE:

           calculate the vertical period

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Tz

        HISTORY:

           2012-06-01 - Written - Bovy (IAS)

           2013-11-27 - Re-written using new actionAngle modules - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return 2.*nu.pi/self._orb._aA.actionsFreqs(self)[5][0]

    def Or(self,pot=None,**kwargs):
        """
        NAME:

           Or

        PURPOSE:

           calculate the radial frequency

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Or

        HISTORY:

           2013-11-27 - Written - Bovy (IAS)

        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA.actionsFreqs(self)[3][0]

    def Op(self,pot=None,**kwargs):
        """
        NAME:

           Op

        PURPOSE:

           calculate the azimuthal frequency

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Op

        HISTORY:

           2013-11-27 - Written - Bovy (IAS)
        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA.actionsFreqs(self)[4][0]

    def Oz(self,pot=None,**kwargs):
        """
        NAME:

           Oz

        PURPOSE:

           calculate the vertical frequency

        INPUT:

           pot - potential

           type= ('adiabatic') type of actionAngle module to use

              1) 'adiabatic'

              2) 'staeckel'

              3) 'isochroneApprox'

              4) 'spherical'
              
           +actionAngle module setup kwargs

        OUTPUT:

           Oz

        HISTORY:

           2013-11-27 - Written - Bovy (IAS)
        """
        self._orb._setupaA(pot=pot,**kwargs)
        return self._orb._aA.actionsFreqs(self)[5][0]

    def R(self,*args,**kwargs):
        """
        NAME:

           R

        PURPOSE:

           return cylindrical radius at time t

        INPUT:

           t - (optional) time at which to get the radius

        OUTPUT:

           R(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.R(*args,**kwargs)

    def vR(self,*args,**kwargs):
        """
        NAME:

           vR

        PURPOSE:

           return radial velocity at time t

        INPUT:

           t - (optional) time at which to get the radial velocity

        OUTPUT:

           vR(t)

        HISTORY:


           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vR(*args,**kwargs)

    def vT(self,*args,**kwargs):
        """
        NAME:

           vT

        PURPOSE:

           return tangential velocity at time t

        INPUT:

           t - (optional) time at which to get the tangential velocity

        OUTPUT:

           vT(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vT(*args,**kwargs)

    def z(self,*args,**kwargs):
        """
        NAME:

           z

        PURPOSE:

           return vertical height

        INPUT:

           t - (optional) time at which to get the vertical height

        OUTPUT:

           z(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.z(*args,**kwargs)

    def vz(self,*args,**kwargs):
        """
        NAME:

           vz

        PURPOSE:

           return vertical velocity

        INPUT:

           t - (optional) time at which to get the vertical velocity

        OUTPUT:

           vz(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vz(*args,**kwargs)

    def phi(self,*args,**kwargs):
        """
        NAME:

           phi

        PURPOSE:

           return azimuth

        INPUT:

           t - (optional) time at which to get the azimuth

        OUTPUT:

           phi(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.phi(*args,**kwargs)

    def vphi(self,*args,**kwargs):
        """
        NAME:

           vphi

        PURPOSE:

           return angular velocity

        INPUT:

           t - (optional) time at which to get the angular velocity

        OUTPUT:

           vphi(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.vphi(*args,**kwargs)

    def x(self,*args,**kwargs):
        """
        NAME:

           x

        PURPOSE:

           return x

        INPUT:

           t - (optional) time at which to get x

        OUTPUT:

           x(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.x(*args,**kwargs)

    def y(self,*args,**kwargs):
        """
        NAME:

           y

        PURPOSE:

           return y

        INPUT:

           t - (optional) time at which to get y

        OUTPUT:

           y(t)

        HISTORY:

           2010-09-21 - Written - Bovy (NYU)

        """
        return self._orb.y(*args,**kwargs)

    def vx(self,*args,**kwargs):
        """
        NAME:

           vx

        PURPOSE:

           return x velocity at time t

        INPUT:

           t - (optional) time at which to get the velocity

        OUTPUT:

           vx(t)

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        return self._orb.vx(*args,**kwargs)

    def vy(self,*args,**kwargs):
        """

        NAME:

           vy

        PURPOSE:

           return y velocity at time t

        INPUT:

           t - (optional) time at which to get the velocity

        OUTPUT:

           vy(t)

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        return self._orb.vy(*args,**kwargs)

    def ra(self,*args,**kwargs):
        """
        NAME:

           ra

        PURPOSE:

           return the right ascension

        INPUT:

           t - (optional) time at which to get ra

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
           (default=[8.5,0.,0.]) OR Orbit object that corresponds to the orbit of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)

        OUTPUT:

           ra(t)

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        return self._orb.ra(*args,**kwargs)

    def dec(self,*args,**kwargs):
        """
        NAME:

           dec

        PURPOSE:

           return the declination

        INPUT:

           t - (optional) time at which to get dec

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
           (default=[8.5,0.,0.]) OR Orbit object that corresponds to the orbit of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)

        OUTPUT:

           dec(t)

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        return self._orb.dec(*args,**kwargs)
    
    def ll(self,*args,**kwargs):
        """
        NAME:

           ll

        PURPOSE:

           return Galactic longitude

        INPUT:

           t - (optional) time at which to get ll

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
           (default=[8.5,0.,0.]) OR Orbit object that corresponds to the orbit of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

        OUTPUT:

           l(t)

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        return self._orb.ll(*args,**kwargs)

    def bb(self,*args,**kwargs):
        """
        NAME:

           bb

        PURPOSE:

           return Galactic latitude

        INPUT:

           t - (optional) time at which to get bb

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
           (default=[8.5,0.,0.]) OR Orbit object that corresponds to the orbit of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

        OUTPUT:

           b(t)

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        return self._orb.bb(*args,**kwargs)

    def dist(self,*args,**kwargs):
        """
        NAME:

           dist

        PURPOSE:

           return distance from the observer

        INPUT:

           t - (optional) time at which to get dist

           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
           (default=[8.5,0.,0.]) OR Orbit object that corresponds to the orbit of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

        OUTPUT:

           dist(t) in kpc

        HISTORY:

           2011-02-23 - Written - Bovy (NYU)

        """
        return self._orb.dist(*args,**kwargs)

    def pmra(self,*args,**kwargs):
        """
        NAME:

           pmra

        PURPOSE:

           return proper motion in right ascension (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmra

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           pm_ra(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.pmra(*args,**kwargs)

    def pmdec(self,*args,**kwargs):
        """
        NAME:

           pmdec

        PURPOSE:

           return proper motion in declination (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmdec

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           pm_dec(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.pmdec(*args,**kwargs)

    def pmll(self,*args,**kwargs):
        """
        NAME:

           pmll

        PURPOSE:

           return proper motion in Galactic longitude (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmll

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           pm_l(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.pmll(*args,**kwargs)

    def pmbb(self,*args,**kwargs):
        """
        NAME:

           pmbb

        PURPOSE:

           return proper motion in Galactic latitude (in mas/yr)

        INPUT:

           t - (optional) time at which to get pmbb

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           pm_b(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.pmbb(*args,**kwargs)

    def vlos(self,*args,**kwargs):
        """
        NAME:

           vlos

        PURPOSE:

           return the line-of-sight velocity (in km/s)

        INPUT:

           t - (optional) time at which to get vlos

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           vlos(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.vlos(*args,**kwargs)

    def vra(self,*args,**kwargs):
        """
        NAME:

           vra

        PURPOSE:

           return velocity in right ascension (km/s)

        INPUT:

           t - (optional) time at which to get vra

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           v_ra(t) in km/s

        HISTORY:

           2011-03-27 - Written - Bovy (NYU)

        """
        return self._orb.dist(*args,**kwargs)*_K*\
            self._orb.pmra(*args,**kwargs)

    def vdec(self,*args,**kwargs):
        """
        NAME:

           vdec

        PURPOSE:

           return velocity in declination (km/s)

        INPUT:

           t - (optional) time at which to get vdec

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           v_dec(t) in km/s

        HISTORY:

           2011-03-27 - Written - Bovy (NYU)

        """
        return self._orb.dist(*args,**kwargs)*_K*\
            self._orb.pmdec(*args,**kwargs)

    def vll(self,*args,**kwargs):
        """
        NAME:

           vll

        PURPOSE:

           return the velocity in Galactic longitude (km/s)

        INPUT:

           t - (optional) time at which to get vll

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           v_l(t) in km/s

        HISTORY:

           2011-03-27 - Written - Bovy (NYU)

        """
        return self._orb.dist(*args,**kwargs)*_K*\
            self._orb.pmll(*args,**kwargs)

    def vbb(self,*args,**kwargs):
        """
        NAME:

           vbb

        PURPOSE:

            return velocity in Galactic latitude (km/s)

        INPUT:

           t - (optional) time at which to get vbb

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           v_b(t) in km/s

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.dist(*args,**kwargs)*_K*\
            self._orb.pmbb(*args,**kwargs)

    def helioX(self,*args,**kwargs):
        """
        NAME:

           helioX

        PURPOSE:

           return Heliocentric Galactic rectangular x-coordinate (aka "X")

        INPUT:

           t - (optional) time at which to get X

           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

        OUTPUT:

           helioX(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.helioX(*args,**kwargs)

    def helioY(self,*args,**kwargs):
        """
        NAME:

           helioY

        PURPOSE:

           return Heliocentric Galactic rectangular y-coordinate (aka "Y")

        INPUT:

           t - (optional) time at which to get Y

           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

        OUTPUT:

           helioY(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.helioY(*args,**kwargs)

    def helioZ(self,*args,**kwargs):
        """
        NAME:

           helioZ

        PURPOSE:

           return Heliocentric Galactic rectangular z-coordinate (aka "Z")

        INPUT:

           t - (optional) time at which to get Z

           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

        OUTPUT:

           helioZ(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.helioZ(*args,**kwargs)

    def U(self,*args,**kwargs):
        """
        NAME:

           U

        PURPOSE:

           return Heliocentric Galactic rectangular x-velocity (aka "U")

        INPUT:

           t - (optional) time at which to get U

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           U(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.U(*args,**kwargs)

    def V(self,*args,**kwargs):
        """
        NAME:

           V

        PURPOSE:

           return Heliocentric Galactic rectangular y-velocity (aka "V")

        INPUT:

           t - (optional) time at which to get U

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           V(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.V(*args,**kwargs)

    def W(self,*args,**kwargs):
        """
        NAME:

           W

        PURPOSE:

           return Heliocentric Galactic rectangular z-velocity (aka "W")

        INPUT:

           t - (optional) time at which to get W

           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer

           ro= distance in kpc corresponding to R=1. (default: 8.5)         

           vo= velocity in km/s corresponding to v=1. (default: 235.)

        OUTPUT:

           W(t)

        HISTORY:

           2011-02-24 - Written - Bovy (NYU)

        """
        return self._orb.W(*args,**kwargs)

    def __call__(self,*args,**kwargs):
        """
        NAME:
 
          __call__

        PURPOSE:

           return the orbit at time t

        INPUT:

           t - desired time

           rect - if true, return rectangular coordinates

        OUTPUT:

           an Orbit instance with initial condition set to the 
           phase-space at time t or list of Orbit instances if multiple 
           times are given

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        thiso= self._orb(*args,**kwargs)
        if len(thiso.shape) == 1: return Orbit(vxvv=thiso)
        else: return [Orbit(vxvv=thiso[:,ii]) for ii in range(thiso.shape[1])]

    def plot(self,*args,**kwargs):
        """
        NAME:

           plot

        PURPOSE:

           plot a previously calculated orbit (with reasonable defaults)

        INPUT:

           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...)

           d2= second dimension to plot

           matplotlib.plot inputs+bovy_plot.plot inputs

        OUTPUT:

           sends plot to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plot(*args,**kwargs)

    def plot3d(self,*args,**kwargs):
        """
        NAME:

           plot3d

        PURPOSE:

           plot 3D aspects of an Orbit

        INPUT:

           bovy_plot3d args and kwargs

        OUTPUT:

           plot

        HISTORY:

           2010-07-26 - Written - Bovy (NYU)

           2010-09-22 - Adapted to more general framework - Bovy (NYU)

           2010-01-08 - Adapted to 3D - Bovy (NYU)
        """
        self._orb.plot3d(*args,**kwargs)

    def plotE(self,*args,**kwargs):
        """
        NAME:

           plotE

        PURPOSE:

           plot E(.) along the orbit

        INPUT:

           pot= Potential instance or list of instances in which the orbit was integrated

           d1= plot Ez vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotE(*args,**kwargs)

    def plotEz(self,*args,**kwargs):
        """
        NAME:

           plotEz

        PURPOSE:

           plot E_z(.) along the orbit

        INPUT:

           pot=  Potential instance or list of instances in which the orbit was integrated

           d1= plot Ez vs d1: e.g., 't', 'z', 'R'

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotEz(*args,**kwargs)

    def plotEzJz(self,*args,**kwargs):
        """
        NAME:

           plotEzJzt

        PURPOSE:

           plot E_z(t)/sqrt(dens(R)) along the orbit (an approximation to the vertical action)

        INPUT:

           pot - Potential instance or list of instances in which the orbit was integrated

           d1= plot Ez vs d1: e.g., 't', 'z', 'R'

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-08-08 - Written - Bovy (NYU)

        """
        self._orb.plotEzJz(*args,**kwargs)

    def plotJacobi(self,*args,**kwargs):
        """
        NAME:

           plotJacobi

        PURPOSE:

           plot the Jacobi integral along the orbit

        INPUT:

           OmegaP= pattern speed

           pot= - Potential instance or list of instances in which the orbit 
                 was integrated

           d1= - plot Ez vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      

           +bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2011-10-10 - Written - Bovy (IAS)

        """
        self._orb.plotJacobi(*args,**kwargs)

    def plotR(self,*args,**kwargs):
        """
        NAME:

           plotR

        PURPOSE:

           plot R(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotR(*args,**kwargs)

    def plotz(self,*args,**kwargs):
        """
        NAME:

           plotz

        PURPOSE:

           plot z(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotz(*args,**kwargs)

    def plotvR(self,*args,**kwargs):
        """
        NAME:

           plotvR

        PURPOSE:

           plot vR(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotvR(*args,**kwargs)

    def plotvT(self,*args,**kwargs):
        """
        NAME:

           plotvT

        PURPOSE:

           plot vT(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotvT(*args,**kwargs)

    def plotphi(self,*args,**kwargs):
        """
        NAME:

           plotphi

        PURPOSE:

           plot \phi(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotphi(*args,**kwargs)

    def plotvz(self,*args,**kwargs):
        """
        NAME:

           plotvz

        PURPOSE:

           plot vz(.) along the orbit

        INPUT:
           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        self._orb.plotvz(*args,**kwargs)

    def plotx(self,*args,**kwargs):
        """
        NAME:

           plotx

        PURPOSE:

           plot x(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        self._orb.plotx(*args,**kwargs)

    def plotvx(self,*args,**kwargs):
        """
        NAME:

           plotvx

        PURPOSE:

           plot vx(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        self._orb.plotvx(*args,**kwargs)

    def ploty(self,*args,**kwargs):
        """
        NAME:

           ploty

        PURPOSE:

           plot y(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        self._orb.ploty(*args,**kwargs)

    def plotvy(self,*args,**kwargs):
        """
        NAME:

           plotvy

        PURPOSE:

           plot vy(.) along the orbit

        INPUT:

           d1= plot vs d1: e.g., 't', 'z', 'R'

           bovy_plot.bovy_plot inputs

        OUTPUT:

           figure to output device

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        self._orb.plotvy(*args,**kwargs)

    def toPlanar(self):
        """
        NAME:

           toPlanar

        PURPOSE:

           convert a 3D orbit into a 2D orbit

        INPUT:

           (none)

        OUTPUT:

           planar Orbit

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        if len(self.vxvv) == 6:
            vxvv= [self.vxvv[0],self.vxvv[1],self.vxvv[2],self.vxvv[5]]
        elif len(self.vxvv) == 5:
            vxvv= [self.vxvv[0],self.vxvv[1],self.vxvv[2]]
        else:
            raise AttributeError("planar or linear Orbits do not have the toPlanar attribute")
        return Orbit(vxvv=vxvv)

    def toLinear(self):
        """
        NAME:

           toLinear

        PURPOSE:

           convert a 3D orbit into a 1D orbit (z)

        INPUT:

           (none)

        OUTPUT:

           linear Orbit

        HISTORY:

           2010-11-30 - Written - Bovy (NYU)

        """
        if len(self.vxvv) == 6 or len(self.vxvv) == 5:
            vxvv= [self.vxvv[3],self.vxvv[4]]
        else:
            raise AttributeError("planar or linear Orbits do not have the toPlanar attribute")
        return Orbit(vxvv=vxvv)

    def __add__(self,linOrb):
        """
        NAME:

           __add__

        PURPOSE:

           add a linear orbit and a planar orbit to make a 3D orbit

        INPUT:

           linear or plane orbit instance

        OUTPUT:

           3D orbit

        HISTORY:

           2010-07-21 - Written - Bovy (NYU)

        """
        if (not (isinstance(self._orb,planarROrbit) and 
                isinstance(linOrb._orb,linearOrbit)) and
            not (isinstance(self._orb,linearOrbit) and 
                 isinstance(linOrb._orb,planarROrbit))):
            raise AttributeError("Only planarROrbit+linearOrbit is supported")
        if isinstance(self._orb,planarROrbit):
            return Orbit(vxvv=[self._orb.vxvv[0],self._orb.vxvv[1],
                               self._orb.vxvv[2],
                               linOrb._orb.vxvv[0],linOrb._orb.vxvv[1]])
        else:
            return Orbit(vxvv=[linOrb._orb.vxvv[0],linOrb._orb.vxvv[1],
                               linOrb._orb.vxvv[2],
                               self._orb.vxvv[0],self._orb.vxvv[1]])

    #4 pickling
    def __getinitargs__(self):
        return (self.vxvv,)

    def __getstate__(self):
        return self.vxvv
    
    def __setstate__(self,state):
        self.vxvv= state
