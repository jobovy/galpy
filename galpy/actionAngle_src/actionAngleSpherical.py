###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleSpherical
#
#      methods:
#             __call__: returns (jr,lz,jz)
#             actionsFreqs: returns (jr,lz,jz,Or,Op,Oz)
#             actionsFreqsAngles: returns (jr,lz,jz,Or,Op,Oz,ar,ap,az)
#
###############################################################################
import copy
import math as m
import numpy as nu
from scipy import integrate, optimize
from galpy.potential import vcirc, epifreq, omegac
from galpy.potential_src.Potential import _evaluatePotentials
from galpy.potential_src.planarPotential import _evaluateplanarPotentials
from galpy.actionAngle_src.actionAngle import actionAngle, UnboundError
_EPS= 10.**-15.
class actionAngleSpherical(actionAngle):
    """Action-angle formalism for spherical potentials"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleSpherical object

        INPUT:

           pot= a Spherical potential

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

           gamma= (default=0.) replace Lz by Lz+gamma Jz in effective potential when using this class as part of actionAngleAdiabatic

        OUTPUT:

           instance

        HISTORY:

           2013-12-28 - Written - Bovy (IAS)

        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleSpherical")
        self._pot= kwargs['pot']
        #Also store a 'planar' (2D) version of the potential, only potential
        # used in this class
        if isinstance(self._pot,list):
            self._2dpot= [p.toPlanar() for p in self._pot]
        else:
            self._2dpot= self._pot.toPlanar()
        #The following for if we ever implement this code in C
        self._c= False
        ext_loaded= False
        if ext_loaded and (('c' in kwargs and kwargs['c'])
                           or not 'c' in kwargs):
            self._c= True #pragma: no cover
        else:
            self._c= False
        # gamma for when we use this as part of the adiabatic approx.
        self._gamma= kwargs.get('gamma',0.)
        # Check the units
        self._check_consistent_units()
        return None

    def _evaluate(self,*args,**kwargs):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           fixed_quad= (False) if True, use n=10 fixed_quad integration
           scipy.integrate.quadrature keywords
        OUTPUT:
           (jr,lz,jz)
        HISTORY:
           2013-12-28 - Written - Bovy (IAS)
        """
        fixed_quad= kwargs.pop('fixed_quad',False)
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
        if self._c: #pragma: no cover
            pass
        else:
            r= nu.sqrt(R**2.+z**2.)
            vr= (R*vR+z*vz)/r
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= _evaluateplanarPotentials(self._2dpot,r)\
                +vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            vt= L/r
            #Actions
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            #Jr requires some more work
            Jr= []
            for ii in range(len(r)):
                rperi,rap= self._calc_rperi_rap(r[ii],vr[ii],vt[ii],
                                                E[ii],L[ii])
                Jr.append(self._calc_jr(rperi,rap,E[ii],L[ii],
                                        fixed_quad,**kwargs))
            return (nu.array(Jr),Jphi,Jz)

    def _actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           _actionsFreqs
        PURPOSE:
           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           fixed_quad= (False) if True, use n=10 fixed_quad integration
           scipy.integrate.quadrature keywords
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        HISTORY:
           2013-12-28 - Written - Bovy (IAS)
        """
        fixed_quad= kwargs.pop('fixed_quad',False)
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
        if self._c: #pragma: no cover
            pass
        else:
            r= nu.sqrt(R**2.+z**2.)
            vr= (R*vR+z*vz)/r
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= _evaluateplanarPotentials(self._2dpot,r)\
                +vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            vt= L/r
            #Actions
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            #Jr requires some more work
            Jr= []
            Or= []
            Op= []
            for ii in range(len(r)):
                rperi,rap= self._calc_rperi_rap(r[ii],vr[ii],vt[ii],
                                                E[ii],L[ii])
                Jr.append(self._calc_jr(rperi,rap,E[ii],L[ii],
                                        fixed_quad,**kwargs))
                #Radial period
                if Jr[-1] < 10.**-9.: #Circular orbit
                    Or.append(epifreq(self._2dpot,r[ii],use_physical=False))
                    Op.append(omegac(self._2dpot,r[ii],use_physical=False))
                    continue
                Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
                Or.append(self._calc_or(Rmean,rperi,rap,E[ii],L[ii],
                                        fixed_quad,**kwargs))
                Op.append(self._calc_op(Or[-1],Rmean,rperi,rap,E[ii],L[ii],
                                        fixed_quad,**kwargs))
            Op= nu.array(Op)
            Oz= copy.copy(Op)
            Op[vT < 0.]*= -1.
            return (nu.array(Jr),Jphi,Jz,nu.array(Or),Op,Oz)
    
    def _actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           _actionsFreqsAngles
        PURPOSE:
           evaluate the actions, frequencies, and angles
           (jr,lz,jz,Omegar,Omegaphi,Omegaz,ar,ap,az)
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           fixed_quad= (False) if True, use n=10 fixed_quad integration
           scipy.integrate.quadrature keywords
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,ar,aphi,az)
        HISTORY:
           2013-12-29 - Written - Bovy (IAS)
        """
        fixed_quad= kwargs.pop('fixed_quad',False)
        if len(args) == 5: #R,vR.vT, z, vz pragma: no cover
            raise IOError("You need to provide phi when calculating angles")
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
        else:
            self._parse_eval_args(*args)
            R= self._eval_R
            vR= self._eval_vR
            vT= self._eval_vT
            z= self._eval_z
            vz= self._eval_vz
            phi= self._eval_phi
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
            phi= nu.array([phi])
        if self._c: #pragma: no cover
            pass
        else:
            r= nu.sqrt(R**2.+z**2.)
            vr= (R*vR+z*vz)/r
            vtheta= (z*vR-R*vz)/r
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= _evaluateplanarPotentials(self._2dpot,r)\
                +vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            vt= L/r
            #Actions
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            #Jr requires some more work
            Jr= []
            Or= []
            Op= []
            ar= []
            az= []
            #Calculate the longitude of the ascending node
            asc= self._calc_long_asc(z,R,vtheta,phi,Lz,L)
            for ii in range(len(r)):
                rperi,rap= self._calc_rperi_rap(r[ii],vr[ii],vt[ii],
                                                E[ii],L[ii])
                Jr.append(self._calc_jr(rperi,rap,E[ii],L[ii],
                                        fixed_quad,**kwargs))
                #Radial period
                Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
                if Jr[-1] < 10.**-9.: #Circular orbit
                    Or.append(epifreq(self._2dpot,r[ii],use_physical=False))
                    Op.append(omegac(self._2dpot,r[ii],use_physical=False))
                else:
                    Or.append(self._calc_or(Rmean,rperi,rap,E[ii],L[ii],
                                            fixed_quad,**kwargs))
                    Op.append(self._calc_op(Or[-1],Rmean,rperi,rap,E[ii],L[ii],
                                            fixed_quad,**kwargs))
                #Angles
                ar.append(self._calc_angler(Or[-1],r[ii],Rmean,rperi,rap,
                                            E[ii],L[ii],
                                            vr[ii],fixed_quad,**kwargs))
                az.append(self._calc_anglez(Or[-1],Op[-1],ar[-1],
                                            z[ii],r[ii],
                                            Rmean,rperi,rap,E[ii],L[ii],Lz[ii],
                                            vr[ii],vtheta[ii],phi[ii],
                                            fixed_quad,**kwargs))
            Op= nu.array(Op)
            Oz= copy.copy(Op)
            Op[vT < 0.]*= -1.
            ap= copy.copy(asc)
            ar= nu.array(ar)
            az= nu.array(az)
            ap[vT < 0.]-= az[vT < 0.]
            ap[vT >= 0.]+= az[vT >= 0.]
            ar= ar % (2.*nu.pi)
            ap= ap % (2.*nu.pi)
            az= az % (2.*nu.pi)
            return (nu.array(Jr),Jphi,Jz,nu.array(Or),Op,Oz,
                    ar,ap,az)
    
    def _EccZmaxRperiRap(self,*args,**kwargs):
        """
        NAME:
           EccZmaxRperiRap (_EccZmaxRperiRap)
        PURPOSE:
           evaluate the eccentricity, maximum height above the plane, peri- and apocenter for a spherical potential
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
           (e,zmax,rperi,rap)
        HISTORY:
           2017-12-22 - Written - Bovy (UofT)
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
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
        if self._c: #pragma: no cover
            pass
        else:
            r= nu.sqrt(R**2.+z**2.)
            vr= (R*vR+z*vz)/r
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            L= nu.sqrt(L2)
            E= _evaluateplanarPotentials(self._2dpot,r)\
                +vR**2./2.+vT**2./2.+vz**2./2.
            vt= L/r
            rperi, rap= [], []
            for ii in range(len(r)):
                trperi,trap= self._calc_rperi_rap(r[ii],vr[ii],vt[ii],
                                                  E[ii],L[ii])
                rperi.append(trperi)
                rap.append(trap)
            rperi= nu.array(rperi)
            rap= nu.array(rap)
            return ((rap-rperi)/(rap+rperi),rap*nu.sqrt(1.-Lz**2./L2),
                    rperi,rap)
        
    def _calc_rperi_rap(self,r,vr,vt,E,L):
        if vr == 0. \
            and m.fabs(vt-vcirc(self._2dpot,r,use_physical=False)) < _EPS:
            #We are on a circular orbit
            rperi= r
            rap = r
        elif vr == 0. and vt > vcirc(self._2dpot,r,use_physical=False):
            #We are exactly at pericenter
            rperi= r
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(r+10.**-8.,E,L,self._2dpot)
                startsign/= m.fabs(startsign)
            else: startsign= 1.
            rend= _rapRperiAxiFindStart(r,E,L,self._2dpot,rap=True,
                                        startsign=startsign)
            rap= optimize.brentq(_rapRperiAxiEq,rperi+0.00001,rend,
                                 args=(E,L,self._2dpot))
        elif vr == 0. and vt < vcirc(self._2dpot,r,use_physical=False):
            #We are exactly at apocenter
            rap= r
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(r-10.**-8.,E,L,self._2dpot)
                startsign/= m.fabs(startsign)
            else: startsign= 1.
            rstart= _rapRperiAxiFindStart(r,E,L,self._2dpot,startsign=startsign)
            if rstart == 0.: rperi= 0.
            else:
                rperi= optimize.brentq(_rapRperiAxiEq,rstart,rap-0.000001,
                                       args=(E,L,self._2dpot))
        else:
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(r,E,L,self._2dpot)
                startsign/= m.fabs(startsign)
            else:
                startsign= 1.
            rstart= _rapRperiAxiFindStart(r,E,L,self._2dpot,startsign=startsign)
            if rstart == 0.: rperi= 0.
            else: 
                try:
                    rperi= optimize.brentq(_rapRperiAxiEq,rstart,r,
                                           (E,L,self._2dpot),
                                           maxiter=200)
                except RuntimeError: #pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
            rend= _rapRperiAxiFindStart(r,E,L,self._2dpot,rap=True,
                                        startsign=startsign)
            rap= optimize.brentq(_rapRperiAxiEq,r,rend,
                                 (E,L,self._2dpot))
        return (rperi,rap)

    def _calc_jr(self,rperi,rap,E,L,fixed_quad,**kwargs):
        if fixed_quad:
            return integrate.fixed_quad(_JrSphericalIntegrand,
                                        rperi,rap,
                                        args=(E,L,self._2dpot),
                                        n=10,
                                        **kwargs)[0]/nu.pi
        else:
            return (nu.array(integrate.quad(_JrSphericalIntegrand,
                                            rperi,rap,
                                            args=(E,L,self._2dpot),
                                            **kwargs)))[0]/nu.pi
    def _calc_or(self,Rmean,rperi,rap,E,L,fixed_quad,**kwargs):
        Tr= 0.
        if Rmean > rperi and not fixed_quad:
            Tr+= nu.array(integrate.quadrature(_TrSphericalIntegrandSmall,
                                               0.,m.sqrt(Rmean-rperi),
                                               args=(E,L,self._2dpot,
                                                     rperi),
                                               **kwargs))[0]
        elif Rmean > rperi and fixed_quad:
            Tr+= integrate.fixed_quad(_TrSphericalIntegrandSmall,
                                      0.,m.sqrt(Rmean-rperi),
                                      args=(E,L,self._2dpot,
                                            rperi),
                                      n=10,**kwargs)[0]
        if Rmean < rap and not fixed_quad:
            Tr+= nu.array(integrate.quadrature(_TrSphericalIntegrandLarge,
                                               0.,m.sqrt(rap-Rmean),
                                               args=(E,L,self._2dpot,
                                                     rap),
                                               **kwargs))[0]
        elif Rmean < rap and fixed_quad:
            Tr+= integrate.fixed_quad(_TrSphericalIntegrandLarge,
                                      0.,m.sqrt(rap-Rmean),
                                      args=(E,L,self._2dpot,
                                            rap),
                                      n=10,**kwargs)[0]
        Tr= 2.*Tr
        return 2.*nu.pi/Tr

    def _calc_op(self,Or,Rmean,rperi,rap,E,L,fixed_quad,**kwargs):
        #Azimuthal period
        I= 0.
        if Rmean > rperi and not fixed_quad:
            I+= nu.array(integrate.quadrature(_ISphericalIntegrandSmall,
                                              0.,m.sqrt(Rmean-rperi),
                                              args=(E,L,self._2dpot,
                                                    rperi),
                                              **kwargs))[0]
        elif Rmean > rperi and fixed_quad:
            I+= integrate.fixed_quad(_ISphericalIntegrandSmall,
                                     0.,m.sqrt(Rmean-rperi),
                                     args=(E,L,self._2dpot,rperi),
                                     n=10,**kwargs)[0]
        if Rmean < rap and not fixed_quad:
            I+= nu.array(integrate.quadrature(_ISphericalIntegrandLarge,
                                              0.,m.sqrt(rap-Rmean),
                                              args=(E,L,self._2dpot,
                                                    rap),
                                              **kwargs))[0]
        elif Rmean < rap and fixed_quad:
            I+= integrate.fixed_quad(_ISphericalIntegrandLarge,
                                     0.,m.sqrt(rap-Rmean),
                                     args=(E,L,self._2dpot,rap),
                                     n=10,**kwargs)[0]
        I*= 2*L
        return I*Or/2./nu.pi

    def _calc_long_asc(self,z,R,vtheta,phi,Lz,L):
        i= nu.arccos(Lz/L)
        sinu= z/R/nu.tan(i)
        pindx= (sinu > 1.)*(sinu < (1.+10.**-7.))
        sinu[pindx]= 1.
        pindx= (sinu < -1.)*(sinu > (-1.-10.**-7.))
        sinu[pindx]= -1.           
        u= nu.arcsin(sinu)
        vzindx= vtheta > 0.
        u[vzindx]= nu.pi-u[vzindx]
        return phi-u
    
    def _calc_angler(self,Or,r,Rmean,rperi,rap,E,L,vr,fixed_quad,**kwargs):
        if r < Rmean:
            if r > rperi and not fixed_quad:
                wr= Or*integrate.quadrature(_TrSphericalIntegrandSmall,
                                            0.,m.sqrt(r-rperi),
                                            args=(E,L,self._2dpot,rperi),
                                            **kwargs)[0]
            elif r > rperi and fixed_quad:
                wr= Or*integrate.fixed_quad(_TrSphericalIntegrandSmall,
                                            0.,m.sqrt(r-rperi),
                                            args=(E,L,self._2dpot,rperi),
                                            n=10,**kwargs)[0]
            else:
                wr= 0.
            if vr < 0.: wr= 2*m.pi-wr
        else:
            if r < rap and not fixed_quad:
                wr= Or*integrate.quadrature(_TrSphericalIntegrandLarge,
                                            0.,m.sqrt(rap-r),
                                            args=(E,L,self._2dpot,rap),
                                            **kwargs)[0]
            elif r < rap and fixed_quad:
                wr= Or*integrate.fixed_quad(_TrSphericalIntegrandLarge,
                                            0.,m.sqrt(rap-r),
                                            args=(E,L,self._2dpot,rap),
                                            n=10,**kwargs)[0]
            else:
                wr= m.pi
            if vr < 0.:
                wr= m.pi+wr
            else:
                wr= m.pi-wr
        return wr
        
    def _calc_anglez(self,Or,Op,ar,z,r,Rmean,rperi,rap,E,L,Lz,vr,vtheta,phi,
                     fixed_quad,**kwargs):
        #First calculate psi
        i= nu.arccos(Lz/L)
        sinpsi= z/r/nu.sin(i)
        if sinpsi > 1. and sinpsi < (1.+10.**-7.):
            sinpsi= 1.
        if sinpsi < -1. and sinpsi > (-1.-10.**-7.):
            sinpsi= -1.
        psi= nu.arcsin(sinpsi)
        if vtheta > 0.: psi= nu.pi-psi
        psi= psi % (2.*nu.pi)
        #Calculate dSr/dL
        dpsi= Op/Or*2.*nu.pi #this is the full I integral
        if r < Rmean:
            if not fixed_quad:
                wz= L*integrate.quadrature(_ISphericalIntegrandSmall,
                                           0.,m.sqrt(r-rperi),
                                           args=(E,L,self._2dpot,
                                                 rperi),
                                           **kwargs)[0]
            elif fixed_quad:
                wz= L*integrate.fixed_quad(_ISphericalIntegrandSmall,
                                           0.,m.sqrt(r-rperi),
                                           args=(E,L,self._2dpot,
                                                 rperi),
                                           n=10,**kwargs)[0]
            if vr < 0.: wz= dpsi-wz
        else:
            if not fixed_quad:
                wz= L*integrate.quadrature(_ISphericalIntegrandLarge,
                                           0.,m.sqrt(rap-r),
                                           args=(E,L,self._2dpot,
                                                 rap),
                                           **kwargs)[0]
            elif fixed_quad:
                wz= L*integrate.fixed_quad(_ISphericalIntegrandLarge,
                                           0.,m.sqrt(rap-r),
                                           args=(E,L,self._2dpot,
                                                 rap),
                                           n=10,**kwargs)[0]
            if vr < 0.:
                wz= dpsi/2.+wz
            else:
                wz= dpsi/2.-wz
        #Add everything
        wz= -wz+psi+Op/Or*ar
        return wz

def _JrSphericalIntegrand(r,E,L,pot):
    """The J_r integrand"""
    return nu.sqrt(2.*(E-_evaluateplanarPotentials(pot,r))-L**2./r**2.)

def _TrSphericalIntegrandSmall(t,E,L,pot,rperi):
    r= rperi+t**2.#part of the transformation
    return 2.*t/_JrSphericalIntegrand(r,E,L,pot)

def _TrSphericalIntegrandLarge(t,E,L,pot,rap):
    r= rap-t**2.#part of the transformation
    return 2.*t/_JrSphericalIntegrand(r,E,L,pot)

def _ISphericalIntegrandSmall(t,E,L,pot,rperi):
    r= rperi+t**2.#part of the transformation
    return 2.*t/_JrSphericalIntegrand(r,E,L,pot)/r**2.

def _ISphericalIntegrandLarge(t,E,L,pot,rap):
    r= rap-t**2.#part of the transformation
    return 2.*t/_JrSphericalIntegrand(r,E,L,pot)/r**2.

def _rapRperiAxiEq(R,E,L,pot):
    """The vr=0 equation that needs to be solved to find apo- and pericenter"""
    return E-_evaluateplanarPotentials(pot,R)-L**2./2./R**2.

def _rapRperiAxiFindStart(R,E,L,pot,rap=False,startsign=1.):
    """
    NAME:
       _rapRperiAxiFindStart
    PURPOSE:
       Find adequate start or end points to solve for rap and rperi
    INPUT:
       R - Galactocentric radius
       E - energy
       L - angular momentum
       pot - potential
       rap - if True, find the rap end-point
       startsign= set to -1 if the function is not positive (due to gamma in the modified adiabatic approximation)
    OUTPUT:
       rstart or rend
    HISTORY:
       2010-12-01 - Written - Bovy (NYU)
    """
    if rap:
        rtry= 2.*R
    else:
        rtry= R/2.
    while startsign*_rapRperiAxiEq(rtry,E,L,pot) > 0. \
            and rtry > 0.000000001:
        if rap:
            if rtry > 100.: #pragma: no cover
                raise UnboundError("Orbit seems to be unbound")
            rtry*= 2.
        else:
            rtry/= 2.
    if rtry < 0.000000001: return 0.
    return rtry

