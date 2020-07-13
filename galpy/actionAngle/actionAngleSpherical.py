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
import numpy
from scipy import integrate
from ..potential import epifreq, omegac, _dim
from ..potential.Potential import _evaluatePotentials
from ..potential.Potential import flatten as flatten_potential
from .actionAngle import actionAngle
from .actionAngleAxi import actionAngleAxi, potentialAxi
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

        OUTPUT:

           instance

        HISTORY:

           2013-12-28 - Written - Bovy (IAS)

        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleSpherical")
        self._pot= flatten_potential(kwargs['pot'])
        #Also store a 'planar' (2D) version of the potential
        if _dim(self._pot) == 2:
            self._2dpot= self._pot
        elif isinstance(self._pot,list):
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
           fixed_quad= (False) if True, use n=10 fixed_quad integration
           scipy.integrate.quadrature or .fixed_quad keywords
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
            R= numpy.array([R])
            vR= numpy.array([vR])
            vT= numpy.array([vT])
            z= numpy.array([z])
            vz= numpy.array([vz])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= _evaluatePotentials(self._pot,R,z)\
                +vR**2./2.+vT**2./2.+vz**2./2.
            L= numpy.sqrt(L2)
            #Actions
            Jphi= Lz
            Jz= L-numpy.fabs(Lz)
            #Jr requires some more work
            #Set up an actionAngleAxi object for EL and rap/rperi calculations
            axiR= numpy.sqrt(R**2.+z**2.)
            axivT= L/axiR
            axivR= (R*vR+z*vz)/axiR
            Jr= []
            for ii in range(len(axiR)):
                axiaA= actionAngleAxi(axiR[ii],axivR[ii],axivT[ii],
                                      pot=self._2dpot)
                (rperi,rap)= axiaA.calcRapRperi()
                EL= axiaA.calcEL()
                E, L= EL
                Jr.append(self._calc_jr(rperi,rap,E,L,fixed_quad,**kwargs))
            return (numpy.array(Jr),Jphi,Jz)

    def _actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           actionsFreqs (_actionsFreqs)
        PURPOSE:
           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
           fixed_quad= (False) if True, use n=10 fixed_quad integration
           scipy.integrate.quadrature or .fixed_quad keywords
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
            R= numpy.array([R])
            vR= numpy.array([vR])
            vT= numpy.array([vT])
            z= numpy.array([z])
            vz= numpy.array([vz])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= _evaluatePotentials(self._pot,R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= numpy.sqrt(L2)
            #Actions
            Jphi= Lz
            Jz= L-numpy.fabs(Lz)
            #Jr requires some more work
            #Set up an actionAngleAxi object for EL and rap/rperi calculations
            axiR= numpy.sqrt(R**2.+z**2.)
            axivT= L/axiR
            axivR= (R*vR+z*vz)/axiR
            Jr= []
            Or= []
            Op= []
            for ii in range(len(axiR)):
                axiaA= actionAngleAxi(axiR[ii],axivR[ii],axivT[ii],
                                      pot=self._2dpot)
                (rperi,rap)= axiaA.calcRapRperi()
                EL= axiaA.calcEL()
                E, L= EL
                Jr.append(self._calc_jr(rperi,rap,E,L,fixed_quad,**kwargs))
                #Radial period
                if Jr[-1] < 10.**-9.: #Circular orbit
                    Or.append(epifreq(self._pot,axiR[ii],use_physical=False))
                    Op.append(omegac(self._pot,axiR[ii],use_physical=False))
                    continue
                Rmean= numpy.exp((numpy.log(rperi)+numpy.log(rap))/2.)
                Or.append(self._calc_or(Rmean,rperi,rap,E,L,fixed_quad,**kwargs))
                Op.append(self._calc_op(Or[-1],Rmean,rperi,rap,E,L,fixed_quad,**kwargs))
            Op= numpy.array(Op)
            Oz= copy.copy(Op)
            Op[vT < 0.]*= -1.
            return (numpy.array(Jr),Jphi,Jz,numpy.array(Or),Op,Oz)
    
    def _actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           actionsFreqsAngles (_actionsFreqsAngles)
        PURPOSE:
           evaluate the actions, frequencies, and angles
           (jr,lz,jz,Omegar,Omegaphi,Omegaz,ar,ap,az)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
           fixed_quad= (False) if True, use n=10 fixed_quad integration
           scipy.integrate.quadrature or .fixed_quad keywords
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
            R= numpy.array([R])
            vR= numpy.array([vR])
            vT= numpy.array([vT])
            z= numpy.array([z])
            vz= numpy.array([vz])
            phi= numpy.array([phi])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= _evaluatePotentials(self._pot,R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= numpy.sqrt(L2)
            #Actions
            Jphi= Lz
            Jz= L-numpy.fabs(Lz)
            #Jr requires some more work
            #Set up an actionAngleAxi object for EL and rap/rperi calculations
            axiR= numpy.sqrt(R**2.+z**2.)
            axivT= L/axiR #these are really spherical coords, called axi bc they go in actionAngleAxi
            axivR= (R*vR+z*vz)/axiR
            axivz= (z*vR-R*vz)/axiR
            Jr= []
            Or= []
            Op= []
            ar= []
            az= []
            #Calculate the longitude of the ascending node
            asc= self._calc_long_asc(z,R,axivz,phi,Lz,L)
            for ii in range(len(axiR)):
                axiaA= actionAngleAxi(axiR[ii],axivR[ii],axivT[ii],
                                      pot=self._2dpot)
                (rperi,rap)= axiaA.calcRapRperi()
                EL= axiaA.calcEL()
                E, L= EL
                Jr.append(self._calc_jr(rperi,rap,E,L,fixed_quad,**kwargs))
                #Radial period
                Rmean= numpy.exp((numpy.log(rperi)+numpy.log(rap))/2.)
                if Jr[-1] < 10.**-9.: #Circular orbit
                    Or.append(epifreq(self._pot,axiR[ii],use_physical=False))
                    Op.append(omegac(self._pot,axiR[ii],use_physical=False))
                else:
                    Or.append(self._calc_or(Rmean,rperi,rap,E,L,fixed_quad,**kwargs))
                    Op.append(self._calc_op(Or[-1],Rmean,rperi,rap,E,L,fixed_quad,**kwargs))
                #Angles
                ar.append(self._calc_angler(Or[-1],axiR[ii],Rmean,rperi,rap,
                                            E,L,axivR[ii],fixed_quad,**kwargs))
                az.append(self._calc_anglez(Or[-1],Op[-1],ar[-1],
                                            z[ii],axiR[ii],
                                            Rmean,rperi,rap,E,L,Lz[ii],
                                            axivR[ii],axivz[ii],phi[ii],
                                            fixed_quad,**kwargs))
            Op= numpy.array(Op)
            Oz= copy.copy(Op)
            Op[vT < 0.]*= -1.
            ap= copy.copy(asc)
            ar= numpy.array(ar)
            az= numpy.array(az)
            ap[vT < 0.]-= az[vT < 0.]
            ap[vT >= 0.]+= az[vT >= 0.]
            ar= ar % (2.*numpy.pi)
            ap= ap % (2.*numpy.pi)
            az= az % (2.*numpy.pi)
            return (numpy.array(Jr),Jphi,Jz,numpy.array(Or),Op,Oz,
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
            R= numpy.array([R])
            vR= numpy.array([vR])
            vT= numpy.array([vT])
            z= numpy.array([z])
            vz= numpy.array([vz])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            L= numpy.sqrt(L2)
            #Set up an actionAngleAxi object for EL and rap/rperi calculations
            axiR= numpy.sqrt(R**2.+z**2.)
            axivT= L/axiR
            axivR= (R*vR+z*vz)/axiR
            rperi, rap= [], []
            for ii in range(len(axiR)):
                axiaA= actionAngleAxi(axiR[ii],axivR[ii],axivT[ii],
                                      pot=self._2dpot)
                trperi,trap= axiaA.calcRapRperi()
                rperi.append(trperi)
                rap.append(trap)
            rperi= numpy.array(rperi)
            rap= numpy.array(rap)
            return ((rap-rperi)/(rap+rperi),rap*numpy.sqrt(1.-Lz**2./L2),
                    rperi,rap)

    def _calc_jr(self,rperi,rap,E,L,fixed_quad,**kwargs):
        if fixed_quad:
            return integrate.fixed_quad(_JrSphericalIntegrand,
                                        rperi,rap,
                                        args=(E,L,self._2dpot),
                                        n=10,
                                        **kwargs)[0]/numpy.pi
        else:
            return (numpy.array(integrate.quad(_JrSphericalIntegrand,
                                            rperi,rap,
                                            args=(E,L,self._2dpot),
                                            **kwargs)))[0]/numpy.pi
    def _calc_or(self,Rmean,rperi,rap,E,L,fixed_quad,**kwargs):
        Tr= 0.
        if Rmean > rperi and not fixed_quad:
            Tr+= numpy.array(integrate.quadrature(_TrSphericalIntegrandSmall,
                                               0.,numpy.sqrt(Rmean-rperi),
                                               args=(E,L,self._2dpot,
                                                     rperi),
                                               **kwargs))[0]
        elif Rmean > rperi and fixed_quad:
            Tr+= integrate.fixed_quad(_TrSphericalIntegrandSmall,
                                      0.,numpy.sqrt(Rmean-rperi),
                                      args=(E,L,self._2dpot,
                                            rperi),
                                      n=10,**kwargs)[0]
        if Rmean < rap and not fixed_quad:
            Tr+= numpy.array(integrate.quadrature(_TrSphericalIntegrandLarge,
                                               0.,numpy.sqrt(rap-Rmean),
                                               args=(E,L,self._2dpot,
                                                     rap),
                                               **kwargs))[0]
        elif Rmean < rap and fixed_quad:
            Tr+= integrate.fixed_quad(_TrSphericalIntegrandLarge,
                                      0.,numpy.sqrt(rap-Rmean),
                                      args=(E,L,self._2dpot,
                                            rap),
                                      n=10,**kwargs)[0]
        Tr= 2.*Tr
        return 2.*numpy.pi/Tr

    def _calc_op(self,Or,Rmean,rperi,rap,E,L,fixed_quad,**kwargs):
        #Azimuthal period
        I= 0.
        if Rmean > rperi and not fixed_quad:
            I+= numpy.array(integrate.quadrature(_ISphericalIntegrandSmall,
                                              0.,numpy.sqrt(Rmean-rperi),
                                              args=(E,L,self._2dpot,
                                                    rperi),
                                              **kwargs))[0]
        elif Rmean > rperi and fixed_quad:
            I+= integrate.fixed_quad(_ISphericalIntegrandSmall,
                                     0.,numpy.sqrt(Rmean-rperi),
                                     args=(E,L,self._2dpot,rperi),
                                     n=10,**kwargs)[0]
        if Rmean < rap and not fixed_quad:
            I+= numpy.array(integrate.quadrature(_ISphericalIntegrandLarge,
                                              0.,numpy.sqrt(rap-Rmean),
                                              args=(E,L,self._2dpot,
                                                    rap),
                                              **kwargs))[0]
        elif Rmean < rap and fixed_quad:
            I+= integrate.fixed_quad(_ISphericalIntegrandLarge,
                                     0.,numpy.sqrt(rap-Rmean),
                                     args=(E,L,self._2dpot,rap),
                                     n=10,**kwargs)[0]
        I*= 2*L
        return I*Or/2./numpy.pi

    def _calc_long_asc(self,z,R,axivz,phi,Lz,L):
        i= numpy.arccos(Lz/L)
        sinu= z/R/numpy.tan(i)
        pindx= (sinu > 1.)*numpy.isfinite(sinu)
        sinu[pindx]= 1.
        pindx= (sinu < -1.)*numpy.isfinite(sinu)
        sinu[pindx]= -1.           
        u= numpy.arcsin(sinu)
        vzindx= axivz > 0.
        u[vzindx]= numpy.pi-u[vzindx]
        # For non-inclined orbits, we set Omega=0 by convention
        u[True^numpy.isfinite(u)]= phi[True^numpy.isfinite(u)]        
        return phi-u
    
    def _calc_angler(self,Or,r,Rmean,rperi,rap,E,L,vr,fixed_quad,**kwargs):
        if r < Rmean:
            if r > rperi and not fixed_quad:
                wr= Or*integrate.quadrature(_TrSphericalIntegrandSmall,
                                            0.,numpy.sqrt(r-rperi),
                                            args=(E,L,self._2dpot,rperi),
                                            **kwargs)[0]
            elif r > rperi and fixed_quad:
                wr= Or*integrate.fixed_quad(_TrSphericalIntegrandSmall,
                                            0.,numpy.sqrt(r-rperi),
                                            args=(E,L,self._2dpot,rperi),
                                            n=10,**kwargs)[0]
            else:
                wr= 0.
            if vr < 0.: wr= 2*numpy.pi-wr
        else:
            if r < rap and not fixed_quad:
                wr= Or*integrate.quadrature(_TrSphericalIntegrandLarge,
                                            0.,numpy.sqrt(rap-r),
                                            args=(E,L,self._2dpot,rap),
                                            **kwargs)[0]
            elif r < rap and fixed_quad:
                wr= Or*integrate.fixed_quad(_TrSphericalIntegrandLarge,
                                            0.,numpy.sqrt(rap-r),
                                            args=(E,L,self._2dpot,rap),
                                            n=10,**kwargs)[0]
            else:
                wr= numpy.pi
            if vr < 0.:
                wr= numpy.pi+wr
            else:
                wr= numpy.pi-wr
        return wr
        
    def _calc_anglez(self,Or,Op,ar,z,r,Rmean,rperi,rap,E,L,Lz,vr,axivz,phi,
                     fixed_quad,**kwargs):
        #First calculate psi
        i= numpy.arccos(Lz/L)
        sinpsi= z/r/numpy.sin(i)
        if numpy.isfinite(sinpsi):
            if sinpsi > 1.:
                sinpsi= 1.
            elif sinpsi < -1.:
                sinpsi= -1.
            psi= numpy.arcsin(sinpsi)
            if axivz > 0.: psi= numpy.pi-psi
        else:
            psi= phi
        psi= psi % (2.*numpy.pi)
        #Calculate dSr/dL
        dpsi= Op/Or*2.*numpy.pi #this is the full I integral
        if r < Rmean:
            if not fixed_quad:
                wz= L*integrate.quadrature(_ISphericalIntegrandSmall,
                                           0.,numpy.sqrt(r-rperi),
                                           args=(E,L,self._2dpot,
                                                 rperi),
                                           **kwargs)[0]
            elif fixed_quad:
                wz= L*integrate.fixed_quad(_ISphericalIntegrandSmall,
                                           0.,numpy.sqrt(r-rperi),
                                           args=(E,L,self._2dpot,
                                                 rperi),
                                           n=10,**kwargs)[0]
            if vr < 0.: wz= dpsi-wz
        else:
            if not fixed_quad:
                wz= L*integrate.quadrature(_ISphericalIntegrandLarge,
                                           0.,numpy.sqrt(rap-r),
                                           args=(E,L,self._2dpot,
                                                 rap),
                                           **kwargs)[0]
            elif fixed_quad:
                wz= L*integrate.fixed_quad(_ISphericalIntegrandLarge,
                                           0.,numpy.sqrt(rap-r),
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
    return numpy.sqrt(2.*(E-potentialAxi(r,pot))-L**2./r**2.)

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
