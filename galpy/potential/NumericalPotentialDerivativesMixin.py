from funcsigs import signature  # py2 backport of inspect.signature

###############################################################################
#   NumericalPotentialDerivativesMixin: helper class to add numerical derivs
###############################################################################
class NumericalPotentialDerivativesMixin(object):
    """Mixin to add numerical derivatives to a Potential class, use as, e.g.,

    .. highlight:: python
    .. code-block:: python

        class PotWithNumericalDerivs(Potential,NumericalPotentialDerivativesMixin):
            def __init__(self,*args,**kwargs):
                NumericalPotentialDerivativesMixin.__init__(self,kwargs) # *not* **kwargs!
                # Remainder of initialization
                ...

            def _evaluate(self,R,z,phi=0.,t=0.):
                # Evaluate the potential

            # All forces and second derivatives then computed by NumericalPotentialDerivativesMixin

    to add numerical derivatives to a new potential class ``PotWithNumericalDerivs`` that only implements the potential itself, but not the forces. The class may implement any of the forces or second derivatives, all non-implemented forces/second-derivatives will be computed numerically by adding this Mixin

    The step used to compute the first (force) and second derivatives can be controlled at object instantiation by the keyword arguments ``dR``, ``dz``, ``dphi`` (for the forces; 1e-8 default) and ``dR2``, ``dz2``, and ``dphi2`` (for the second derivaives; 1e-4 default)"""
    def __new__(cls, *args, **kwargs):
        self = object.__new__(cls)  # a clean instance of cls
        # signature
        sig = signature(cls.__init__)
        # don't include self
        params = list(sig.parameters.values())[1:]
        sig = sig.replace(parameters=params)
        # correcting defaults for passed values
        params = list(sig.parameters.values())
        for i, param in enumerate(params):
            if i < len(args):  # get arguments
                params[i] = param.replace(default=args[i])
            elif param.name in kwargs.keys():  # get kwargs
                params[i] = param.replace(default=kwargs[param.name])
        sig = sig.replace(parameters=params)  # apply to signature
        self._init_args = sig
        return self  # send to init

    def __getnewargs__(self):
        return (NumericalPotentialDerivativesMixin.__str__(self),)

    def __init__(self,kwargs): # no **kwargs to get a reference, not a copy!
        # For first derivatives
        self._dR= kwargs.pop('dR',1e-8)
        self._dphi= kwargs.pop('dphi',1e-8)
        self._dz= kwargs.pop('dz',1e-8)
        # For second derivatives
        self._dR2= kwargs.pop('dR2',1e-4)
        self._dphi2= kwargs.pop('dphi2',1e-4)
        self._dz2= kwargs.pop('dz2',1e-4)

    @property
    def init_args(self):
        """Arguments used to initialize Potential.
        make bound argument for easy application
        ba = sig.bind(**{n: p.default for n, p in sig.parameters.items()})
        storing
        allows any potential to be reconstructed
        if want to use, do
        >>> potential(*potential.init_args.args, **potential.init_args.kwargs)
        """
        # enforce shallow copy
        sig = self._init_args
        sig = sig.replace(parameters=list(sig.parameters.values()))
        # return a bound argument
        return sig.bind(**{n: p.default for n, p in sig.parameters.items()})

    def copy(self):
        """
        NAME:

           copy

        PURPOSE:

           make a copy of this potential

        INPUT:

           (none)

        OUTPUT:

           copy of current potential

        HISTORY:

           2019-11-07 - Written - Starkman (UofT)

        """
        # ba = self._init_args.bind(self.__init__)
        return self.__class__(*self.init_args.args, **self.init_args.kwargs) 
    
    def _Rforce(self,R,z,phi=0.,t=0.):
        # Do forward difference because R cannot be negative
        RplusdR= R+self._dR
        Rplus2dR= R+2.*self._dR
        dR= (Rplus2dR-R)/2.
        return (1.5*self._evaluate(R,z,phi=phi,t=t)
                   -2.*self._evaluate(RplusdR,z,phi=phi,t=t)
                   +0.5*self._evaluate(Rplus2dR,z,phi=phi,t=t))/dR

    def _zforce(self,R,z,phi=0.,t=0.):
        # Central difference to get derivative at z=0 right
        zplusdz= z+self._dz
        zminusdz= z-self._dz
        dz= zplusdz-zminusdz
        return (self._evaluate(R,zminusdz,phi=phi,t=t)
                   -self._evaluate(R,zplusdz,phi=phi,t=t))/dz

    def _phiforce(self,R,z,phi=0.,t=0.):
        if not self.isNonAxi: return 0.
        # Central difference
        phiplusdphi= phi+self._dphi
        phiminusdphi= phi-self._dphi
        dphi= phiplusdphi-phiminusdphi
        return (self._evaluate(R,z,phi=phiminusdphi,t=t)
                   -self._evaluate(R,z,phi=phiplusdphi,t=t))/dphi

    def _R2deriv(self,R,z,phi=0.,t=0.):
        # Do forward difference because R cannot be negative
        RplusdR= R+self._dR2
        Rplus2dR= R+2.*self._dR2
        Rplus3dR= R+3.*self._dR2
        dR= (Rplus3dR-R)/3.
        return (2.*self._evaluate(R,z,phi=phi,t=t)
                -5.*self._evaluate(RplusdR,z,phi=phi,t=t)
                +4.*self._evaluate(Rplus2dR,z,phi=phi,t=t)
                -1.*self._evaluate(Rplus3dR,z,phi=phi,t=t))/dR**2.

    def _z2deriv(self,R,z,phi=0.,t=0.):
        # Central derivative
        zplusdz= z+self._dz2
        zminusdz= z-self._dz2
        dz= (zplusdz-zminusdz)/2.
        return (self._evaluate(R,zplusdz,phi=phi,t=t)
                   +self._evaluate(R,zminusdz,phi=phi,t=t)
                   -2.*self._evaluate(R,z,phi=phi,t=t))/dz**2.

    def _phi2deriv(self,R,z,phi=0.,t=0.):
        if not self.isNonAxi: return 0.
        # Central derivative
        phiplusdphi= phi+self._dphi2
        phiminusdphi= phi-self._dphi2
        dphi= (phiplusdphi-phiminusdphi)/2.
        return (self._evaluate(R,z,phi=phiplusdphi,t=t)
                   +self._evaluate(R,z,phi=phiminusdphi,t=t)
                   -2.*self._evaluate(R,z,phi=phi,t=t))/dphi**2.
       
    def _Rzderiv(self,R,z,phi=0.,t=0.):
        # Do forward difference in R because R cannot be negative
        RplusdR= R+self._dR2
        Rplus2dR= R+2.*self._dR2
        dR= (Rplus2dR-R)/2.
        zplusdz= z+self._dz2
        zminusdz= z-self._dz2
        dz= zplusdz-zminusdz
        return (-1.5*self._evaluate(R,zplusdz,phi=phi,t=t)
                   +2.*self._evaluate(RplusdR,zplusdz,phi=phi,t=t)
                   -0.5*self._evaluate(Rplus2dR,zplusdz,phi=phi,t=t)
                   +1.5*self._evaluate(R,zminusdz,phi=phi,t=t)
                   -2.*self._evaluate(RplusdR,zminusdz,phi=phi,t=t)
                   +0.5*self._evaluate(Rplus2dR,zminusdz,phi=phi,t=t))/dR/dz

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        if not self.isNonAxi: return 0.
        # Do forward difference in R because R cannot be negative
        RplusdR= R+self._dR2
        Rplus2dR= R+2.*self._dR2
        dR= (Rplus2dR-R)/2.
        phiplusdphi= phi+self._dphi2
        phiminusdphi= phi-self._dphi2
        dphi= phiplusdphi-phiminusdphi
        return (-1.5*self._evaluate(R,z,phi=phiplusdphi,t=t)
                   +2.*self._evaluate(RplusdR,z,phi=phiplusdphi,t=t)
                   -0.5*self._evaluate(Rplus2dR,z,phi=phiplusdphi,t=t)
                   +1.5*self._evaluate(R,z,phi=phiminusdphi,t=t)
                   -2.*self._evaluate(RplusdR,z,phi=phiminusdphi,t=t)
                   +0.5*self._evaluate(Rplus2dR,z,phi=phiminusdphi,t=t))\
                   /dR/dphi

