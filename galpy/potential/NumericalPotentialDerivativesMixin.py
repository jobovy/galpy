###############################################################################
#   NumericalPotentialDerivativesMixin: helper class to add numerical derivs
###############################################################################
class NumericalPotentialDerivativesMixin(object):
    """Mixin to add numerical derivatives to a Potential class, use as, e.g.,

    class myPotential(SCFPotential,NumericalPotentialDerivativesMixin):
        def __init__(self,*args,**kwargs):
            NumericalPotentialDerivativesMixin.__init__(self,kwargs) # *not* **kwargs!
            # Remainder of initialization
            ....

    to add numerical derivatives to the SCFPotential class (which at the time of writing does not have second derivatives, so adding the Mixin makes them numerical)"""
    def __init__(self,kwargs): # no **kwargs to get a reference, not a copy!
        # For first derivatives
        self._dR= kwargs.pop('dR',1e-8)
        self._dphi= kwargs.pop('dphi',1e-8)
        self._dz= kwargs.pop('dz',1e-8)
        # For second derivatives
        self._dR2= kwargs.pop('dR2',1e-4)
        self._dphi2= kwargs.pop('dphi2',1e-4)
        self._dz2= kwargs.pop('dz2',1e-4)
    
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

        
