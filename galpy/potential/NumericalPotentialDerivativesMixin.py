###############################################################################
#   NumericalPotentialDerivativesMixin: helper class to add numerical derivs
###############################################################################
class NumericalPotentialDerivativesMixin(object):
    def __init__(self,*args,**kwargs):
        self._dR= kwargs.get('dR',1e-8)
        self._dphi= kwargs.get('dphi',1e-8)
        self._dz= kwargs.get('dz',1e-8)
    
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
