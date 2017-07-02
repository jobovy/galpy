###############################################################################
#   WrapperPotential.py: Super-class for wrapper potentials
###############################################################################
from galpy.potential_src.Potential import Potential, _isNonAxi
from galpy.potential_src.Potential import evaluatePotentials, \
    evaluateRforces, evaluatephiforces, evaluatezforces, \
    evaluaterforces, evaluateR2derivs, evaluatez2derivs, \
    evaluateRzderivs, evaluateDensities
class WrapperPotential(Potential):
    def __init__(self,amp=1.,pot=None,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a WrapperPotential, a super-class for wrapper potentials

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; the amplitude of this will be grown by this wrapper

        OUTPUT:

           (none)

        HISTORY:

           2017-06-26 - Started - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo)
        self._pot= pot
        self.isNonAxi= _isNonAxi(self._pot)

    def __getattr__(self,attribute):
        if attribute == '_evaluate' \
                or attribute == '_Rforce' or attribute == '_zforce' \
                or attribute == '_phiforce' \
                or attribute == '_R2deriv' or attribute == '_z2deriv' \
                or attribute == '_Rzderiv' or attribute == '_phi2deriv' \
                or attribute == '_Rphideriv' or attribute == '_dens':
            return lambda R,Z,phi=0.,t=0.: \
                self._wrap(attribute,R,Z,phi=phi,t=t)
        else:
            return super(WrapperPotential,self).__getattr__(attribute)

    def _wrap_pot_func(self,attribute):
        if attribute == '_evaluate':
            return evaluatePotentials
        elif attribute == '_dens':
            return evaluateDensities
        elif attribute == '_Rforce':
            return evaluateRforces
        elif attribute == '_zforce':
            return evaluatezforces
        elif attribute == '_phiforce':
            return evaluatephiforces
        elif attribute == '_R2deriv':
            return evaluateR2derivs
        elif attribute == '_z2deriv':
            return evaluatez2derivs
        elif attribute == '_Rzderiv':
            return evaluateRzderivs
        elif attribute == '_phi2deriv':
            return lambda p,R,Z,phi=0.,t=0.: \
                evaluatePotentials(p,R,Z,phi=phi,t=t,dphi=2)
        elif attribute == '_Rphideriv':
            return lambda p,R,Z,phi=0.,t=0.: \
                evaluatePotentials(p,R,Z,phi=phi,t=t,dR=1,dphi=1)
        else: #pragma: no cover
            raise AttributeError("Attribute %s not found in for this WrapperPotential" % attribute)
