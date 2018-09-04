import warnings
from . import Potential
from . import planarPotential
from . import linearPotential
from . import verticalPotential
from . import MiyamotoNagaiPotential
from . import IsochronePotential
from . import LogarithmicHaloPotential
from . import DoubleExponentialDiskPotential
from . import PowerSphericalPotential
from . import PowerSphericalPotentialwCutoff
from . import TwoPowerSphericalPotential
from . import plotRotcurve
from . import plotEscapecurve
from . import KGPotential
from . import interpRZPotential
from . import DehnenBarPotential
from . import SteadyLogSpiralPotential
from . import TransientLogSpiralPotential
from . import MovingObjectPotential
from . import ForceSoftening
from . import EllipticalDiskPotential
from . import CosmphiDiskPotential
from . import RazorThinExponentialDiskPotential
from . import FlattenedPowerPotential
from . import SnapshotRZPotential
from . import BurkertPotential
from . import MN3ExponentialDiskPotential
from . import KuzminKutuzovStaeckelPotential
from . import PlummerPotential
from . import PseudoIsothermalPotential
from . import KuzminDiskPotential
from . import TwoPowerTriaxialPotential
from . import FerrersPotential
from . import SCFPotential
from . import SoftenedNeedleBarPotential
from . import DiskSCFPotential
from . import SpiralArmsPotential
from . import HenonHeilesPotential
from . import DehnenSmoothWrapperPotential
from . import SolidBodyRotationWrapperPotential
from . import CorotatingRotationWrapperPotential
from . import GaussianAmplitudeWrapperPotential
from . import ChandrasekharDynamicalFrictionForce
from . import SphericalShellPotential
from . import RingPotential
from . import PerfectEllipsoidPotential
#
# Functions
#
evaluatePotentials= Potential.evaluatePotentials
evaluateDensities= Potential.evaluateDensities
evaluateSurfaceDensities= Potential.evaluateSurfaceDensities
evaluateRforces= Potential.evaluateRforces
evaluatephiforces= Potential.evaluatephiforces
evaluatezforces= Potential.evaluatezforces
evaluaterforces= Potential.evaluaterforces
evaluateR2derivs= Potential.evaluateR2derivs
evaluatez2derivs= Potential.evaluatez2derivs
evaluateRzderivs= Potential.evaluateRzderivs
evaluatephi2derivs= Potential.evaluatephi2derivs
evaluateRphiderivs= Potential.evaluateRphiderivs
evaluater2derivs= Potential.evaluater2derivs
RZToplanarPotential= planarPotential.RZToplanarPotential
toPlanarPotential= planarPotential.toPlanarPotential
RZToverticalPotential= verticalPotential.RZToverticalPotential
plotPotentials= Potential.plotPotentials
plotDensities= Potential.plotDensities
plotplanarPotentials= planarPotential.plotplanarPotentials
plotlinearPotentials= linearPotential.plotlinearPotentials
calcRotcurve= plotRotcurve.calcRotcurve
vcirc= plotRotcurve.vcirc
dvcircdR= plotRotcurve.dvcircdR
epifreq= Potential.epifreq
verticalfreq= Potential.verticalfreq
flattening= Potential.flattening
rl= Potential.rl
omegac= Potential.omegac
vterm= Potential.vterm
lindbladR= Potential.lindbladR
plotRotcurve= plotRotcurve.plotRotcurve
calcEscapecurve= plotEscapecurve.calcEscapecurve
vesc= plotEscapecurve.vesc
plotEscapecurve= plotEscapecurve.plotEscapecurve
evaluateplanarPotentials= planarPotential.evaluateplanarPotentials
evaluateplanarRforces= planarPotential.evaluateplanarRforces
evaluateplanarR2derivs= planarPotential.evaluateplanarR2derivs
evaluateplanarphiforces= planarPotential.evaluateplanarphiforces
evaluatelinearPotentials= linearPotential.evaluatelinearPotentials
evaluatelinearForces= linearPotential.evaluatelinearForces
PotentialError= Potential.PotentialError
LinShuReductionFactor= planarPotential.LinShuReductionFactor
nemo_accname= Potential.nemo_accname
nemo_accpars= Potential.nemo_accpars
turn_physical_off= Potential.turn_physical_off
turn_physical_on= Potential.turn_physical_on
_dim= Potential._dim
_isNonAxi= Potential._isNonAxi
scf_compute_coeffs_spherical = SCFPotential.scf_compute_coeffs_spherical
scf_compute_coeffs_axi = SCFPotential.scf_compute_coeffs_axi
scf_compute_coeffs = SCFPotential.scf_compute_coeffs
rtide= Potential.rtide
ttensor= Potential.ttensor
flatten= Potential.flatten
#
# Classes
#
Potential= Potential.Potential
planarAxiPotential= planarPotential.planarAxiPotential
planarPotential= planarPotential.planarPotential
linearPotential= linearPotential.linearPotential
MiyamotoNagaiPotential= MiyamotoNagaiPotential.MiyamotoNagaiPotential
IsochronePotential= IsochronePotential.IsochronePotential
DoubleExponentialDiskPotential= DoubleExponentialDiskPotential.DoubleExponentialDiskPotential
LogarithmicHaloPotential= LogarithmicHaloPotential.LogarithmicHaloPotential
KeplerPotential= PowerSphericalPotential.KeplerPotential
PowerSphericalPotential= PowerSphericalPotential.PowerSphericalPotential
PowerSphericalPotentialwCutoff= PowerSphericalPotentialwCutoff.PowerSphericalPotentialwCutoff
NFWPotential= TwoPowerSphericalPotential.NFWPotential
JaffePotential= TwoPowerSphericalPotential.JaffePotential
HernquistPotential= TwoPowerSphericalPotential.HernquistPotential
TwoPowerSphericalPotential= TwoPowerSphericalPotential.TwoPowerSphericalPotential
KGPotential= KGPotential.KGPotential
interpRZPotential= interpRZPotential.interpRZPotential
DehnenBarPotential= DehnenBarPotential.DehnenBarPotential
SteadyLogSpiralPotential= SteadyLogSpiralPotential.SteadyLogSpiralPotential
TransientLogSpiralPotential= TransientLogSpiralPotential.TransientLogSpiralPotential
MovingObjectPotential= MovingObjectPotential.MovingObjectPotential
EllipticalDiskPotential= EllipticalDiskPotential.EllipticalDiskPotential
LopsidedDiskPotential= CosmphiDiskPotential.LopsidedDiskPotential
CosmphiDiskPotential= CosmphiDiskPotential.CosmphiDiskPotential
RazorThinExponentialDiskPotential= RazorThinExponentialDiskPotential.RazorThinExponentialDiskPotential
FlattenedPowerPotential= FlattenedPowerPotential.FlattenedPowerPotential
InterpSnapshotRZPotential = SnapshotRZPotential.InterpSnapshotRZPotential
SnapshotRZPotential = SnapshotRZPotential.SnapshotRZPotential
BurkertPotential= BurkertPotential.BurkertPotential
MN3ExponentialDiskPotential= MN3ExponentialDiskPotential.MN3ExponentialDiskPotential
KuzminKutuzovStaeckelPotential = KuzminKutuzovStaeckelPotential.KuzminKutuzovStaeckelPotential
PlummerPotential = PlummerPotential.PlummerPotential
PseudoIsothermalPotential = PseudoIsothermalPotential.PseudoIsothermalPotential
KuzminDiskPotential = KuzminDiskPotential.KuzminDiskPotential
TriaxialHernquistPotential= TwoPowerTriaxialPotential.TriaxialHernquistPotential
TriaxialNFWPotential= TwoPowerTriaxialPotential.TriaxialNFWPotential
TriaxialJaffePotential= TwoPowerTriaxialPotential.TriaxialJaffePotential
TwoPowerTriaxialPotential= TwoPowerTriaxialPotential.TwoPowerTriaxialPotential
FerrersPotential= FerrersPotential.FerrersPotential
SCFPotential = SCFPotential.SCFPotential
SoftenedNeedleBarPotential= SoftenedNeedleBarPotential.SoftenedNeedleBarPotential
DiskSCFPotential = DiskSCFPotential.DiskSCFPotential
SpiralArmsPotential = SpiralArmsPotential.SpiralArmsPotential
HenonHeilesPotential= HenonHeilesPotential.HenonHeilesPotential
ChandrasekharDynamicalFrictionForce= ChandrasekharDynamicalFrictionForce.ChandrasekharDynamicalFrictionForce
SphericalShellPotential= SphericalShellPotential.SphericalShellPotential
RingPotential= RingPotential.RingPotential
PerfectEllipsoidPotential= PerfectEllipsoidPotential.PerfectEllipsoidPotential
#Wrappers
DehnenSmoothWrapperPotential= DehnenSmoothWrapperPotential.DehnenSmoothWrapperPotential
SolidBodyRotationWrapperPotential= SolidBodyRotationWrapperPotential.SolidBodyRotationWrapperPotential
CorotatingRotationWrapperPotential= CorotatingRotationWrapperPotential.CorotatingRotationWrapperPotential
GaussianAmplitudeWrapperPotential= GaussianAmplitudeWrapperPotential.GaussianAmplitudeWrapperPotential
#Softenings
PlummerSoftening= ForceSoftening.PlummerSoftening

#
# Constants
#
MWPotential= [MiyamotoNagaiPotential(a=0.5,b=0.0375,normalize=.6),
              NFWPotential(a=4.5,normalize=.35),
              HernquistPotential(a=0.6/8,normalize=0.05)]
# See Table 1 in galpy paper: Bovy (2014)
MWPotential2014= [PowerSphericalPotentialwCutoff(normalize=0.05,alpha=1.8,rc=1.9/8.),
                  MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=0.6),
                  NFWPotential(a=2.,normalize=0.35)]
