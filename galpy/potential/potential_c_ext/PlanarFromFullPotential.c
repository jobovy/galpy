#include <galpy_potentials.h>
// PlanarFromFullPotential: the planar version of a 3D potential that has no
// planar C implementation of its own (e.g., RotateAndTiltWrapperPotential or
// a non-axisymmetric MovingObjectPotential): the 3D potential's forces and
// second derivatives at z = 0, like planarPotentialFromFullPotential in Python.
// No arguments; the wrapped potentials are parsed as 3D potentials.
double PlanarFromFullPotentialPlanarRforce(double R,double phi,double t,
					   struct potentialArg * potentialArgs){
  return calcRforce(R,0.,phi,t,potentialArgs->nwrapped,
		    potentialArgs->wrappedPotentialArg);
}
double PlanarFromFullPotentialPlanarphitorque(double R,double phi,double t,
					      struct potentialArg * potentialArgs){
  return calcphitorque(R,0.,phi,t,potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
}
double PlanarFromFullPotentialPlanarR2deriv(double R,double phi,double t,
					    struct potentialArg * potentialArgs){
  return calcR2deriv(R,0.,phi,t,potentialArgs->nwrapped,
		     potentialArgs->wrappedPotentialArg);
}
double PlanarFromFullPotentialPlanarphi2deriv(double R,double phi,double t,
					      struct potentialArg * potentialArgs){
  return calcphi2deriv(R,0.,phi,t,potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
}
double PlanarFromFullPotentialPlanarRphideriv(double R,double phi,double t,
					      struct potentialArg * potentialArgs){
  return calcRphideriv(R,0.,phi,t,potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
}
