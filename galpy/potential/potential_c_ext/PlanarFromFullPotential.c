#include <galpy_potentials.h>
// PlanarFromFullPotential: the planar version of a 3D force that has no
// planar C implementation of its own (e.g., RotateAndTiltWrapperPotential, a
// non-axisymmetric MovingObjectPotential, or ChandrasekharDynamicalFriction-
// Force): the 3D forces and second derivatives at z = 0 (and v_z = 0 for
// velocity-dependent forces), like planarPotentialFromFullPotential and
// planarDissipativeForceFromFullDissipativeForce in Python. No arguments; the
// wrapped forces are parsed as 3D ones.
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
double PlanarFromFullPotentialPlanarRforceVelocity(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs,
						   double vR,double vT){
  return calcRforce(R,0.,phi,t,potentialArgs->nwrapped,
		    potentialArgs->wrappedPotentialArg,1,vR,vT,0.);
}
double PlanarFromFullPotentialPlanarphitorqueVelocity(double R,double phi,
						      double t,
						      struct potentialArg * potentialArgs,
						      double vR,double vT){
  return calcphitorque(R,0.,phi,t,potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg,1,vR,vT,0.);
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
