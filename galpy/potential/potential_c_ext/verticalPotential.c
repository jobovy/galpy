#include <galpy_potentials.h>
//Routines to evaluate forces etc. for a verticalPotential, i.e., a 1D
//potential derived from a 3D potential as the z potential at a given (R,phi)
double verticalPotentialLinearForce(double x, double t,
				    struct potentialArg * potentialArgs){
  double R= *potentialArgs->args;
  double phi= *(potentialArgs->args+1);
  return calczforce(R,x,phi,t,
		    potentialArgs->nwrapped,
		    potentialArgs->wrappedPotentialArg);
}
// d^2 Phi / dx^2 = d^2 Phi / dz^2 of the wrapped 3D potential at (R,x,phi),
// mirroring verticalPotentialLinearForce (which uses the wrapped z-force).
double verticalPotentialLinear2deriv(double x, double t,
				     struct potentialArg * potentialArgs){
  double R= *potentialArgs->args;
  double phi= *(potentialArgs->args+1);
  return calcz2deriv(R,x,phi,t,
		     potentialArgs->nwrapped,
		     potentialArgs->wrappedPotentialArg);
}
