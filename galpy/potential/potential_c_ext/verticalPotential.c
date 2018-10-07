#include <galpy_potentials.h>
//Routines to evaluate forces etc. for a verticalPotential, i.e., a 1D 
//potential derived from a 3D potential as the z potential at a given (R,phi)
double verticalPotentialLinearForce(double x, double t,
				    struct potentialArg * potentialArgs){
  double R= *potentialArgs->args;
  return calczforce(R,x,0.,t,
		    potentialArgs->nwrapped,
		    potentialArgs->wrappedPotentialArg);
}
