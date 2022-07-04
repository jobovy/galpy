#include <math.h>
#include <galpy_potentials.h>
//General routines for SphericalPotentials
double SphericalPotentialEval(double R,double z,double phi,double t,
			      struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  //Calculate potential
  double r= sqrt(R*R+z*z);
  return amp * potentialArgs->revaluate(r,t,potentialArgs);
}
double SphericalPotentialRforce(double R,double z,double phi,double t,
				struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  //Calculate Rforce
  double r= sqrt(R*R+z*z);
  return amp * potentialArgs->rforce(r,t,potentialArgs)*R/r;
}
double SphericalPotentialPlanarRforce(double R,double phi,double t,
				      struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  //Calculate planar Rforce
  return amp * potentialArgs->rforce(R,t,potentialArgs);
}
double SphericalPotentialzforce(double R,double z,double phi,double t,
				struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  //Calculate zforce
  double r= sqrt(R*R+z*z);
  return amp * potentialArgs->rforce(r,t,potentialArgs)*z/r;
}
double SphericalPotentialPlanarR2deriv(double R,double phi,double t,
				       struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  //Calculate planar R2deriv
  return amp * potentialArgs->r2deriv(R,t,potentialArgs);
}
double SphericalPotentialDens(double R,double z,double phi,double t,
			      struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  //Calculate density through the Poisson equation
  double r= sqrt(R*R+z*z);
  /*
     Uncomment next few commented-out lines if you ever want to automatically
     use the Poisson equation to calculate the density rather than
     implement rdens
  */
  //  if ( potentialArgs->rdens )
  return amp * potentialArgs->rdens(r,t,potentialArgs);
  //  else
  //    return amp * M_1_PI / 4. * ( potentialArgs->r2deriv(r,t,potentialArgs)
  //			   - 2. * potentialArgs->rforce(r,t,potentialArgs)/r);
}
