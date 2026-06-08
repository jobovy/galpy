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
double SphericalPotentialR2deriv(double R,double Z,double phi,double t,
				 struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= SphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -SphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R * ir2 + Phip * Z * Z * ir3;
}
double SphericalPotentialz2deriv(double R,double Z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= SphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -SphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z * ir2 + Phip * R * R * ir3;
}
double SphericalPotentialRzderiv(double R,double Z,double phi,double t,
				 struct potentialArg * potentialArgs){
  double r2= R * R + Z * Z;
  double r= sqrt( r2 );
  double Phipp= SphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -SphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  double ir2= 1. / r2;
  double ir3= ir2 / r;
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp * ir2 - Phip * ir3 );
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
