#include <math.h>
#include "galpy_potentials.h"
//CoreDehnenPotential
//2 arguments: amp, a
double DehnenCoreSphericalPotentialEval(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return -amp * (1. - pow(sqrtRz/(sqrtRz+a), 2.)) / (6. * a);
}

double DehnenCoreSphericalPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (R / pow(a+sqrtRz,3.) / 3.);
}

double DehnenCoreSphericalPotentialPlanarRforce(double R,double phi,
					    double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  return - amp * (R / pow(a+R, 3.) / 3.);
}

double DehnenCoreSphericalPotentialzforce(double R,double Z,double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (Z / pow(a+sqrtRz, 3.) / 3.);
}

double DehnenCoreSphericalPotentialPlanarR2deriv(double R,double phi,
					     double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  // return
  return -amp * (pow(a+R, -4.) * (2.*R - a)) / 3.;
}
double DehnenCoreSphericalPotentialR2deriv(double R,double Z, double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r= sqrt( R * R + Z * Z );
  double Phipp= DehnenCoreSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -DehnenCoreSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R / r / r + Phip * Z * Z / r / r / r;
}
double DehnenCoreSphericalPotentialz2deriv(double R,double Z, double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double r= sqrt( R * R + Z * Z );
  double Phipp= DehnenCoreSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -DehnenCoreSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z / r / r + Phip * R * R / r / r / r;
}
double DehnenCoreSphericalPotentialRzderiv(double R,double Z, double phi,
					   double t,
					   struct potentialArg * potentialArgs){
  double r= sqrt( R * R + Z * Z );
  double Phipp= DehnenCoreSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -DehnenCoreSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp / r / r - Phip / r / r / r );
}
double DehnenCoreSphericalPotentialDens(double R,double Z, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  //Calculate Rforce
  double r= sqrt ( R * R + Z * Z );
  return amp * M_1_PI / 4. * pow ( 1. + r / a, -4.) * pow (a, - 3.);
}
