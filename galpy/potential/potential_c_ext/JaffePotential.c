#include <math.h>
#include <galpy_potentials.h>
//JaffePotential
//2 arguments: amp, a
double JaffePotentialEval(double R,double Z, double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * log ( 1. + a / sqrtRz ) / a;
}
double JaffePotentialRforce(double R,double Z, double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * R * pow( sqrtRz , -3. ) / ( 1. + a / sqrtRz );
}
double JaffePotentialPlanarRforce(double R,double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  return - amp * pow(R,-2.) / ( 1. + a / R );
}
double JaffePotentialzforce(double R,double Z,double phi,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * Z * pow( sqrtRz , -3. ) / ( 1. + a / sqrtRz );
}
double JaffePotentialPlanarR2deriv(double R,double phi,
				   double t,
				   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate R2deriv
  return - amp * (a + 2. * R) * pow(R,-4.) * pow(1.+a/R,-2.);
}
double JaffePotentialR2deriv(double R,double Z, double phi,
			     double t,
			     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= pow(r,-2.) / ( 1. + a / r ); // Phi'/amp
  double d2phi= - (a + 2. * r) * pow(r,-4.) * pow(1.+a/r,-2.); // Phi''/amp
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return amp * ( d2phi * R * R / r / r + dphi * Z * Z / r / r / r );
}
double JaffePotentialz2deriv(double R,double Z, double phi,
			     double t,
			     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= pow(r,-2.) / ( 1. + a / r ); // Phi'/amp
  double d2phi= - (a + 2. * r) * pow(r,-4.) * pow(1.+a/r,-2.); // Phi''/amp
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return amp * ( d2phi * Z * Z / r / r + dphi * R * R / r / r / r );
}
double JaffePotentialRzderiv(double R,double Z, double phi,
			     double t,
			     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= pow(r,-2.) / ( 1. + a / r ); // Phi'/amp
  double d2phi= - (a + 2. * r) * pow(r,-4.) * pow(1.+a/r,-2.); // Phi''/amp
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return amp * R * Z * ( d2phi / r / r - dphi / r / r / r );
}
double JaffePotentialDens(double R,double Z, double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate density
  double r= sqrt ( R * R + Z * Z );
  return amp * M_1_PI / 4. / a * pow ( r * ( 1. + r / a ), -2. );
}
