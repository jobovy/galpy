#include <math.h>
#include <galpy_potentials.h>
//HernquistPotential
//2 arguments: amp, a
double HernquistPotentialEval(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp / (1. + sqrtRz / a ) / 2. / a;
}
double HernquistPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * R / a / sqrtRz * pow(1. + sqrtRz / a , -2. ) / 2. / a;
}
double HernquistPotentialPlanarRforce(double R,double phi,
					    double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  return - amp / a * pow(1. + R / a , -2. ) / 2. / a;
}
double HernquistPotentialzforce(double R,double Z,double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * Z / a / sqrtRz * pow(1. + sqrtRz / a , -2. ) / 2. / a;
}
double HernquistPotentialPlanarR2deriv(double R,double phi,
					     double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate R2deriv
  return -amp / a / a / a * pow(1. + R / a, -3. );
}
double HernquistPotentialR2deriv(double R,double Z, double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= amp / 2. / a / a * pow(1. + r / a , -2.); // Phi'
  double d2phi= -amp / a / a / a * pow(1. + r / a , -3.); // Phi''
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return d2phi * R * R / r / r + dphi * Z * Z / r / r / r;
}
double HernquistPotentialz2deriv(double R,double Z, double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= amp / 2. / a / a * pow(1. + r / a , -2.); // Phi'
  double d2phi= -amp / a / a / a * pow(1. + r / a , -3.); // Phi''
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return d2phi * Z * Z / r / r + dphi * R * R / r / r / r;
}
double HernquistPotentialRzderiv(double R,double Z, double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Spherical: r, Phi'(r), Phi''(r)
  double r= sqrt( R * R + Z * Z );
  double dphi= amp / 2. / a / a * pow(1. + r / a , -2.); // Phi'
  double d2phi= -amp / a / a / a * pow(1. + r / a , -3.); // Phi''
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( d2phi / r / r - dphi / r / r / r );
}
double HernquistPotentialDens(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args;
  //Calculate density
  double r= sqrt ( R * R + Z * Z );
  return amp * M_1_PI / 4. / a / a / r * pow ( 1. + r / a , -3. );
}
