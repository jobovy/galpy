#include <math.h>
#include <galpy_potentials.h>
//NFWPotential
//2 arguments: amp, a
double NFWPotentialEval(double R,double Z, double phi,
			  double t,
			struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * log ( 1. + sqrtRz / a ) / sqrtRz;
}
double NFWPotentialRforce(double R,double Z, double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double Rz= R*R+Z*Z;
  double sqrtRz= pow(Rz,0.5);
  return amp * R * (1. / Rz / (a + sqrtRz)-log(1.+sqrtRz / a)/sqrtRz/Rz);
}
double NFWPotentialPlanarRforce(double R,double phi,
					    double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  return amp / R * (1. / (a + R)-log(1.+ R / a)/ R);
}
double NFWPotentialzforce(double R,double Z,double phi,
			  double t,
			  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  double Rz= R*R+Z*Z;
  double sqrtRz= pow(Rz,0.5);
  return amp * Z * (1. / Rz / (a + sqrtRz)-log(1.+sqrtRz / a)/sqrtRz/Rz);
}
double NFWPotentialPlanarR2deriv(double R,double phi,
				 double t,
				 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate R2deriv
  double aR= a+R;
  double aR2= aR*aR;
  return amp * (((R*(2.*a+3.*R))-2.*aR2*log(1.+R/a))/R/R/R/aR2);
}
