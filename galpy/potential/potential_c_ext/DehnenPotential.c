#include <math.h>
#include "galpy_potentials.h"
//DehnenPotential
//3 arguments: amp, a, alpha
double DehnenSphericalPotentialEval(double R,double Z, double phi,
			      double t,
			      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  // if (alpha == 2.) {return -amp * R * pow( sqrtRz , -3. ) / ( 1. + a / sqrtRz );}  // not needed b/c Jaffe
  return -amp * (1. - pow(sqrtRz/(sqrtRz+a), 2.-alpha)) / (a * (2. - alpha) * (3. - alpha));
}

double DehnenSphericalPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (R * pow(1.+a/sqrtRz, alpha) / pow(a+sqrtRz,3.) / (3.-alpha));
}

double DehnenSphericalPotentialPlanarRforce(double R,double phi,
					    double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  //Calculate Rforce
  return - amp * (R * pow(1.+a/R, alpha) / pow(a+R, 3.) / (3.-alpha));
}

double DehnenSphericalPotentialzforce(double R,double Z,double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (Z * pow(1. + a/sqrtRz, alpha) / pow(a+sqrtRz, 3.) / (3.-alpha));
}

double DehnenSphericalPotentialPlanarR2deriv(double R,double phi,
					     double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  // return
  return -amp * (pow(1.+a/R, alpha) * pow(a+R, -4.) * (2.*R + a*(alpha-1.))) / (3.-alpha);
}
double DehnenSphericalPotentialR2deriv(double R,double Z, double phi,
				       double t,
				       struct potentialArg * potentialArgs){
  //Spherical: Phi''(r)=PlanarR2deriv(r), Phi'(r)=-PlanarRforce(r) (incl. amp)
  double r= sqrt( R * R + Z * Z );
  double Phipp= DehnenSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -DehnenSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  //R2deriv = Phi''*R^2/r^2 + Phi'*z^2/r^3
  return Phipp * R * R / r / r + Phip * Z * Z / r / r / r;
}
double DehnenSphericalPotentialz2deriv(double R,double Z, double phi,
				       double t,
				       struct potentialArg * potentialArgs){
  double r= sqrt( R * R + Z * Z );
  double Phipp= DehnenSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -DehnenSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  //z2deriv = Phi''*z^2/r^2 + Phi'*R^2/r^3
  return Phipp * Z * Z / r / r + Phip * R * R / r / r / r;
}
double DehnenSphericalPotentialRzderiv(double R,double Z, double phi,
				       double t,
				       struct potentialArg * potentialArgs){
  double r= sqrt( R * R + Z * Z );
  double Phipp= DehnenSphericalPotentialPlanarR2deriv(r,phi,t,potentialArgs);
  double Phip= -DehnenSphericalPotentialPlanarRforce(r,phi,t,potentialArgs);
  //Rzderiv = R*z*(Phi''/r^2 - Phi'/r^3)
  return R * Z * ( Phipp / r / r - Phip / r / r / r );
}
double DehnenSphericalPotentialDens(double R,double Z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  //Calculate density
  double r= sqrt ( R * R + Z * Z );
  return amp * M_1_PI / 4. * pow (r,-alpha ) * pow ( 1. + r / a, alpha-4.) \
    * pow (a, alpha - 3.);
}
