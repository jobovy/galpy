#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//PowerSphericalPotentialwCutoff
//3  arguments: amp, alpha, rc
double mass(double r2,double alpha, double rc){
  return 2. * M_PI * pow ( rc , 3. - alpha ) * ( gsl_sf_gamma ( 1.5 - 0.5 * alpha ) - gsl_sf_gamma_inc ( 1.5 - 0.5 * alpha , r2 / rc / rc ) );
}
double PowerSphericalPotentialwCutoffEval(double R,double Z, double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double rc= *args;
  //Radius
  double r2= R*R+Z*Z;
  double r= sqrt(r2);
  return amp * 2. * M_PI * pow(rc,3.-alpha) / r * ( r / rc * ( gsl_sf_gamma ( 1. - 0.5 * alpha ) - gsl_sf_gamma_inc ( 1. - 0.5 * alpha , r2 / rc / rc ) ) - ( gsl_sf_gamma ( 1.5 - 0.5 * alpha ) - gsl_sf_gamma_inc ( 1.5 - 0.5 * alpha , r2 / rc / rc) ) );
}
double PowerSphericalPotentialwCutoffRforce(double R,double Z, double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double rc= *args;
  //Radius
  double r2= R*R+Z*Z;
  //Calculate Rforce
  return - amp * mass (r2,alpha,rc) * R / pow(r2,1.5);
}
double PowerSphericalPotentialwCutoffPlanarRforce(double R,double phi,
						  double t,
						  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double rc= *args;
  //Radius
  double r2= R*R;
  //Calculate Rforce
  return - amp * mass (r2,alpha,rc) / r2;
}
double PowerSphericalPotentialwCutoffzforce(double R,double Z,double phi,
					    double t,
					    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double rc= *args;
  //Radius
  double r2= R*R+Z*Z;
  //Calculate Rforce
  return - amp * mass (r2,alpha,rc) * Z / pow(r2,1.5);
}
double PowerSphericalPotentialwCutoffPlanarR2deriv(double R,double phi,
						   double t,
						   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double rc= *args;
  //Radius
  double r2= R*R;
  //Calculate R2deriv
  return amp * ( 4. * M_PI * pow(r2,- 0.5 * alpha) * exp(-r2/rc/rc) - 2. * mass(r2,alpha,rc)/pow(r2,1.5) );
}
double PowerSphericalPotentialwCutoffDens(double R,double Z, double phi,
					  double t,
					  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double rc= *args;
  //Radius
  double r2= R * R + Z * Z;
  double r= sqrt(r2);
  return amp * pow(r,-alpha) * exp ( -r2 / rc / rc );
}
