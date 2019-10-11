#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include "sf_math.h"
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
  // See if Hernquist or Jaffe
  if (alpha == 1.) {
    return HernquistPotentialEval(R, Z, phi, t, potentialArgs);
  }
  else if (alpha == 2.) {
    return JaffePotentialEval(R, Z, phi, t, potentialArgs);
  }
  // else, calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return amp * (
          ((a / sqrtRz) / 2. * hyp2f1(1., 4.-alpha, 3., -(a / sqrtRz))) -
          (gsl_sf_gamma(3.-alpha) / gsl_sf_gamma(4.-alpha))
         ) / sqrtRz;
}

double DehnenSphericalPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  // See if Hernquist or Jaffe
  if (alpha == 1.) {
    return HernquistPotentialRforce(R, Z, phi, t, potentialArgs);
  }
  else if (alpha == 2.) {
    return JaffePotentialRforce(R, Z, phi, t, potentialArgs);
  }
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (R / pow(sqrtRz, alpha) *
                  pow(a, alpha-3.) / (3.-alpha) *
                  hyp2f1(3.-alpha, 4.-alpha, 4.-alpha, -sqrtRz / a));
}

double DehnenSphericalPotentialPlanarRforce(double R,double phi,
					    double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  // See if Hernquist or Jaffe
  if (alpha == 1.) {
    return HernquistPotentialPlanarRforce(R, phi, t, potentialArgs);
  }
  else if (alpha == 2.) {
    return JaffePotentialPlanarRforce(R, phi, t, potentialArgs);
  }
  //Calculate Rforce
  double sqrtRz= pow(R*R,0.5);
  return - amp * (pow(sqrtRz, 1.-alpha) *
                  pow(a, (alpha-3.)) / (3.-alpha) *
                  hyp2f1(3.-alpha, 4.-alpha, 4.-alpha, -sqrtRz / a));
}

double DehnenSphericalPotentialzforce(double R,double Z,double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  // See if Hernquist or Jaffe
  if (alpha == 1.) {
    return HernquistPotentialzforce(R, Z, phi, t, potentialArgs);
  }
  else if (alpha == 2.) {
   return JaffePotentialzforce(R, Z, phi, t, potentialArgs);
  }
  //Calculate Rforce
  double sqrtRz= pow(R*R+Z*Z,0.5);
  return - amp * (Z / pow(sqrtRz, alpha) * pow(a, alpha-3.) / (3.-alpha) *
                  hyp2f1(3.-alpha, 4.-alpha, 4.-alpha, -sqrtRz / a));
}

double DehnenSphericalPotentialPlanarR2deriv(double R,double phi,
					     double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double alpha= *args;
  // See if Hernquist or Jaffe
  if (alpha == 1.) {
    return HernquistPotentialPlanarR2deriv(R, phi, t, potentialArgs);
  }
  else if (alpha == 2.) {
    return JaffePotentialPlanarR2deriv(R, phi, t, potentialArgs);
  }
  // Calculate R2deri
  return amp * (
    pow(R, -alpha) * pow(a+R, alpha-4.) * (2.*R + a*(alpha-1.))
    ) / (alpha-3.);
}
