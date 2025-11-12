#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include "galpy_potentials.h"
#include "wrap_xsf.h"
//TwoPowerSphericalPotential
//4 arguments: amp, a, alpha, beta
double TwoPowerSphericalPotentialEval(double R,double Z, double phi,
                                       double t,
                                       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate potential
  double r= sqrt(R*R+Z*Z);
  if (beta == 3.0) {
    return amp * (1./a) * (1. - pow(r/a, 2.-alpha) / (3.-alpha)
                           * hyp2f1(3.-alpha, 2.-alpha, 4.-alpha, -r/a))
                         / (alpha - 2.);
  } else {
    r += 1e-11; // avoid division by zero and numerical instability
    return amp * gsl_sf_gamma(beta - 3.)
               * (pow(r/a, 3.-beta) / gsl_sf_gamma(beta - 1.)
                  * hyp2f1(beta - 3., beta - alpha, beta - 1., -a/r)
                  - gsl_sf_gamma(3.-alpha) / gsl_sf_gamma(beta - alpha))
               / r;
  }
}

double TwoPowerSphericalPotentialRforce(double R,double Z, double phi,
                                        double t,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate Rforce
  double r= sqrt(R*R+Z*Z);
  return -amp * R / pow(r, alpha) * pow(a, alpha - 3.) / (3. - alpha)
              * hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -r/a);
}

double TwoPowerSphericalPotentialPlanarRforce(double R,double phi,
                                              double t,
                                              struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate Rforce
  return -amp / pow(R, alpha) * pow(a, alpha - 3.) / (3. - alpha)
              * hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -R/a);
}

double TwoPowerSphericalPotentialzforce(double R,double Z,double phi,
                                        double t,
                                        struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate zforce
  double r= sqrt(R*R+Z*Z);
  return -amp * Z / pow(r, alpha) * pow(a, alpha - 3.) / (3. - alpha)
              * hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -r/a);
}

double TwoPowerSphericalPotentialPlanarR2deriv(double R,double phi,
                                               double t,
                                               struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate R2deriv using analytical derivative
  double A = pow(a, alpha - 3.) / (3. - alpha);
  double hyper = hyp2f1(3. - alpha, beta - alpha, 4. - alpha, -R/a);
  double hyper_deriv = (3. - alpha) * (beta - alpha) / (4. - alpha)
                       * hyp2f1(4. - alpha, 1. + beta - alpha, 5. - alpha, -R/a);

  double term1 = A * pow(R, -alpha) * hyper;
  double term2 = -alpha * A * pow(R, -alpha - 1.) * hyper;
  double term3 = -A * pow(R, -alpha) / a * hyper_deriv;
  return amp * (term1 + term2 + term3);
}

double TwoPowerSphericalPotentialDens(double R,double Z, double phi,
                                      double t,
                                      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args++;
  double a= *args++;
  double alpha= *args++;
  double beta= *args;
  //Calculate density
  double r= sqrt(R*R+Z*Z);
  return amp * pow(a/r, alpha) / pow(1. + r/a, beta - alpha)
             / 4. / M_PI / (a*a*a);
}
