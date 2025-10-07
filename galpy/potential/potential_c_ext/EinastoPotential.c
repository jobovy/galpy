#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//EinastoPotential
// 3 arguments: amp, h, n
double EinastoPotentialrevaluate(double r,double t,
  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double h= *(args+1);
  double n= *(args+2);

  double s = r / h;
  double s_1n = pow(s,1.0/n);

  return (
      -(4.0 * M_PI * pow(h,2) * n * gsl_sf_gamma(3*n))
      * pow(s,-1)
      * (
          1 - gsl_sf_gamma_inc_Q(3*n, s_1n) + s * (gsl_sf_gamma_inc(2*n, s_1n)) / gsl_sf_gamma(3*n)
      )
  );
}

double EinastoPotentialrforce(double r,double t,
  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double h= *(args+1);
  double n= *(args+2);

  double s = r / h;

  double gamma_3n = gsl_sf_gamma(3 * n);
  double gamma_upper_3n = gsl_sf_gamma_inc_Q(3 * n, pow(s, 1.0 / n));

  double s_2 = pow(s, -2.0);

  return (
    (4.0 * M_PI * h * n * gamma_3n)
    * s_2
    * (gamma_upper_3n - 1.0)
  );
}

double EinastoPotentialr2deriv(double r,double t,
   struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double h= *(args+1);
  double n= *(args+2);

  double s = r / h;
  double s_1n = pow(s,1.0/n);

  double gamma_3n = gsl_sf_gamma(3*n);
  double gamma_upper_3n = gsl_sf_gamma_inc_Q(3*n, s_1n);

  return (
      - (4.0 * M_PI * n * gamma_3n)
      * ((-2 * pow(s,-3)) * (gamma_upper_3n - 1)
      - ((1/n) * exp(-s_1n)/gamma_3n))
  );
}

double EinastoPotentialrdens(double r,
			    double t,
			    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double h= *(args+1);
  double n= *(args+2);
  //Calculate potential

  return exp(-pow(r/h, 1.0 / n));
}
