#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//Double exponential disk potential
double DoubleExponentialDiskPotentialEval(double R,double z, double phi,
				  double t,
				  int nargs, double *args){
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double beta= *args++;
  double kmaxFac= *args++;
  double kmax= kmaxFac * beta;
  int nzeros= (int) *args++;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  double * j0zeros= args + 2 * glorder;
  double * dj0zeros= args + 2 * glorder + nzeros + 1;
  //Calculate potential
  double out= 0.;
  double k;
  int ii, jj;
  if ( R < 1. ) kmax= kmax/R;
  for (ii=0; ii < ( nzeros + 1 ); ii++) {
    for (jj=0; jj < glorder; jj++) {
      k= 0.5 * ( *(glx+jj) + 1. ) * *(dj0zeros+ii+1) + *(j0zeros+ii);
      out+= *(glw+jj) * *(dj0zeros+ii+1) * gsl_sf_bessel_J0(k*R) 
	* pow(alpha * alpha + k * k,-1.5) 
	* (beta * exp(-k * fabs(z) ) - k * exp(-beta * fabs(z) ))
	/ (beta * beta - k * k);
    }
    if ( k > kmax ) break;
  }
  return - amp * 2 * M_PI * alpha * out;
}
double DoubleExponentialDiskPotentialRforce(double R,double z, double phi,
					    double t,
					    int nargs, double *args){
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double beta= *args++;
  double kmaxFac= *args++;
  double kmax= 2. * kmaxFac * beta;
  int nzeros= (int) *args++;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  double * j1zeros= args + 2 * glorder + 2 * (nzeros + 1);
  double * dj1zeros= args + 2 * glorder + 3 * (nzeros + 1);
  //Calculate potential
  double out= 0.;
  double k;
  int ii, jj;
  if ( R < 1. ) kmax= kmax/R;
  for (ii=0; ii < ( nzeros + 1 ); ii++) {
    for (jj=0; jj < glorder; jj++) {
      k= 0.5 * ( *(glx+jj) + 1. ) * *(dj1zeros+ii+1) + *(j1zeros+ii);
      out+= *(glw+jj) * *(dj1zeros+ii+1) * k * gsl_sf_bessel_J1(k*R) 
	* pow(alpha * alpha + k * k,-1.5) 
	* (beta * exp(-k * fabs(z) ) - k * exp(-beta * fabs(z) ))
	/ (beta * beta - k * k);
    }
    if ( k > kmax ) break;
  }
  return - amp * 2 * M_PI * alpha * out;
}
double DoubleExponentialDiskPotentialPlanarRforce(double R,double phi,
						  double t,
						  int nargs, double *args){
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double beta= *args++;
  double kmaxFac= *args++;
  double kmax= 2. * kmaxFac * beta;
  int nzeros= (int) *args++;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  double * j1zeros= args + 2 * glorder + 2 * (nzeros + 1);
  double * dj1zeros= args + 2 * glorder + 3 * (nzeros + 1);
  //Calculate potential
  double out= 0.;
  double k;
  int ii, jj;
  if ( R < 1. ) kmax= kmax/R;
  for (ii=0; ii < ( nzeros + 1 ); ii++) {
    for (jj=0; jj < glorder; jj++) {
      k= 0.5 * ( *(glx+jj) + 1. ) * *(dj1zeros+ii+1) + *(j1zeros+ii);
      out+= *(glw+jj) * *(dj1zeros+ii+1) * k * gsl_sf_bessel_J1(k*R) 
	* pow(alpha * alpha + k * k,-1.5) 
	/ (beta + k);
    }
    if ( k > kmax ) break;
  }
  return - amp * 2 * M_PI * alpha * out;
}
double DoubleExponentialDiskPotentialzforce(double R,double z,double phi,
					    double t,
					    int nargs, double *args){
  //Get args
  double amp= *args++;
  double alpha= *args++;
  double beta= *args++;
  double kmaxFac= *args++;
  double kmax= kmaxFac * beta;
  int nzeros= (int) *args++;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  double * j0zeros= args + 2 * glorder;
  double * dj0zeros= args + 2 * glorder + nzeros + 1;
  //Calculate potential
  double out= 0.;
  double k;
  int ii, jj;
  if ( R < 1. ) kmax= kmax/R;
  for (ii=0; ii < ( nzeros + 1 ); ii++) {
    for (jj=0; jj < glorder; jj++) {
      k= 0.5 * ( *(glx+jj) + 1. ) * *(dj0zeros+ii+1) + *(j0zeros+ii);
      out+= *(glw+jj) * *(dj0zeros+ii+1) * k * gsl_sf_bessel_J0(k*R) 
	* pow(alpha * alpha + k * k,-1.5) 
	* (exp(-k * fabs(z) ) - exp(-beta * fabs(z) ))
	/ (beta * beta - k * k);
    }
    if ( k > kmax ) break;
  }
  if ( z > 0. )
    return - amp * 2 * M_PI * alpha * beta * out;
  else
    return amp * 2 * M_PI * alpha * beta * out;
}
