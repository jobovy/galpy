#include <gsl/gsl_spline.h>
#include <galpy_potentials.h>
// interpSphericalPotential: 6 arguments: amp (not used here), rmin, rmax,
// M(<rmax), Phi0, Phimax
double interpSphericalPotentialrevaluate(double r,double t,
					 struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double rmin= *(args+1);
  double rmax= *(args+2);
  double Mmax= *(args+3);
  double Phi0= *(args+4);
  double Phimax= *(args+5);
  if ( r >= rmax ) {
    return -Mmax/r+Phimax;
  }
  else {
    return r < rmin ? 0. : \
      -gsl_spline_eval_integ(*potentialArgs->spline1d,
			     rmin,r,*potentialArgs->acc1d) + Phi0;
  }
}
double interpSphericalPotentialrforce(double r,double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double rmin= *(args+1);
  double rmax= *(args+2);
  double Mmax= *(args+3);
  if ( r >= rmax ) {
    return -Mmax/r/r;
  }
  else {
    return r < rmin ? 0. : gsl_spline_eval(*potentialArgs->spline1d,
					   r,*potentialArgs->acc1d);
  }
}
double interpSphericalPotentialr2deriv(double r,double t,
				       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double rmin= *(args+1);
  double rmax= *(args+2);
  double Mmax= *(args+3);
  if ( r >= rmax ) {
    return -2. * Mmax / r / r / r;
  }
  else {
    return r < rmin ? 0. : -gsl_spline_eval_deriv(*potentialArgs->spline1d,
						  r,*potentialArgs->acc1d);
  }
}
double interpSphericalPotentialrdens(double r,double t,
				     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double rmin= *(args+1);
  double rmax= *(args+2);
  if ( r >= rmax ) {
    return 0.;
  }
  else {
    return r < rmin ? 0. : M_1_PI / 4. \
      * ( interpSphericalPotentialr2deriv(r,t,potentialArgs)
	  - 2. * interpSphericalPotentialrforce(r,t,potentialArgs)/r);
  }
}
