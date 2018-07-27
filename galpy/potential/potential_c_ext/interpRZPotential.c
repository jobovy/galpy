#include <math.h>
#include <galpy_potentials.h>
#include <interp_2d.h>
//interpRZpotential
//2 remaining arguments: amp, logR
double interpRZPotentialEval(double R,double z, double phi,
			     double t,
			     struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double y;
  double amp= *args++;
  int logR= (int) *args;
  if ( logR == 1)
    y= ( R > 0. ) ? log(R): -20.72326583694641;
  else
    y= R;
  //Calculate potential through interpolation
  return amp * interp_2d_eval_cubic_bspline(potentialArgs->i2d,y,fabs(z),
					    potentialArgs->accx,
					    potentialArgs->accy);
}
double interpRZPotentialRforce(double R,double z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double y;
  double amp= *args++;
  int logR= (int) *args;
  if ( logR == 1)
    y= ( R > 0. ) ? log(R): -20.72326583694641;
  else
    y= R;
  //Calculate potential through interpolation
  return amp * interp_2d_eval_cubic_bspline(potentialArgs->i2drforce,y,fabs(z),
					    potentialArgs->accxrforce,
					    potentialArgs->accyrforce);
}
double interpRZPotentialzforce(double R,double z, double phi,
			       double t,
			       struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double y;
  double amp= *args++;
  int logR= (int) *args;
  if ( logR == 1)
    y= ( R > 0. ) ? log(R): -20.72326583694641;
  else
    y= R;
  //Calculate potential through interpolation
  if ( z < 0. )
    return - amp * interp_2d_eval_cubic_bspline(potentialArgs->i2dzforce,y,
						-z,
						potentialArgs->accxzforce,
						potentialArgs->accyzforce);
  else
    return amp * interp_2d_eval_cubic_bspline(potentialArgs->i2dzforce,y,
					      z,
					      potentialArgs->accxzforce,
					      potentialArgs->accyzforce);
}
