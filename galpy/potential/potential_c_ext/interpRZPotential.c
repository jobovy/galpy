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
// Full 3D Hessian (R2deriv/z2deriv/Rzderiv; phi derivatives are zero for this
// axisymmetric potential) for the 3D variational equations: like the forces,
// each 2nd derivative is interpolated from its own grid of exact (original
// potential) values via 2D cubic B-splines. d2Phi/dR2 and d2Phi/dz2 are EVEN
// in z (evaluate at |z|, like the potential/Rforce), d2Phi/dRdz is ODD in z
// (evaluate at |z| and flip the sign for z<0, like the zforce).
double interpRZPotentialR2deriv(double R,double z, double phi,
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
  //Calculate R2deriv through interpolation
  return amp * interp_2d_eval_cubic_bspline(potentialArgs->i2dr2deriv,y,
					    fabs(z),
					    potentialArgs->accxr2deriv,
					    potentialArgs->accyr2deriv);
}
double interpRZPotentialz2deriv(double R,double z, double phi,
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
  //Calculate z2deriv through interpolation
  return amp * interp_2d_eval_cubic_bspline(potentialArgs->i2dz2deriv,y,
					    fabs(z),
					    potentialArgs->accxz2deriv,
					    potentialArgs->accyz2deriv);
}
double interpRZPotentialRzderiv(double R,double z, double phi,
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
  //Calculate Rzderiv through interpolation; odd in z
  if ( z < 0. )
    return - amp * interp_2d_eval_cubic_bspline(potentialArgs->i2drzderiv,y,
						-z,
						potentialArgs->accxrzderiv,
						potentialArgs->accyrzderiv);
  else
    return amp * interp_2d_eval_cubic_bspline(potentialArgs->i2drzderiv,y,
					      z,
					      potentialArgs->accxrzderiv,
					      potentialArgs->accyrzderiv);
}
