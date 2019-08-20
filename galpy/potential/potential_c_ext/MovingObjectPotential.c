#include <math.h>
#include <galpy_potentials.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
// MovingObjectPotential
// 3 arguments: amp, t0, tf
void constrain_range(double * d) {
  // Constrains index to be within interpolation range
  if (*d < 0) *d = 0.0;
  if (*d > 1) *d = 1.0;
}
double MovingObjectPotentialRforce(double R,double z, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double t0= *(args+1);
  double tf= *(args+2);

  double d_ind = ((t-t0)/(tf-t0));

  double x = R*cos(phi);
  double y = R*sin(phi);

  constrain_range(&d_ind);
  // Interpolate x, y, z
  double obj_x = gsl_spline_eval(potentialArgs->xSpline, d_ind, potentialArgs->accx);
  double obj_y = gsl_spline_eval(potentialArgs->ySpline, d_ind, potentialArgs->accy);
  double obj_z = gsl_spline_eval(potentialArgs->zSpline, d_ind, potentialArgs->accz);

  double Rdist = pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate R force
  double RF = calcRforce(Rdist,(obj_z-z),phi,t,potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg);

  return -amp*RF*(cos(phi)*(obj_x-x)+sin(phi)*(obj_y-y))/Rdist;
}

double MovingObjectPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double t0= *(args+1);
  double tf= *(args+2);

  double d_ind = ((t-t0)/(tf-t0));
  double x = R*cos(phi);
  double y = R*sin(phi);

  constrain_range(&d_ind);
  // Interpolate x, y, z
  double obj_x = gsl_spline_eval(potentialArgs->xSpline, d_ind, potentialArgs->accx);
  double obj_y = gsl_spline_eval(potentialArgs->ySpline, d_ind, potentialArgs->accy);
  double obj_z = gsl_spline_eval(potentialArgs->zSpline, d_ind, potentialArgs->accz);

  double Rdist = pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate z force
  double zF = calczforce(Rdist,(obj_z-z),phi,t,potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg);
  return -amp*zF;
}

double MovingObjectPotentialphiforce(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double t0= *(args+1);
  double tf= *(args+2);

  double d_ind = ((t-t0)/(tf-t0));
  double x = R*cos(phi);
  double y = R*sin(phi);
  constrain_range(&d_ind);
  // Interpolate x, y, z
  double obj_x = gsl_spline_eval(potentialArgs->xSpline, d_ind, potentialArgs->accx);
  double obj_y = gsl_spline_eval(potentialArgs->ySpline, d_ind, potentialArgs->accy);
  double obj_z = gsl_spline_eval(potentialArgs->zSpline, d_ind, potentialArgs->accz);

  double Rdist = pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate phiforce
  double RF = calcRforce(Rdist,(obj_z-z),phi,t,potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg);

  return -amp*RF*R*(cos(phi)*(obj_y-y)-sin(phi)*(obj_x-x))/Rdist;
}

double MovingObjectPotentialPlanarRforce(double R, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args;
  double t0= *(args+1);
  double tf= *(args+2);

  double d_ind = ((t-t0)/(tf-t0));
  double x = R*cos(phi);
  double y = R*sin(phi);

  constrain_range(&d_ind);
  // Interpolate x, y
  double obj_x = gsl_spline_eval(potentialArgs->xSpline, d_ind, potentialArgs->accx);
  double obj_y = gsl_spline_eval(potentialArgs->ySpline, d_ind, potentialArgs->accy);

  double Rdist = pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate R force
  double RF = calcPlanarRforce(Rdist, phi, t, potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg);

  return -amp*RF*(cos(phi)*(obj_x-x)+sin(phi)*(obj_y-y))/Rdist;
}

double MovingObjectPotentialPlanarphiforce(double R, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  // Get args
  double amp= *args;
  double t0= *(args+1);
  double tf= *(args+2);

  double d_ind = ((t-t0)/(tf-t0));
  double x = R*cos(phi);
  double y = R*sin(phi);

  constrain_range(&d_ind);
  // Interpolate x, y
  double obj_x = gsl_spline_eval(potentialArgs->xSpline, d_ind, potentialArgs->accx);
  double obj_y = gsl_spline_eval(potentialArgs->ySpline, d_ind, potentialArgs->accy);

  double Rdist = pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate phiforce
  double RF = calcPlanarRforce(Rdist, phi, t, potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg);

  return -amp*RF*R*(cos(phi)*(obj_y-y)-sin(phi)*(obj_x-x))/Rdist;
}
