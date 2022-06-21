#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <galpy_potentials.h>
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
  double amp,t0,tf,d_ind,x,y,obj_x,obj_y,obj_z, Rdist,RF;
  double * args= potentialArgs->args;
  //Get args
  amp= *args;
  t0= *(args+1);
  tf= *(args+2);
  d_ind= (t-t0)/(tf-t0);
  x= R*cos(phi);
  y= R*sin(phi);
  constrain_range(&d_ind);
  // Interpolate x, y, z
  obj_x= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  obj_y= gsl_spline_eval(*(potentialArgs->spline1d+1),d_ind,
			 *(potentialArgs->acc1d+1));
  obj_z= gsl_spline_eval(*(potentialArgs->spline1d+2),d_ind,
			 *(potentialArgs->acc1d+2));
  Rdist= pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate R force
  RF= calcRforce(Rdist,(obj_z-z),phi,t,potentialArgs->nwrapped,
		 potentialArgs->wrappedPotentialArg);
  return -amp*RF*(cos(phi)*(obj_x-x)+sin(phi)*(obj_y-y))/Rdist;
}

double MovingObjectPotentialzforce(double R,double z,double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double amp,t0,tf,d_ind,x,y,obj_x,obj_y,obj_z, Rdist;
  double * args= potentialArgs->args;
  //Get args
  amp= *args;
  t0= *(args+1);
  tf= *(args+2);
  d_ind= (t-t0)/(tf-t0);
  x= R*cos(phi);
  y= R*sin(phi);
  constrain_range(&d_ind);
  // Interpolate x, y, z
  obj_x= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  obj_y= gsl_spline_eval(*(potentialArgs->spline1d+1),d_ind,
			 *(potentialArgs->acc1d+1));
  obj_z= gsl_spline_eval(*(potentialArgs->spline1d+2),d_ind,
			 *(potentialArgs->acc1d+2));
  Rdist= pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate z force
  return -amp * calczforce(Rdist,(obj_z-z),phi,t,potentialArgs->nwrapped,
			   potentialArgs->wrappedPotentialArg);
}

double MovingObjectPotentialphitorque(double R,double z,double phi,
					double t,
					struct potentialArg * potentialArgs){
  double amp,t0,tf,d_ind,x,y,obj_x,obj_y,obj_z, Rdist,RF;
  double * args= potentialArgs->args;
  //Get args
  amp= *args;
  t0= *(args+1);
  tf= *(args+2);
  d_ind= (t-t0)/(tf-t0);
  x= R*cos(phi);
  y= R*sin(phi);
  constrain_range(&d_ind);
  // Interpolate x, y, z
  obj_x= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  obj_y= gsl_spline_eval(*(potentialArgs->spline1d+1),d_ind,
			 *(potentialArgs->acc1d+1));
  obj_z= gsl_spline_eval(*(potentialArgs->spline1d+2),d_ind,
			 *(potentialArgs->acc1d+2));
  Rdist= pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate phitorque
  RF= calcRforce(Rdist,(obj_z-z),phi,t,potentialArgs->nwrapped,
		 potentialArgs->wrappedPotentialArg);
  return -amp*RF*R*(cos(phi)*(obj_y-y)-sin(phi)*(obj_x-x))/Rdist;
}

double MovingObjectPotentialPlanarRforce(double R, double phi,
				      double t,
				      struct potentialArg * potentialArgs){
  double amp,t0,tf,d_ind,x,y,obj_x,obj_y,Rdist,RF;
  double * args= potentialArgs->args;
  //Get args
  amp= *args;
  t0= *(args+1);
  tf= *(args+2);
  d_ind= (t-t0)/(tf-t0);
  x= R*cos(phi);
  y= R*sin(phi);
  constrain_range(&d_ind);
  // Interpolate x, y
  obj_x= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  obj_y= gsl_spline_eval(*(potentialArgs->spline1d+1),d_ind,
			 *(potentialArgs->acc1d+1));
  Rdist= pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate R force
  RF= calcPlanarRforce(Rdist, phi, t, potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
  return -amp*RF*(cos(phi)*(obj_x-x)+sin(phi)*(obj_y-y))/Rdist;
}

double MovingObjectPotentialPlanarphitorque(double R, double phi,
					double t,
					struct potentialArg * potentialArgs){
  double amp,t0,tf,d_ind,x,y,obj_x,obj_y,Rdist,RF;
  double * args= potentialArgs->args;
  // Get args
  amp= *args;
  t0= *(args+1);
  tf= *(args+2);
  d_ind= (t-t0)/(tf-t0);
  x= R*cos(phi);
  y= R*sin(phi);
  constrain_range(&d_ind);
  // Interpolate x, y
  obj_x= gsl_spline_eval(*potentialArgs->spline1d,d_ind,*potentialArgs->acc1d);
  obj_y= gsl_spline_eval(*(potentialArgs->spline1d+1),d_ind,
			 *(potentialArgs->acc1d+1));
  Rdist= pow(pow(x-obj_x, 2)+pow(y-obj_y, 2), 0.5);
  // Calculate phitorque
  RF= calcPlanarRforce(Rdist, phi, t, potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg);
  return -amp*RF*R*(cos(phi)*(obj_y-y)-sin(phi)*(obj_x-x))/Rdist;
}
