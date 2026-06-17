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

// ---------------------------------------------------------------------------
// Second derivatives for the variational equations (integrate_dxdv).
// Phi(x,t) = amp * Psi(R'(x,y), z'(z)) with R'^2 = (x-x_o(t))^2 + (y-y_o(t))^2
// and z' = z_o(t) - z (the same conventions as the force evaluators above).
// The moving-object shift x_o(t) is a pure translation of the evaluation
// point, so the Hessian is just the kernel's Hessian at the shifted point
// chain-ruled to the field point's cylindrical coordinates: the
// time-dependence enters only through (R', z'), with no extra terms.
// The kernel Psi is required to be axisymmetric (enforced at setup in
// MovingObjectPotential.__init__), so its phi derivatives vanish and the four
// quantities dPsi/dR', d2Psi/dR'2, d2Psi/dR'dz', d2Psi/dz'2 -- evaluated
// through the wrapped potential's force/Hessian pointers, exactly like the
// forces -- determine the full Cartesian Hessian of Phi at the field point.
// ---------------------------------------------------------------------------
static void MovingObjectPotentialxyzHess(double R, double z, double phi,
					 double t,
					 struct potentialArg * potentialArgs,
					 double * phix, double * phiy,
					 double * phixx, double * phixy,
					 double * phiyy, double * phixz,
					 double * phiyz, double * phizz){
  // Cartesian first (phix,phiy; phiz is never needed) and second derivatives
  // of Phi at the field point (R,z,phi) at time t, incl. the overall amp.
  double amp,t0,tf,d_ind,x,y,obj_x,obj_y,obj_z,xd,yd,zd,Rdist,RF,R2d,Rzd,z2d;
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
  xd= obj_x-x;
  yd= obj_y-y;
  zd= obj_z-z;
  Rdist= sqrt(xd*xd+yd*yd);
  // Kernel cylindrical derivatives at the shifted point (R',z')=(Rdist,zd)
  RF= calcRforce(Rdist,zd,phi,t,potentialArgs->nwrapped,
		 potentialArgs->wrappedPotentialArg); // = -dPsi/dR'
  R2d= calcR2deriv(Rdist,zd,phi,t,potentialArgs->nwrapped,
		   potentialArgs->wrappedPotentialArg); // = d2Psi/dR'2
  Rzd= calcRzderiv(Rdist,zd,phi,t,potentialArgs->nwrapped,
		   potentialArgs->wrappedPotentialArg); // = d2Psi/dR'dz'
  z2d= calcz2deriv(Rdist,zd,phi,t,potentialArgs->nwrapped,
		   potentialArgs->wrappedPotentialArg); // = d2Psi/dz'2
  // Chain rule with dR'/dx = (x-obj_x)/R' = -xd/R' and dz'/dz = -1; the minus
  // signs cancel pairwise in the pure-second-derivative terms below
  *phix= amp*RF*xd/Rdist;
  *phiy= amp*RF*yd/Rdist;
  *phixx= amp*(R2d*xd*xd/Rdist/Rdist-RF*yd*yd/Rdist/Rdist/Rdist);
  *phiyy= amp*(R2d*yd*yd/Rdist/Rdist-RF*xd*xd/Rdist/Rdist/Rdist);
  *phixy= amp*(R2d+RF/Rdist)*xd*yd/Rdist/Rdist;
  *phixz= amp*Rzd*xd/Rdist;
  *phiyz= amp*Rzd*yd/Rdist;
  *phizz= amp*z2d;
}

double MovingObjectPotentialR2deriv(double R,double z,double phi,double t,
				    struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,phixz,phiyz,phizz,cp,sp;
  MovingObjectPotentialxyzHess(R,z,phi,t,potentialArgs,&phix,&phiy,
			       &phixx,&phixy,&phiyy,&phixz,&phiyz,&phizz);
  cp= cos(phi);
  sp= sin(phi);
  return cp*cp*phixx+2.*cp*sp*phixy+sp*sp*phiyy;
}

double MovingObjectPotentialz2deriv(double R,double z,double phi,double t,
				    struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,phixz,phiyz,phizz;
  MovingObjectPotentialxyzHess(R,z,phi,t,potentialArgs,&phix,&phiy,
			       &phixx,&phixy,&phiyy,&phixz,&phiyz,&phizz);
  return phizz;
}

double MovingObjectPotentialRzderiv(double R,double z,double phi,double t,
				    struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,phixz,phiyz,phizz,cp,sp;
  MovingObjectPotentialxyzHess(R,z,phi,t,potentialArgs,&phix,&phiy,
			       &phixx,&phixy,&phiyy,&phixz,&phiyz,&phizz);
  cp= cos(phi);
  sp= sin(phi);
  return cp*phixz+sp*phiyz;
}

double MovingObjectPotentialphi2deriv(double R,double z,double phi,double t,
				      struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,phixz,phiyz,phizz,cp,sp;
  MovingObjectPotentialxyzHess(R,z,phi,t,potentialArgs,&phix,&phiy,
			       &phixx,&phixy,&phiyy,&phixz,&phiyz,&phizz);
  cp= cos(phi);
  sp= sin(phi);
  return R*R*(sp*sp*phixx-2.*cp*sp*phixy+cp*cp*phiyy)-R*(cp*phix+sp*phiy);
}

double MovingObjectPotentialRphideriv(double R,double z,double phi,double t,
				      struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,phixz,phiyz,phizz,cp,sp;
  MovingObjectPotentialxyzHess(R,z,phi,t,potentialArgs,&phix,&phiy,
			       &phixx,&phixy,&phiyy,&phixz,&phiyz,&phizz);
  cp= cos(phi);
  sp= sin(phi);
  return R*(cp*sp*(phiyy-phixx)+(cp*cp-sp*sp)*phixy)-sp*phix+cp*phiy;
}

double MovingObjectPotentialzphideriv(double R,double z,double phi,double t,
				      struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,phixz,phiyz,phizz,cp,sp;
  MovingObjectPotentialxyzHess(R,z,phi,t,potentialArgs,&phix,&phiy,
			       &phixx,&phixy,&phiyy,&phixz,&phiyz,&phizz);
  cp= cos(phi);
  sp= sin(phi);
  return R*(cp*phiyz-sp*phixz);
}

// Planar (2D, in-plane object track) second derivatives: same translation
// argument with the kernel's PLANAR force/R2deriv (the kernel at z'=0),
// mirroring how MovingObjectPotentialPlanarRforce evaluates the kernel.
static void MovingObjectPotentialPlanarxyHess(double R, double phi, double t,
					      struct potentialArg * potentialArgs,
					      double * phix, double * phiy,
					      double * phixx, double * phixy,
					      double * phiyy){
  double amp,t0,tf,d_ind,x,y,obj_x,obj_y,xd,yd,Rdist,RF,R2d;
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
  xd= obj_x-x;
  yd= obj_y-y;
  Rdist= sqrt(xd*xd+yd*yd);
  RF= calcPlanarRforce(Rdist,phi,t,potentialArgs->nwrapped,
		       potentialArgs->wrappedPotentialArg); // = -dPsi/dR'
  R2d= calcPlanarR2deriv(Rdist,phi,t,potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg); // = d2Psi/dR'2
  *phix= amp*RF*xd/Rdist;
  *phiy= amp*RF*yd/Rdist;
  *phixx= amp*(R2d*xd*xd/Rdist/Rdist-RF*yd*yd/Rdist/Rdist/Rdist);
  *phiyy= amp*(R2d*yd*yd/Rdist/Rdist-RF*xd*xd/Rdist/Rdist/Rdist);
  *phixy= amp*(R2d+RF/Rdist)*xd*yd/Rdist/Rdist;
}

double MovingObjectPotentialPlanarR2deriv(double R,double phi,double t,
					  struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,cp,sp;
  MovingObjectPotentialPlanarxyHess(R,phi,t,potentialArgs,&phix,&phiy,
				    &phixx,&phixy,&phiyy);
  cp= cos(phi);
  sp= sin(phi);
  return cp*cp*phixx+2.*cp*sp*phixy+sp*sp*phiyy;
}

double MovingObjectPotentialPlanarphi2deriv(double R,double phi,double t,
					    struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,cp,sp;
  MovingObjectPotentialPlanarxyHess(R,phi,t,potentialArgs,&phix,&phiy,
				    &phixx,&phixy,&phiyy);
  cp= cos(phi);
  sp= sin(phi);
  return R*R*(sp*sp*phixx-2.*cp*sp*phixy+cp*cp*phiyy)-R*(cp*phix+sp*phiy);
}

double MovingObjectPotentialPlanarRphideriv(double R,double phi,double t,
					    struct potentialArg * potentialArgs){
  double phix,phiy,phixx,phixy,phiyy,cp,sp;
  MovingObjectPotentialPlanarxyHess(R,phi,t,potentialArgs,&phix,&phiy,
				    &phixx,&phixy,&phiyy);
  cp= cos(phi);
  sp= sin(phi);
  return R*(cp*sp*(phiyy-phixx)+(cp*cp-sp*sp)*phixy)-sp*phix+cp*phiy;
}
