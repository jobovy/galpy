#include <galpy_potentials.h>
void init_potentialArgs(int npot, struct potentialArg * potentialArgs){
  int ii;
  for (ii=0; ii < npot; ii++) {
    (potentialArgs+ii)->i2d= NULL;
    (potentialArgs+ii)->accx= NULL;
    (potentialArgs+ii)->accy= NULL;
    (potentialArgs+ii)->i2drforce= NULL;
    (potentialArgs+ii)->accxrforce= NULL;
    (potentialArgs+ii)->accyrforce= NULL;
    (potentialArgs+ii)->i2dzforce= NULL;
    (potentialArgs+ii)->accxzforce= NULL;
    (potentialArgs+ii)->accyzforce= NULL;
    (potentialArgs+ii)->wrappedPotentialArg= NULL;
    (potentialArgs+ii)->spline1d= NULL;
    (potentialArgs+ii)->acc1d= NULL;
    (potentialArgs+ii)->tfuncs= NULL;
  }
}
void free_potentialArgs(int npot, struct potentialArg * potentialArgs){
  int ii, jj;
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->i2d )
      interp_2d_free((potentialArgs+ii)->i2d) ;
    if ( (potentialArgs+ii)->accx )
      gsl_interp_accel_free ((potentialArgs+ii)->accx);
    if ( (potentialArgs+ii)->accy )
      gsl_interp_accel_free ((potentialArgs+ii)->accy);
    if ( (potentialArgs+ii)->i2drforce )
      interp_2d_free((potentialArgs+ii)->i2drforce) ;
    if ( (potentialArgs+ii)->accxrforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accxrforce);
    if ( (potentialArgs+ii)->accyrforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accyrforce);
    if ( (potentialArgs+ii)->i2dzforce )
      interp_2d_free((potentialArgs+ii)->i2dzforce) ;
    if ( (potentialArgs+ii)->accxzforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accxzforce);
    if ( (potentialArgs+ii)->accyzforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accyzforce);
    if ( (potentialArgs+ii)->wrappedPotentialArg ) {
      free_potentialArgs((potentialArgs+ii)->nwrapped,
			 (potentialArgs+ii)->wrappedPotentialArg);
      free((potentialArgs+ii)->wrappedPotentialArg);
    }
    if ( (potentialArgs+ii)->spline1d ) {
      for (jj=0; jj < (potentialArgs+ii)->nspline1d; jj++)
	gsl_spline_free(*((potentialArgs+ii)->spline1d+jj));
      free((potentialArgs+ii)->spline1d);
    }
    if ( (potentialArgs+ii)->acc1d ) {
      for (jj=0; jj < (potentialArgs+ii)->nspline1d; jj++)
	gsl_interp_accel_free (*((potentialArgs+ii)->acc1d+jj));
      free((potentialArgs+ii)->acc1d);
    }
    free((potentialArgs+ii)->args);
  }
}
double evaluatePotentials(double R, double Z,
			  int nargs, struct potentialArg * potentialArgs){
  int ii;
  double pot= 0.;
  for (ii=0; ii < nargs; ii++){
    pot+= potentialArgs->potentialEval(R,Z,0.,0.,
				       potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return pot;
}
// function name in parentheses, because actual function defined by macro
// in galpy_potentials.h and parentheses are necessary to avoid macro expansion
double (calcRforce)(double R, double Z, double phi, double t,
		    int nargs, struct potentialArg * potentialArgs,
		    double vR, double vT, double vZ){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
      Rforce+= potentialArgs->RforceVelocity(R,Z,phi,t,potentialArgs,vR,vT,vZ);
    else
      Rforce+= potentialArgs->Rforce(R,Z,phi,t,
				     potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return Rforce;
}
double (calczforce)(double R, double Z, double phi, double t,
		    int nargs, struct potentialArg * potentialArgs,
		    double vR, double vT, double vZ){
  int ii;
  double zforce= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
      zforce+= potentialArgs->zforceVelocity(R,Z,phi,t,potentialArgs,vR,vT,vZ);
    else
      zforce+= potentialArgs->zforce(R,Z,phi,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return zforce;
}
double (calcphitorque)(double R, double Z, double phi, double t,
		      int nargs, struct potentialArg * potentialArgs,
		      double vR, double vT, double vZ){
  int ii;
  double phitorque= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
      phitorque+= potentialArgs->phitorqueVelocity(R,Z,phi,t,potentialArgs,
						 vR,vT,vZ);
    else
      phitorque+= potentialArgs->phitorque(R,Z,phi,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return phitorque;
}
double (calcPlanarRforce)(double R, double phi, double t,
			int nargs, struct potentialArg * potentialArgs,
            double vR, double vT){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
        Rforce+= potentialArgs->planarRforceVelocity(R,phi,t,potentialArgs,vR,vT);
    else
        Rforce+= potentialArgs->planarRforce(R,phi,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return Rforce;
}
double (calcPlanarphitorque)(double R, double phi, double t,
			  int nargs, struct potentialArg * potentialArgs,
              double vR, double vT){
  int ii;
  double phitorque= 0.;
  for (ii=0; ii < nargs; ii++){
    if ( potentialArgs->requiresVelocity )
        phitorque+= potentialArgs->planarphitorqueVelocity(R,phi,t,potentialArgs,vR,vT);
    else
        phitorque+= potentialArgs->planarphitorque(R,phi,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return phitorque;
}

// LCOV_EXCL_START
double calcR2deriv(double R, double Z, double phi, double t,
		   int nargs, struct potentialArg * potentialArgs){
  int ii;
  double R2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    R2deriv+= potentialArgs->R2deriv(R,Z,phi,t,
				     potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return R2deriv;
}

double calcphi2deriv(double R, double Z, double phi, double t,
			 int nargs, struct potentialArg * potentialArgs){
  int ii;
  double phi2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    phi2deriv+= potentialArgs->phi2deriv(R,Z,phi,t,
					 potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return phi2deriv;
}
double calcRphideriv(double R, double Z, double phi, double t,
			   int nargs, struct potentialArg * potentialArgs){
  int ii;
  double Rphideriv= 0.;
  for (ii=0; ii < nargs; ii++){
    Rphideriv+= potentialArgs->Rphideriv(R,Z,phi,t,
					 potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return Rphideriv;
}
// LCOV_EXCL_STOP
double calcPlanarR2deriv(double R, double phi, double t,
			 int nargs, struct potentialArg * potentialArgs){
  int ii;
  double R2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    R2deriv+= potentialArgs->planarR2deriv(R,phi,t,
					   potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return R2deriv;
}

double calcPlanarphi2deriv(double R, double phi, double t,
			 int nargs, struct potentialArg * potentialArgs){
  int ii;
  double phi2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    phi2deriv+= potentialArgs->planarphi2deriv(R,phi,t,
					       potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return phi2deriv;
}
double calcPlanarRphideriv(double R, double phi, double t,
			 int nargs, struct potentialArg * potentialArgs){
  int ii;
  double Rphideriv= 0.;
  for (ii=0; ii < nargs; ii++){
    Rphideriv+= potentialArgs->planarRphideriv(R,phi,t,
					       potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return Rphideriv;
}
double calcLinearForce(double x, double t,
		       int nargs, struct potentialArg * potentialArgs){
  int ii;
  double force= 0.;
  for (ii=0; ii < nargs; ii++){
    force+= potentialArgs->linearForce(x,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return force;
}
double calcDensity(double R, double Z, double phi, double t,
		   int nargs, struct potentialArg * potentialArgs){
  int ii;
  double dens= 0.;
  for (ii=0; ii < nargs; ii++){
    dens+= potentialArgs->dens(R,Z,phi,t,potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return dens;
}
void rotate(double *x, double *y, double *z, double *rot){
  double xp,yp,zp;
  xp= *(rot)   * *x + *(rot+1) * *y + *(rot+2) * *z;
  yp= *(rot+3) * *x + *(rot+4) * *y + *(rot+5) * *z;
  zp= *(rot+6) * *x + *(rot+7) * *y + *(rot+8) * *z;
  *x= xp;
  *y= yp;
  *z= zp;
}
void rotate_force(double *Fx, double *Fy, double *Fz,
				double *rot){
  double Fxp,Fyp,Fzp;
  Fxp= *(rot)   * *Fx + *(rot+3) * *Fy + *(rot+6) * *Fz;
  Fyp= *(rot+1) * *Fx + *(rot+4) * *Fy + *(rot+7) * *Fz;
  Fzp= *(rot+2) * *Fx + *(rot+5) * *Fy + *(rot+8) * *Fz;
  *Fx= Fxp;
  *Fy= Fyp;
  *Fz= Fzp;
}
