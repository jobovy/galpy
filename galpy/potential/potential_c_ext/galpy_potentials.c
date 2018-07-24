#include <galpy_potentials.h>
void cyl_to_rect(double R, double phi,double *x, double *y){
  *x= R * cos ( phi );
  *y= R * sin ( phi );
}
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
  }
}
void free_potentialArgs(int npot, struct potentialArg * potentialArgs){
  int ii;
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
double calcRforce(double R, double Z, double phi, double t, 
		  int nargs, struct potentialArg * potentialArgs){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    Rforce+= potentialArgs->Rforce(R,Z,phi,t,
				   potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return Rforce;
}
double calczforce(double R, double Z, double phi, double t, 
		  int nargs, struct potentialArg * potentialArgs){
  int ii;
  double zforce= 0.;
  for (ii=0; ii < nargs; ii++){
    zforce+= potentialArgs->zforce(R,Z,phi,t,
				   potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return zforce;
}
double calcPhiforce(double R, double Z, double phi, double t, 
			  int nargs, struct potentialArg * potentialArgs){
  int ii;
  double phiforce= 0.;
  for (ii=0; ii < nargs; ii++){
    phiforce+= potentialArgs->phiforce(R,Z,phi,t,
				       potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return phiforce;
}
double calcPlanarRforce(double R, double phi, double t, 
			int nargs, struct potentialArg * potentialArgs){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    Rforce+= potentialArgs->planarRforce(R,phi,t,
					 potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return Rforce;
}
double calcPlanarphiforce(double R, double phi, double t, 
			  int nargs, struct potentialArg * potentialArgs){
  int ii;
  double phiforce= 0.;
  for (ii=0; ii < nargs; ii++){
    phiforce+= potentialArgs->planarphiforce(R,phi,t,
					     potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return phiforce;
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
