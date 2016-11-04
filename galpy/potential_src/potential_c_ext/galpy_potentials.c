#include <galpy_potentials.h>
void cyl_to_rect(double R, double phi,double *x, double *y){
  *x= R * cos ( phi );
  *y= R * sin ( phi );
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
