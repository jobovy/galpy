/*
  Wrappers around the C integration code for planar Orbits
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <bovy_symplecticode.h>
#include <bovy_rk.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
  Function Declarations
*/
void evalPlanarRectForce(double, double *, double *,
			 int, struct potentialArg *);
void evalPlanarRectDeriv(double, double *, double *,
			 int, struct potentialArg *);
void evalPlanarRectDeriv_dxdv(double, double *, double *,
			      int, struct potentialArg *);
double calcPlanarRforce(double, double, double, 
			int, struct potentialArg *);
double calcPlanarphiforce(double, double, double, 
			int, struct potentialArg *);
double calcPlanarR2deriv(double, double, double, 
			 int, struct potentialArg *);
double calcPlanarphi2deriv(double, double, double, 
			   int, struct potentialArg *);
double calcPlanarRphideriv(double, double, double, 
			   int, struct potentialArg *);
/*
  Actual functions
*/
inline void parse_leapFuncArgs(int npot,struct potentialArg * potentialArgs,
			       int * pot_type,
			       double * pot_args){
  int ii,jj;
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 2 arguments
      potentialArgs->planarRforce= &LogarithmicHaloPotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &LogarithmicHaloPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      break;
    case 1: //DehnenBarPotential, 7 arguments
      potentialArgs->planarRforce= &DehnenBarPotentialRforce;
      potentialArgs->planarphiforce= &DehnenBarPotentialphiforce;
      potentialArgs->planarR2deriv= &DehnenBarPotentialR2deriv;
      potentialArgs->planarphi2deriv= &DehnenBarPotentialphi2deriv;
      potentialArgs->planarRphideriv= &DehnenBarPotentialRphideriv;
      potentialArgs->nargs= 7;
      break;
    case 2: //TransientLogSpiralPotential, 8 arguments
      potentialArgs->planarRforce= &TransientLogSpiralPotentialRforce;
      potentialArgs->planarphiforce= &TransientLogSpiralPotentialphiforce;
      potentialArgs->nargs= 8;
      break;
    case 3: //SteadyLogSpiralPotential, 8 arguments
      potentialArgs->planarRforce= &SteadyLogSpiralPotentialRforce;
      potentialArgs->planarphiforce= &SteadyLogSpiralPotentialphiforce;
      potentialArgs->nargs= 8;
      break;
    case 4: //EllipticalDiskPotential, 6 arguments
      potentialArgs->planarRforce= &EllipticalDiskPotentialRforce;
      potentialArgs->planarphiforce= &EllipticalDiskPotentialphiforce;
      potentialArgs->planarR2deriv= &EllipticalDiskPotentialR2deriv;
      potentialArgs->planarphi2deriv= &EllipticalDiskPotentialphi2deriv;
      potentialArgs->planarRphideriv= &EllipticalDiskPotentialRphideriv;
      potentialArgs->nargs= 6;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->planarRforce= &MiyamotoNagaiPotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &MiyamotoNagaiPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      break;
    case 6: //LopsidedDiskPotential, 6 arguments
      potentialArgs->planarRforce= &LopsidedDiskPotentialRforce;
      potentialArgs->planarphiforce= &LopsidedDiskPotentialphiforce;
      potentialArgs->planarR2deriv= &LopsidedDiskPotentialR2deriv;
      potentialArgs->planarphi2deriv= &LopsidedDiskPotentialphi2deriv;
      potentialArgs->planarRphideriv= &LopsidedDiskPotentialRphideriv;
      potentialArgs->nargs= 6;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      potentialArgs->planarRforce= &PowerSphericalPotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &PowerSphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      potentialArgs->planarRforce= &HernquistPotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &HernquistPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->planarRforce= &NFWPotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &NFWPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      potentialArgs->planarRforce= &JaffePotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &JaffePotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      potentialArgs->planarRforce= &DoubleExponentialDiskPotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      //potentialArgs->planarR2deriv= &DoubleExponentialDiskPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      //Look at pot_args to figure out the number of arguments
      potentialArgs->nargs= (int) (8 + 2 * *(pot_args+5) + 4 * ( *(pot_args+4) + 1 ));
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      potentialArgs->planarRforce= &FlattenedPowerPotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &FlattenedPowerPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      break;
    case 14: //IsochronePotential, 2 arguments
      potentialArgs->planarRforce= &IsochronePotentialPlanarRforce;
      potentialArgs->planarphiforce= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &IsochronePotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      break;
    }
    potentialArgs->args= (double *) malloc( potentialArgs->nargs * sizeof(double));
    for (jj=0; jj < potentialArgs->nargs; jj++){
      *(potentialArgs->args)= *pot_args++;
      potentialArgs->args++;
    }
    potentialArgs->args-= potentialArgs->nargs;
    potentialArgs++;
  }
  potentialArgs-= npot;
}
void integratePlanarOrbit(double *yo,
			  int nt, 
			  double *t,
			  int npot,
			  int * pot_type,
			  double * pot_args,
			  double rtol,
			  double atol,
			  double *result,
			  int * err,
			  int odeint_type){
  //Set up the forces, first count
  int ii;
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs(npot,potentialArgs,pot_type,pot_args);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  }
  odeint_func(odeint_deriv_func,dim,yo,nt,t,npot,potentialArgs,rtol,atol,
	      result,err);
  //Free allocated memory
  for (ii=0; ii < npot; ii++) {
    free(potentialArgs->args);
    potentialArgs++;
  }
  potentialArgs-= npot;
  free(potentialArgs);
  //Done!
}

void integratePlanarOrbit_dxdv(double *yo,
			       int nt, 
			       double *t,
			       int npot,
			       int * pot_type,
			       double * pot_args,
			       double rtol,
			       double atol,
			       double *result,
			       int * err,
			       int odeint_type){
  //Set up the forces, first count
  int ii;
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs(npot,potentialArgs,pot_type,pot_args);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 4;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalPlanarRectDeriv_dxdv;
    dim= 8;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalPlanarRectDeriv_dxdv;
    dim= 8;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 4;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 4;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalPlanarRectDeriv_dxdv;
    dim= 8;
    break;
  }
  odeint_func(odeint_deriv_func,dim,yo,nt,t,npot,potentialArgs,rtol,atol,
	      result,err);
  //Free allocated memory
  for (ii=0; ii < npot; ii++) {
    free(potentialArgs->args);
    potentialArgs++;
  }
  potentialArgs-= npot;
  free(potentialArgs);
  //Done!
}

void evalPlanarRectForce(double t, double *q, double *a,
			 int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce;
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,potentialArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a--= sinphi*Rforce+1./R*cosphi*phiforce;
}
void evalPlanarRectDeriv(double t, double *q, double *a,
			 int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce;
  //first two derivatives are just the velocities
  *a++= *(q+2);
  *a++= *(q+3);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,potentialArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a= sinphi*Rforce+1./R*cosphi*phiforce;
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

void evalPlanarRectDeriv_dxdv(double t, double *q, double *a,
			      int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce;
  double R2deriv, phi2deriv, Rphideriv, dFxdx, dFxdy, dFydx, dFydy;
  //first two derivatives are just the velocities
  *a++= *(q+2);
  *a++= *(q+3);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,potentialArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  //dx derivatives are just dv
  *a++= *(q+6);
  *a++= *(q+7);
  //for the dv derivatives we need also R2deriv, phi2deriv, and Rphideriv
  R2deriv= calcPlanarR2deriv(R,phi,t,nargs,potentialArgs);
  phi2deriv= calcPlanarphi2deriv(R,phi,t,nargs,potentialArgs);
  Rphideriv= calcPlanarRphideriv(R,phi,t,nargs,potentialArgs);
  //..and dFxdx, dFxdy, dFydx, dFydy
  dFxdx= -cosphi*cosphi*R2deriv
    +2.*cosphi*sinphi/R/R*phiforce
    +sinphi*sinphi/R*Rforce
    +2.*sinphi*cosphi/R*Rphideriv
    -sinphi*sinphi/R/R*phi2deriv;
  dFxdy= -sinphi*cosphi*R2deriv
    +(sinphi*sinphi-cosphi*cosphi)/R/R*phiforce
    -cosphi*sinphi/R*Rforce
    -(cosphi*cosphi-sinphi*sinphi)/R*Rphideriv
    +cosphi*sinphi/R/R*phi2deriv;
  dFydx= -cosphi*sinphi*R2deriv
    +(sinphi*sinphi-cosphi*cosphi)/R/R*phiforce
    +(sinphi*sinphi-cosphi*cosphi)/R*Rphideriv
    -sinphi*cosphi/R*Rforce
    +sinphi*cosphi/R/R*phi2deriv;
  dFydy= -sinphi*sinphi*R2deriv
    -2.*sinphi*cosphi/R/R*phiforce
    -2.*sinphi*cosphi/R*Rphideriv
    +cosphi*cosphi/R*Rforce
    -cosphi*cosphi/R/R*phi2deriv;
  *a++= dFxdx * *(q+4) + dFxdy * *(q+5);
  *a= dFydx * *(q+4) + dFydy * *(q+5);
}

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
