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
			 int, struct leapFuncArg *);
void evalPlanarRectDeriv(double, double *, double *,
			 int, struct leapFuncArg *);
void evalPlanarRectDeriv_dxdv(double, double *, double *,
			      int, struct leapFuncArg *);
double calcPlanarRforce(double, double, double, 
			int, struct leapFuncArg *);
double calcPlanarphiforce(double, double, double, 
			int, struct leapFuncArg *);
double calcPlanarR2deriv(double, double, double, 
			 int, struct leapFuncArg *);
double calcPlanarphi2deriv(double, double, double, 
			   int, struct leapFuncArg *);
double calcPlanarRphideriv(double, double, double, 
			   int, struct leapFuncArg *);
/*
  Actual functions
*/
inline void parse_leapFuncArgs(int npot,struct leapFuncArg * leapFuncArgs,
			       int * pot_type,
			       double * pot_args){
  int ii,jj;
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 2 arguments
      leapFuncArgs->planarRforce= &LogarithmicHaloPotentialPlanarRforce;
      leapFuncArgs->planarphiforce= &ZeroPlanarForce;
      leapFuncArgs->planarR2deriv= &LogarithmicHaloPotentialPlanarR2deriv;
      leapFuncArgs->planarphi2deriv= &ZeroPlanarForce;
      leapFuncArgs->planarRphideriv= &ZeroPlanarForce;
      leapFuncArgs->nargs= 2;
      break;
    case 1: //DehnenBarPotential, 7 arguments
      leapFuncArgs->planarRforce= &DehnenBarPotentialRforce;
      leapFuncArgs->planarphiforce= &DehnenBarPotentialphiforce;
      leapFuncArgs->planarR2deriv= &DehnenBarPotentialR2deriv;
      leapFuncArgs->planarphi2deriv= &DehnenBarPotentialphi2deriv;
      leapFuncArgs->planarRphideriv= &DehnenBarPotentialRphideriv;
      leapFuncArgs->nargs= 7;
      break;
    case 2: //TransientLogSpiralPotential, 8 arguments
      leapFuncArgs->planarRforce= &TransientLogSpiralPotentialRforce;
      leapFuncArgs->planarphiforce= &TransientLogSpiralPotentialphiforce;
      leapFuncArgs->nargs= 8;
      break;
    case 3: //SteadyLogSpiralPotential, 8 arguments
      leapFuncArgs->planarRforce= &SteadyLogSpiralPotentialRforce;
      leapFuncArgs->planarphiforce= &SteadyLogSpiralPotentialphiforce;
      leapFuncArgs->nargs= 8;
      break;
    case 4: //EllipticalDiskPotential, 6 arguments
      leapFuncArgs->planarRforce= &EllipticalDiskPotentialRforce;
      leapFuncArgs->planarphiforce= &EllipticalDiskPotentialphiforce;
      leapFuncArgs->planarR2deriv= &EllipticalDiskPotentialR2deriv;
      leapFuncArgs->planarphi2deriv= &EllipticalDiskPotentialphi2deriv;
      leapFuncArgs->planarRphideriv= &EllipticalDiskPotentialRphideriv;
      leapFuncArgs->nargs= 6;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      leapFuncArgs->planarRforce= &MiyamotoNagaiPotentialPlanarRforce;
      leapFuncArgs->planarphiforce= &ZeroPlanarForce;
      leapFuncArgs->planarR2deriv= &MiyamotoNagaiPotentialPlanarR2deriv;
      leapFuncArgs->planarphi2deriv= &ZeroPlanarForce;
      leapFuncArgs->planarRphideriv= &ZeroPlanarForce;
      leapFuncArgs->nargs= 3;
      break;
    case 6: //LopsidedDiskPotential, 6 arguments
      leapFuncArgs->planarRforce= &LopsidedDiskPotentialRforce;
      leapFuncArgs->planarphiforce= &LopsidedDiskPotentialphiforce;
      leapFuncArgs->planarR2deriv= &LopsidedDiskPotentialR2deriv;
      leapFuncArgs->planarphi2deriv= &LopsidedDiskPotentialphi2deriv;
      leapFuncArgs->planarRphideriv= &LopsidedDiskPotentialRphideriv;
      leapFuncArgs->nargs= 6;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      leapFuncArgs->planarRforce= &PowerSphericalPotentialPlanarRforce;
      leapFuncArgs->planarphiforce= &ZeroPlanarForce;
      leapFuncArgs->planarR2deriv= &PowerSphericalPotentialPlanarR2deriv;
      leapFuncArgs->planarphi2deriv= &ZeroPlanarForce;
      leapFuncArgs->planarRphideriv= &ZeroPlanarForce;
      leapFuncArgs->nargs= 2;
      break;
    }
    leapFuncArgs->args= (double *) malloc( leapFuncArgs->nargs * sizeof(double));
    for (jj=0; jj < leapFuncArgs->nargs; jj++){
      *(leapFuncArgs->args)= *pot_args++;
      leapFuncArgs->args++;
    }
    leapFuncArgs->args-= leapFuncArgs->nargs;
    leapFuncArgs++;
  }
  leapFuncArgs-= npot;
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
  struct leapFuncArg * leapFuncArgs= (struct leapFuncArg *) malloc ( npot * sizeof (struct leapFuncArg) );
  parse_leapFuncArgs(npot,leapFuncArgs,pot_type,pot_args);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct leapFuncArg *),
		      int,
		      double *,
		      int, double *,
		      int, struct leapFuncArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct leapFuncArg *);
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
  odeint_func(odeint_deriv_func,dim,yo,nt,t,npot,leapFuncArgs,rtol,atol,
	      result,err);
  //Free allocated memory
  for (ii=0; ii < npot; ii++) {
    free(leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= npot;
  free(leapFuncArgs);
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
  struct leapFuncArg * leapFuncArgs= (struct leapFuncArg *) malloc ( npot * sizeof (struct leapFuncArg) );
  parse_leapFuncArgs(npot,leapFuncArgs,pot_type,pot_args);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct leapFuncArg *),
		      int,
		      double *,
		      int, double *,
		      int, struct leapFuncArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct leapFuncArg *);
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
  odeint_func(odeint_deriv_func,dim,yo,nt,t,npot,leapFuncArgs,rtol,atol,
	      result,err);
  //Free allocated memory
  for (ii=0; ii < npot; ii++) {
    free(leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= npot;
  free(leapFuncArgs);
  //Done!
}

void evalPlanarRectForce(double t, double *q, double *a,
			 int nargs, struct leapFuncArg * leapFuncArgs){
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
  Rforce= calcPlanarRforce(R,phi,t,nargs,leapFuncArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a--= sinphi*Rforce+1./R*cosphi*phiforce;
}
void evalPlanarRectDeriv(double t, double *q, double *a,
			 int nargs, struct leapFuncArg * leapFuncArgs){
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
  Rforce= calcPlanarRforce(R,phi,t,nargs,leapFuncArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a= sinphi*Rforce+1./R*cosphi*phiforce;
}

double calcPlanarRforce(double R, double phi, double t, 
			int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    Rforce+= leapFuncArgs->planarRforce(R,phi,t,
					leapFuncArgs->nargs,
					leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return Rforce;
}
double calcPlanarphiforce(double R, double phi, double t, 
			  int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double phiforce= 0.;
  for (ii=0; ii < nargs; ii++){
    phiforce+= leapFuncArgs->planarphiforce(R,phi,t,
					    leapFuncArgs->nargs,
					    leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return phiforce;
}

void evalPlanarRectDeriv_dxdv(double t, double *q, double *a,
			      int nargs, struct leapFuncArg * leapFuncArgs){
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
  Rforce= calcPlanarRforce(R,phi,t,nargs,leapFuncArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  //dx derivatives are just dv
  *a++= *(q+6);
  *a++= *(q+7);
  //for the dv derivatives we need also R2deriv, phi2deriv, and Rphideriv
  R2deriv= calcPlanarR2deriv(R,phi,t,nargs,leapFuncArgs);
  phi2deriv= calcPlanarphi2deriv(R,phi,t,nargs,leapFuncArgs);
  Rphideriv= calcPlanarRphideriv(R,phi,t,nargs,leapFuncArgs);
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
			 int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double R2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    R2deriv+= leapFuncArgs->planarR2deriv(R,phi,t,
					  leapFuncArgs->nargs,
					  leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return R2deriv;
}

double calcPlanarphi2deriv(double R, double phi, double t, 
			 int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double phi2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    phi2deriv+= leapFuncArgs->planarphi2deriv(R,phi,t,
					  leapFuncArgs->nargs,
					  leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return phi2deriv;
}
double calcPlanarRphideriv(double R, double phi, double t, 
			 int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double Rphideriv= 0.;
  for (ii=0; ii < nargs; ii++){
    Rphideriv+= leapFuncArgs->planarRphideriv(R,phi,t,
					  leapFuncArgs->nargs,
					  leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return Rphideriv;
}
