/*
  Wrappers around the C integration code for Full Orbits
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
void evalRectForce(double, double *, double *,
		   int, struct leapFuncArg *);
void evalRectDeriv(double, double *, double *,
			 int, struct leapFuncArg *);
void evalRectDeriv_dxdv(double,double *, double *,
			      int, struct leapFuncArg *);
double calcRforce(double, double,double, double, 
			int, struct leapFuncArg *);
double calczforce(double, double,double, double, 
			int, struct leapFuncArg *);
double calcPhiforce(double, double,double, double, 
			int, struct leapFuncArg *);
double calcR2deriv(double, double, double,double, 
			 int, struct leapFuncArg *);
double calcphi2deriv(double, double, double,double, 
			   int, struct leapFuncArg *);
double calcRphideriv(double, double, double,double, 
			   int, struct leapFuncArg *);
/*
  Actual functions
*/
inline void parse_leapFuncArgs_Full(int npot,
				    struct leapFuncArg * leapFuncArgs,
				    int * pot_type,
				    double * pot_args){
  int ii,jj;
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 2 arguments
      leapFuncArgs->Rforce= &LogarithmicHaloPotentialRforce;
      leapFuncArgs->zforce= &LogarithmicHaloPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      //leapFuncArgs->R2deriv= &LogarithmicHaloPotentialR2deriv;
      //leapFuncArgs->planarphi2deriv= &ZeroForce;
      //leapFuncArgs->planarRphideriv= &ZeroForce;
      leapFuncArgs->nargs= 3;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      leapFuncArgs->Rforce= &MiyamotoNagaiPotentialRforce;
      leapFuncArgs->zforce= &MiyamotoNagaiPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      //leapFuncArgs->R2deriv= &MiyamotoNagaiPotentialR2deriv;
      //leapFuncArgs->planarphi2deriv= &ZeroForce;
      //leapFuncArgs->planarRphideriv= &ZeroForce;
      leapFuncArgs->nargs= 3;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      leapFuncArgs->Rforce= &PowerSphericalPotentialRforce;
      leapFuncArgs->zforce= &PowerSphericalPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      //leapFuncArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      //leapFuncArgs->planarphi2deriv= &ZeroForce;
      //leapFuncArgs->planarRphideriv= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      leapFuncArgs->Rforce= &HernquistPotentialRforce;
      leapFuncArgs->zforce= &HernquistPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      //leapFuncArgs->R2deriv= &HernquistPotentialR2deriv;
      //leapFuncArgs->planarphi2deriv= &ZeroForce;
      //leapFuncArgs->planarRphideriv= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      leapFuncArgs->Rforce= &NFWPotentialRforce;
      leapFuncArgs->zforce= &NFWPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      //leapFuncArgs->R2deriv= &NFWPotentialR2deriv;
      //leapFuncArgs->planarphi2deriv= &ZeroForce;
      //leapFuncArgs->planarRphideriv= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      leapFuncArgs->Rforce= &JaffePotentialRforce;
      leapFuncArgs->zforce= &JaffePotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      //leapFuncArgs->R2deriv= &JaffePotentialR2deriv;
      //leapFuncArgs->planarphi2deriv= &ZeroForce;
      //leapFuncArgs->planarRphideriv= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      actionAngleArgs->Rforce= &DoubleExponentialDiskPotentialRforce;
      actionAngleArgs->zforce= &DoubleExponentialDiskPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      //Look at pot_args to figure out the number of arguments
      actionAngleArgs->nargs= 6 + 2 * *(pot_args+5) + 4 * *(pot_args+4);
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
void integrateFullOrbit(double *yo,
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
  parse_leapFuncArgs_Full(npot,leapFuncArgs,pot_type,pot_args);
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
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
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

void integrateOrbit_dxdv(double *yo,
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
  parse_leapFuncArgs_Full(npot,leapFuncArgs,pot_type,pot_args);
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
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
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

void evalRectForce(double t, double *q, double *a,
		   int nargs, struct leapFuncArg * leapFuncArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce, z, zforce;
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,leapFuncArgs);
  zforce= calczforce(R,z,phi,t,nargs,leapFuncArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a= zforce;
}
void evalRectDeriv(double t, double *q, double *a,
		   int nargs, struct leapFuncArg * leapFuncArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce,z,zforce;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,leapFuncArgs);
  zforce= calczforce(R,z,phi,t,nargs,leapFuncArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a= zforce;
}

double calcRforce(double R, double Z, double phi, double t, 
		  int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    Rforce+= leapFuncArgs->Rforce(R,Z,phi,t,
				  leapFuncArgs->nargs,
				  leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return Rforce;
}
double calczforce(double R, double Z, double phi, double t, 
		  int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double zforce= 0.;
  for (ii=0; ii < nargs; ii++){
    zforce+= leapFuncArgs->zforce(R,Z,phi,t,
				  leapFuncArgs->nargs,
				  leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return zforce;
}
double calcPhiforce(double R, double Z, double phi, double t, 
			  int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double phiforce= 0.;
  for (ii=0; ii < nargs; ii++){
    phiforce+= leapFuncArgs->phiforce(R,Z,phi,t,
				      leapFuncArgs->nargs,
				      leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return phiforce;
}

void evalRectDeriv_dxdv(double t, double *q, double *a,
			int nargs, struct leapFuncArg * leapFuncArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce,z,zforce;
  double R2deriv, phi2deriv, Rphideriv, dFxdx, dFxdy, dFydx, dFydy;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,leapFuncArgs);
  zforce= calczforce(R,z,phi,t,nargs,leapFuncArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a++= zforce;
  //dx derivatives are just dv
  *a++= *(q+9);
  *a++= *(q+10);
  *a++= *(q+11);
  //for the dv derivatives we need also R2deriv, phi2deriv, and Rphideriv
  R2deriv= calcR2deriv(R,z,phi,t,nargs,leapFuncArgs);
  phi2deriv= calcphi2deriv(R,z,phi,t,nargs,leapFuncArgs);
  Rphideriv= calcRphideriv(R,z,phi,t,nargs,leapFuncArgs);
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
  *a++= dFydx * *(q+4) + dFydy * *(q+5);
  *a= 0; //BOVY: PUT IN Z2DERIVS
}

double calcR2deriv(double R, double Z, double phi, double t, 
		   int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double R2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    R2deriv+= leapFuncArgs->R2deriv(R,Z,phi,t,
				    leapFuncArgs->nargs,
				    leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return R2deriv;
}

double calcphi2deriv(double R, double Z, double phi, double t, 
			 int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double phi2deriv= 0.;
  for (ii=0; ii < nargs; ii++){
    phi2deriv+= leapFuncArgs->phi2deriv(R,Z,phi,t,
					leapFuncArgs->nargs,
					leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return phi2deriv;
}
double calcRphideriv(double R, double Z, double phi, double t, 
			   int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double Rphideriv= 0.;
  for (ii=0; ii < nargs; ii++){
    Rphideriv+= leapFuncArgs->Rphideriv(R,Z,phi,t,
					leapFuncArgs->nargs,
					leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return Rphideriv;
}
