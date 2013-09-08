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
		   int, struct potentialArg *);
void evalRectDeriv(double, double *, double *,
			 int, struct potentialArg *);
void evalRectDeriv_dxdv(double,double *, double *,
			      int, struct potentialArg *);
double calcRforce(double, double,double, double, 
			int, struct potentialArg *);
double calczforce(double, double,double, double, 
			int, struct potentialArg *);
double calcPhiforce(double, double,double, double, 
			int, struct potentialArg *);
double calcR2deriv(double, double, double,double, 
			 int, struct potentialArg *);
double calcphi2deriv(double, double, double,double, 
			   int, struct potentialArg *);
double calcRphideriv(double, double, double,double, 
			   int, struct potentialArg *);
/*
  Actual functions
*/
void parse_leapFuncArgs_Full(int npot,
			     struct potentialArg * potentialArgs,
			     int * pot_type,
			     double * pot_args){
  int ii,jj,kk,ll;
  int nR, nz;
  double * Rgrid, * zgrid, * potGrid_splinecoeffs, * row;
  for (ii=0; ii < npot; ii++){
    potentialArgs->i2drforce= NULL;
    potentialArgs->accrforce= NULL;
    potentialArgs->i2dzforce= NULL;
    potentialArgs->acczforce= NULL;
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 2 arguments
      potentialArgs->Rforce= &LogarithmicHaloPotentialRforce;
      potentialArgs->zforce= &LogarithmicHaloPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &LogarithmicHaloPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 3;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->Rforce= &MiyamotoNagaiPotentialRforce;
      potentialArgs->zforce= &MiyamotoNagaiPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &MiyamotoNagaiPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 3;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      potentialArgs->Rforce= &PowerSphericalPotentialRforce;
      potentialArgs->zforce= &PowerSphericalPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      potentialArgs->Rforce= &HernquistPotentialRforce;
      potentialArgs->zforce= &HernquistPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &HernquistPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->Rforce= &NFWPotentialRforce;
      potentialArgs->zforce= &NFWPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &NFWPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      potentialArgs->Rforce= &JaffePotentialRforce;
      potentialArgs->zforce= &JaffePotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //potentialArgs->R2deriv= &JaffePotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      potentialArgs->Rforce= &DoubleExponentialDiskPotentialRforce;
      potentialArgs->zforce= &DoubleExponentialDiskPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      //Look at pot_args to figure out the number of arguments
      potentialArgs->nargs= (int) (8 + 2 * *(pot_args+5) + 4 * ( *(pot_args+4) + 1 ));
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      potentialArgs->Rforce= &FlattenedPowerPotentialRforce;
      potentialArgs->zforce= &FlattenedPowerPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= 4;
      break;
    case 13: //interpRZPotential, XX arguments
      //Grab the grids and the coefficients
      nR= (int) *pot_args++;
      nz= (int) *pot_args++;
      Rgrid= (double *) malloc ( nR * sizeof ( double ) );
      zgrid= (double *) malloc ( nz * sizeof ( double ) );
      row= (double *) malloc ( nz * sizeof ( double ) );
      potGrid_splinecoeffs= (double *) malloc ( nR * nz * sizeof ( double ) );
      for (kk=0; kk < nR; kk++)
	*(Rgrid+kk)= *pot_args++;
      for (kk=0; kk < nz; kk++)
	*(zgrid+kk)= *pot_args++;
      for (kk=0; kk < nR; kk++){
	for (ll=0; ll < nz; ll++)
	  *(row+ll)= *pot_args++;
	put_row(potGrid_splinecoeffs,kk,row,nz); 
      }
      potentialArgs->i2drforce= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2drforce,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accrforce= gsl_interp_accel_alloc ();
      for (kk=0; kk < nR; kk++){
	for (ll=0; ll < nz; ll++)
	  *(row+ll)= *pot_args++;
	put_row(potGrid_splinecoeffs,kk,row,nz); 
      }
      potentialArgs->i2dzforce= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2dzforce,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->acczforce= gsl_interp_accel_alloc ();
      potentialArgs->Rforce= &interpRZPotentialRforce;
      potentialArgs->zforce= &interpRZPotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
      potentialArgs->nargs= 2;
      //clean up
      free(Rgrid);
      free(zgrid);
      free(row);
      free(potGrid_splinecoeffs);
      break;
    case 14: //IsochronePotential, 2 arguments
      potentialArgs->Rforce= &IsochronePotentialRforce;
      potentialArgs->zforce= &IsochronePotentialzforce;
      potentialArgs->phiforce= &ZeroForce;
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
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,pot_type,pot_args);
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
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,pot_type,pot_args);
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

void evalRectForce(double t, double *q, double *a,
		   int nargs, struct potentialArg * potentialArgs){
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
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs);
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a= zforce;
}
void evalRectDeriv(double t, double *q, double *a,
		   int nargs, struct potentialArg * potentialArgs){
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
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs);
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a= zforce;
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

void evalRectDeriv_dxdv(double t, double *q, double *a,
			int nargs, struct potentialArg * potentialArgs){
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
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs);
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs);
  phiforce= calcPhiforce(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a++= sinphi*Rforce+1./R*cosphi*phiforce;
  *a++= zforce;
  //dx derivatives are just dv
  *a++= *(q+9);
  *a++= *(q+10);
  *a++= *(q+11);
  //for the dv derivatives we need also R2deriv, phi2deriv, and Rphideriv
  R2deriv= calcR2deriv(R,z,phi,t,nargs,potentialArgs);
  phi2deriv= calcphi2deriv(R,z,phi,t,nargs,potentialArgs);
  Rphideriv= calcRphideriv(R,z,phi,t,nargs,potentialArgs);
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
