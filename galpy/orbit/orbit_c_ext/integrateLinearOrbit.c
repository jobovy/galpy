/*
  Wrappers around the C integration code for linear Orbits
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <bovy_symplecticode.h>
#include <bovy_rk.h>
#include <leung_dop853.h>
//Potentials
#include <galpy_potentials.h>
#include <integrateFullOrbit.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//Macros to export functions in DLL on different OS
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
// Just do nothing?
#define EXPORT
#endif
/*
  Function Declarations
*/
void evalLinearForce(double, double *, double *,
		     int, struct potentialArg *);
void evalLinearDeriv(double, double *, double *,
		     int, struct potentialArg *);
/*
  Actual functions
*/
void parse_leapFuncArgs_Linear(int npot,struct potentialArg * potentialArgs,
			       int ** pot_type,
			       double ** pot_args){
  int ii,jj;
  init_potentialArgs(npot,potentialArgs);
  for (ii=0; ii < npot; ii++){
    switch ( *(*pot_type)++ ) {
    default: //verticalPotential
      potentialArgs->linearForce= &verticalPotentialLinearForce;
      break;
    case 31: // KGPotential
      potentialArgs->linearForce= &KGPotentialLinearForce;
      potentialArgs->nargs= 4;
      break; 
//////////////////////////////// WRAPPERS /////////////////////////////////////
      // NOT CURRENTLY SUPPORTED
      /*
    case -1: //DehnenSmoothWrapperPotential
      potentialArgs->linearForce= &DehnenSmoothWrapperPotentialPlanarRforce;
      potentialArgs->nargs= (int) 3;
      break;
    case -2: //SolidBodyRotationWrapperPotential
      potentialArgs->linearForce= &SolidBodyRotationWrapperPotentialPlanarRforce;
      potentialArgs->nargs= (int) 3;
      break;
    case -4: //CorotatingRotationWrapperPotential
      potentialArgs->linearForce= &CorotatingRotationWrapperPotentialPlanarRforce;
      potentialArgs->nargs= (int) 5;
      break;
    case -5: //GaussianAmplitudeWrapperPotential
      potentialArgs->linearForce= &GaussianAmplitudeWrapperPotentialPlanarRforce;
      potentialArgs->nargs= (int) 3;
      break;
      */
    }
    /*
    if ( *(*pot_type-1) < 0) { // Parse wrapped potential for wrappers
      potentialArgs->nwrapped= (int) *(*pot_args)++;
      potentialArgs->wrappedPotentialArg= \
	(struct potentialArg *) malloc ( potentialArgs->nwrapped	\
					 * sizeof (struct potentialArg) );
      parse_leapFuncArgs_Linear(potentialArgs->nwrapped,
				potentialArgs->wrappedPotentialArg,
				pot_type,pot_args);
    }
      */
    // linear from 3D: assign R location parameter as the only one, rest
    // of potential as wrapped
    if ( potentialArgs->linearForce == &verticalPotentialLinearForce ) {
      potentialArgs->nwrapped= (int) 1;
      potentialArgs->wrappedPotentialArg= \
	(struct potentialArg *) malloc ( potentialArgs->nwrapped	\
					 * sizeof (struct potentialArg) );
      *(pot_type)-= 1; // Do FullOrbit processing for same potential
      parse_leapFuncArgs_Full(potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg,
			      pot_type,pot_args);
      potentialArgs->nargs= 2; // R, phi
    }
    potentialArgs->args= (double *) malloc( potentialArgs->nargs * sizeof(double));
    for (jj=0; jj < potentialArgs->nargs; jj++){
      *(potentialArgs->args)= *(*pot_args)++;
      potentialArgs->args++;
    }
    potentialArgs->args-= potentialArgs->nargs;
    potentialArgs++;
  }
  potentialArgs-= npot;
}
EXPORT void integrateLinearOrbit(double *yo,
				 int nt, 
				 double *t,
				 int npot,
				 int * pot_type,
				 double * pot_args,
				 double dt,
				 double rtol,
				 double atol,
				 double *result,
				 int * err,
				 int odeint_type){
  //Set up the forces, first count
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Linear(npot,potentialArgs,&pot_type,&pot_args);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalLinearForce;
    dim= 1;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalLinearDeriv;
    dim= 2;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalLinearDeriv;
    dim= 2;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalLinearForce;
    dim= 1;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalLinearForce;
    dim= 1;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalLinearDeriv;
    dim= 2;
    break;
  case 6: //DOP853
    odeint_func= &dop853;
    odeint_deriv_func= &evalLinearDeriv;
    dim= 2;
    break;
  }
  odeint_func(odeint_deriv_func,dim,yo,nt,dt,t,npot,potentialArgs,rtol,atol,
	      result,err);
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}

void evalLinearForce(double t, double *q, double *a,
		     int nargs, struct potentialArg * potentialArgs){
  *a= calcLinearForce(*q,t,nargs,potentialArgs);
}
void evalLinearDeriv(double t, double *q, double *a,
		     int nargs, struct potentialArg * potentialArgs){
  *a++= *(q+1);
  *a= calcLinearForce(*q,t,nargs,potentialArgs);
}
