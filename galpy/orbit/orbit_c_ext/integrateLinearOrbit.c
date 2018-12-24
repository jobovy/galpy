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
void symplec_drift_1d(double, double *);
void symplec_kick_1d(double,double,double *,int,struct potentialArg *);
void tol_scaling_1d(double *,double *);
void compare_metric_1d(int,double *,double *,double *);
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
  int dim= 2;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Linear(npot,potentialArgs,&pot_type,&pot_args);
  //Integrate
  if ( odeint_type == 0 || odeint_type == 3 || odeint_type == 4 ) { // symplec
    void (*odeint_func)(void (*drift)(double, double *),
			void  (*kick)(double, double, double *,
				      int, struct potentialArg *),
			int,
			double *,
			int, double, double *,
			int, struct potentialArg *,
			double, double,
			void (*tol_scaling)(double *,double *),
			void (*metric)(int,double *, double *,double *),
			bool,
			double *,int *);
    void (*odeint_drift_func)(double, double *);
    odeint_drift_func= &symplec_drift_1d;
    void (*odeint_kick_func)(double, double, double *,
			     int,struct potentialArg *);
    odeint_kick_func= &symplec_kick_1d;
    // Scaling of initial condition for stepsize determination
    void (*tol_scaling)(double *yo,double *result);
    tol_scaling= &tol_scaling_1d;
    // Metric for comparing solutions with different stepsizes
    void (*metric)(int dim, double *x, double *y,double *result);
    metric= &compare_metric_1d;
    switch ( odeint_type ) {
    case 0: //leapfrog
      odeint_func= &leapfrog;
      break;
      case 3: //symplec4
      odeint_func= &symplec4;
      break;
      case 4: //symplec6
      odeint_func= &symplec6;
      break;
    }
    odeint_func(odeint_drift_func,odeint_kick_func,
		dim,yo,nt,dt,t,npot,potentialArgs,rtol,atol,
		tol_scaling,metric,false,result,err);
  }
  else { // non-symplec
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
    case 1: //RK4
      odeint_func= &bovy_rk4;
      odeint_deriv_func= &evalLinearDeriv;
      break;
    case 2: //RK6
      odeint_func= &bovy_rk6;
      odeint_deriv_func= &evalLinearDeriv;
      break;
    case 5: //DOPR54
      odeint_func= &bovy_dopr54;
      odeint_deriv_func= &evalLinearDeriv;
      break;
    case 6: //DOP853
      odeint_func= &dop853;
      odeint_deriv_func= &evalLinearDeriv;
      break;
    }
    odeint_func(odeint_deriv_func,dim,yo,nt,dt,t,npot,potentialArgs,rtol,atol,
		result,err);
  }
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}

void symplec_drift_1d(double dt, double *y){
  *y    += dt * *(y+1);
}
void symplec_kick_1d(double dt, double t, double *y,
		     int nargs, struct potentialArg * potentialArgs){
  *(y+1)+= dt * calcLinearForce(*y,t,nargs,potentialArgs);
}
void tol_scaling_1d(double *yo,double *result){
  *(result+0)= fabs( *yo );
  *(result+1)= fabs( *(yo+1) );
}
void compare_metric_1d(int dim, double *x, double *y, double *result){
  int ii;
  for (ii=0; ii < dim; ii++)
    *(result+ii)= fabs( *(x+ii) - *(y+ii) );
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
