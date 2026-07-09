/*
  Wrappers around the C integration code for linear Orbits
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <bovy_symplecticode.h>
#include <bovy_rk.h>
#include <wez_ias15.h>
#include <leung_dop853.h>
#include <integrateFullOrbit.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef ORBITS_CHUNKSIZE
#define ORBITS_CHUNKSIZE 1
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
void evalLinearDeriv_dxdv(double, double *, double *,
			  int, struct potentialArg *);
// Augmented force+Hessian evaluator (linear/1D) for the symplectic variational
// steppers leapfrog_dxdv/symplec4_dxdv/symplec6_dxdv (dim_base=1): the base
// acceleration and, per deviation column, the kick tangent K.dq_j with
// K= dF/dx= -linear2deriv (see integrateLinearOrbit_dxdv).
void evalLinearForce_dxdv(double, double *, double *,
			  int, struct potentialArg *, int);
/*
  Actual functions
*/
void parse_leapFuncArgs_Linear(int npot,struct potentialArg * potentialArgs,
			       int ** pot_type,
			       double ** pot_args,
             tfuncs_type_arr * pot_tfuncs){
  int ii,jj;
  init_potentialArgs(npot,potentialArgs);
  for (ii=0; ii < npot; ii++){
    switch ( *(*pot_type)++ ) {
    default: //verticalPotential
      potentialArgs->linearForce= &verticalPotentialLinearForce;
      potentialArgs->linear2deriv= &verticalPotentialLinear2deriv;
      break;
    case 31: // KGPotential
      potentialArgs->linearForce= &KGPotentialLinearForce;
      potentialArgs->linear2deriv= &KGPotentialLinear2deriv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      break;
    case 32: // IsothermalDiskPotential
      potentialArgs->linearForce= &IsothermalDiskPotentialLinearForce;
      potentialArgs->linear2deriv= &IsothermalDiskPotentialLinear2deriv;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
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
			      pot_type,pot_args,pot_tfuncs);
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
EXPORT void integrateLinearOrbit(int nobj,
				 double *yo,
				 int nt,
				 double *t,
				 int indiv_t,
				 int npot,
				 int * pot_type,
				 double * pot_args,
         tfuncs_type_arr pot_tfuncs,
				 double dt,
				 double rtol,
				 double atol,
				 double *result,
				 int * err,
				 int odeint_type,
         orbint_callback_type cb){
  //Set up the forces, first count
  int dim;
  int ii;
  int max_threads;
  int * thread_pot_type;
  double * thread_pot_args;
  tfuncs_type_arr thread_pot_tfuncs;
  max_threads= ( nobj < omp_get_max_threads() ) ? nobj : omp_get_max_threads();
  // Because potentialArgs may cache, safest to have one / thread
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( max_threads * npot * sizeof (struct potentialArg) );
#pragma omp parallel for schedule(static,1) private(ii,thread_pot_type,thread_pot_args,thread_pot_tfuncs) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++) {
    thread_pot_type= pot_type; // need to make thread-private pointers, bc
    thread_pot_args= pot_args; // these pointers are changed in parse_...
    thread_pot_tfuncs= pot_tfuncs; // ...
    parse_leapFuncArgs_Linear(npot,potentialArgs+ii*npot,
			      &thread_pot_type,&thread_pot_args,&thread_pot_tfuncs);
  }
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
  case 7: //ias15
    odeint_func= &wez_ias15;
    odeint_deriv_func= &evalLinearForce;
    dim= 1;
    break;
  }
#pragma omp parallel for schedule(dynamic,ORBITS_CHUNKSIZE) private(ii) num_threads(max_threads)
  for (ii=0; ii < nobj; ii++) {
    odeint_func(odeint_deriv_func,dim,yo+2*ii,nt,dt,t+nt*ii*indiv_t,
		npot,potentialArgs+omp_get_thread_num()*npot,rtol,atol,
		result+2*nt*ii,err+ii);
    if ( cb ) // Callback if not void
      cb();
  }
  //Free allocated memory
#pragma omp parallel for schedule(static,1) private(ii) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++)
    free_potentialArgs(npot,potentialArgs+ii*npot);
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

/*
  1D (linear) variational (state-transition/dxdv) integration. Mirrors
  integratePlanarOrbit_dxdv: the RK integrators propagate the 4D state
  [x,v,dx,dv] via the variational RHS evalLinearDeriv_dxdv (dim=4); the
  symplectic ones (leapfrog=0/symplec4=3/symplec6=4) carry the deviation
  through the closed-form drift/kick tangent maps of the shared *_dxdv steppers
  in bovy_symplecticode.c (dim_base=1, nde=1 deviation column). ias15 has no
  dxdv path and is blocked upstream by Orbit.integrate_dxdv (check_integrator).
  There is no cyl<->rect transform in 1D, so the deviation is the raw [dx,dv].
*/
EXPORT void integrateLinearOrbit_dxdv(double *yo,
				      int nt,
				      double *t,
				      int npot,
				      int * pot_type,
				      double * pot_args,
				      tfuncs_type_arr pot_tfuncs,
				      double dt,
				      double rtol,
				      double atol,
				      double *result,
				      int * err,
				      int odeint_type){
  //Set up the forces
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Linear(npot,potentialArgs,&pot_type,&pot_args,&pot_tfuncs);
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
  odeint_func= NULL;
  odeint_deriv_func= &evalLinearDeriv_dxdv;
  dim= 4;
  switch ( odeint_type ) {
  case 1: //RK4
    odeint_func= &bovy_rk4;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    break;
  case 6: //DOP853
    odeint_func= &dop853;
    break;
  }
  switch ( odeint_type ) {
  case 0: //leapfrog
    leapfrog_dxdv(&evalLinearForce_dxdv,&evalLinearForce,1,1,yo,nt,dt,t,
		  npot,potentialArgs,rtol,atol,result,err);
    break;
  case 3: //symplec4
    symplec4_dxdv(&evalLinearForce_dxdv,&evalLinearForce,1,1,yo,nt,dt,t,
		  npot,potentialArgs,rtol,atol,result,err);
    break;
  case 4: //symplec6
    symplec6_dxdv(&evalLinearForce_dxdv,&evalLinearForce,1,1,yo,nt,dt,t,
		  npot,potentialArgs,rtol,atol,result,err);
    break;
  default: //RK method selected above
    odeint_func(odeint_deriv_func,dim,yo,nt,dt,t,npot,potentialArgs,rtol,atol,
		result,err);
  }
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}

// RK variational RHS, state dim=4 [x,v,dx,dv]: base [v,F] and deviation
// [dv, K.dx] with K= dF/dx= -linear2deriv (mirror evalPlanarRectDeriv_dxdv).
void evalLinearDeriv_dxdv(double t, double *q, double *a,
			  int nargs, struct potentialArg * potentialArgs){
  *a++= *(q+1);
  *a++= calcLinearForce(*q,t,nargs,potentialArgs);
  *a++= *(q+3);
  *a= -calcLinear2deriv(*q,t,nargs,potentialArgs) * *(q+2);
}

// Augmented force for the symplectic variational steppers (dim_base=1): fill
// the base acceleration (block 0, byte-identical to evalLinearForce) and, per
// deviation column j, the kick tangent K.dq_j with K= dF/dx= -linear2deriv.
void evalLinearForce_dxdv(double t, double *q, double *a,
			  int nargs, struct potentialArg * potentialArgs,
			  int nde){
  int jj;
  double K;
  //Base acceleration: identical call to evalLinearForce (bit-identical base)
  *a= calcLinearForce(*q,t,nargs,potentialArgs);
  if ( nde == 0 ) return;
  K= -calcLinear2deriv(*q,t,nargs,potentialArgs);
  //Kick tangent per deviation column: dv_j += h K dx_j
  for (jj=1; jj <= nde; jj++)
    *(a+jj)= K * *(q+jj);
}
