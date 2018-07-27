/*
  C code for the adiabatic approximation
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_integration.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define CHUNKSIZE 10
//Potentials
#include <galpy_potentials.h>
#include <integrateFullOrbit.h>
#include <actionAngle.h>
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
  Structure Declarations
*/
struct JRAdiabaticArg{
  double ER;
  double Lz22;
  int nargs;
  struct potentialArg * actionAngleArgs;
};
struct JzAdiabaticArg{
  double Ez;
  double R;
  int nargs;
  struct potentialArg * actionAngleArgs;
};
/*
  Function Declarations
*/
EXPORT void actionAngleAdiabatic_RperiRapZmax(int,double *,double *,double *,double *,
				       double *,int,int *,double *,double,
				       double *,double *,double *,int *);
EXPORT void actionAngleAdiabatic_actions(int,double *,double *,double *,double *,
				 double *,int,int *,double *,double,
				 double *,double *,int *);
void calcJRAdiabatic(int,double *,double *,double *,double *,double *,
		     int,struct potentialArg *,int);
void calcJzAdiabatic(int,double *,double *,double *,double *,int,
		     struct potentialArg *,int);
void calcRapRperi(int,double *,double *,double *,double *,double *,
		  int,struct potentialArg *);
void calcZmax(int,double *,double *,double *,double *,int,
	      struct potentialArg *);
double JRAdiabaticIntegrandSquared(double,void *);
double JRAdiabaticIntegrand(double,void *);
double JzAdiabaticIntegrandSquared(double,void *);
double JzAdiabaticIntegrand(double,void *);
double evaluateVerticalPotentials(double, double,int, struct potentialArg *);
/*
  Actual functions, inlines first
*/
static inline void calcEREzL(int ndata,
			     double *R,
			     double *vR,
			     double *vT,
			     double *z,
			     double *vz,
			     double *ER,
			     double *Ez,
			     double *Lz,
			     int nargs,
			     struct potentialArg * actionAngleArgs){
  int ii;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii)
  for (ii=0; ii < ndata; ii++){
    *(ER+ii)= evaluatePotentials(*(R+ii),0.,
				 nargs,actionAngleArgs)
      + 0.5 * *(vR+ii) * *(vR+ii)
      + 0.5 * *(vT+ii) * *(vT+ii);
    *(Ez+ii)= evaluateVerticalPotentials(*(R+ii),*(z+ii),
					 nargs,actionAngleArgs)     
      + 0.5 * *(vz+ii) * *(vz+ii);
    *(Lz+ii)= *(R+ii) * *(vT+ii);
  }
}
/*
  MAIN FUNCTIONS
 */
void actionAngleAdiabatic_RperiRapZmax(int ndata,
				       double *R,
				       double *vR,
				       double *vT,
				       double *z,
				       double *vz,
				       int npot,
				       int * pot_type,
				       double * pot_args,
				       double gamma,
				       double *rperi,
				       double *rap,
				       double *zmax,
				       int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args);
  //ER, Ez, Lz
  double *ER= (double *) malloc ( ndata * sizeof(double) );
  double *Ez= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEREzL(ndata,R,vR,vT,z,vz,ER,Ez,Lz,npot,actionAngleArgs);
  //Calculate peri and apocenters
  double *jz= (double *) malloc ( ndata * sizeof(double) );
  calcZmax(ndata,zmax,z,R,Ez,npot,actionAngleArgs);
  calcJzAdiabatic(ndata,jz,zmax,R,Ez,npot,actionAngleArgs,10);
  //Adjust planar effective potential for gamma
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii)
  for (ii=0; ii < ndata; ii++){
    *(Lz+ii)= fabs( *(Lz+ii) ) + gamma * *(jz+ii);
    *(ER+ii)+= 0.5 * *(Lz+ii) * *(Lz+ii) / *(R+ii) / *(R+ii) 
      - 0.5 * *(vT+ii) * *(vT+ii);
  }
  calcRapRperi(ndata,rperi,rap,R,ER,Lz,npot,actionAngleArgs);
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(ER);
  free(Ez);
  free(Lz);
  free(jz);
}
void actionAngleAdiabatic_actions(int ndata,
				  double *R,
				  double *vR,
				  double *vT,
				  double *z,
				  double *vz,
				  int npot,
				  int * pot_type,
				  double * pot_args,
				  double gamma,
				  double *jr,
				  double *jz,
				  int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args);
  //ER, Ez, Lz
  double *ER= (double *) malloc ( ndata * sizeof(double) );
  double *Ez= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEREzL(ndata,R,vR,vT,z,vz,ER,Ez,Lz,npot,actionAngleArgs);
  //Calculate peri and apocenters
  double *rperi= (double *) malloc ( ndata * sizeof(double) );
  double *rap= (double *) malloc ( ndata * sizeof(double) );
  double *zmax= (double *) malloc ( ndata * sizeof(double) );
  calcZmax(ndata,zmax,z,R,Ez,npot,actionAngleArgs);
  calcJzAdiabatic(ndata,jz,zmax,R,Ez,npot,actionAngleArgs,10);
  //Adjust planar effective potential for gamma
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii)
  for (ii=0; ii < ndata; ii++){
    *(Lz+ii)= fabs( *(Lz+ii) ) + gamma * *(jz+ii);
    *(ER+ii)+= 0.5 * *(Lz+ii) * *(Lz+ii) / *(R+ii) / *(R+ii) 
      - 0.5 * *(vT+ii) * *(vT+ii);
  }
  calcRapRperi(ndata,rperi,rap,R,ER,Lz,npot,actionAngleArgs);
  calcJRAdiabatic(ndata,jr,rperi,rap,ER,Lz,npot,actionAngleArgs,10);
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(ER);
  free(Ez);
  free(Lz);
  free(rperi);
  free(rap);
  free(zmax);
}
void calcJRAdiabatic(int ndata,
		     double * jr,
		     double * rperi,
		     double * rap,
		     double * ER,
		     double * Lz,
		     int nargs,
		     struct potentialArg * actionAngleArgs,
		     int order){
  int ii, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * JRInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct JRAdiabaticArg * params= (struct JRAdiabaticArg *) malloc ( nthreads * sizeof (struct JRAdiabaticArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii)							\
  shared(jr,rperi,rap,JRInt,params,T,ER,Lz)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(rperi+ii) == -9999.99 || *(rap+ii) == -9999.99 ){
      *(jr+ii)= 9999.99;
      continue;
    }
    if ( (*(rap+ii) - *(rperi+ii)) / *(rap+ii) < 0.000001 ){//circular
      *(jr+ii) = 0.;
      continue;
    }
    //Setup function
    (params+tid)->ER= *(ER+ii);
    (params+tid)->Lz22= 0.5 * *(Lz+ii) * *(Lz+ii);
    (JRInt+tid)->function = &JRAdiabaticIntegrand;
    (JRInt+tid)->params = params+tid;
    //Integrate
    *(jr+ii)= gsl_integration_glfixed (JRInt+tid,*(rperi+ii),*(rap+ii),T)
      * sqrt(2.) / M_PI;
  }
  free(JRInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
void calcJzAdiabatic(int ndata,
		     double * jz,
		     double * zmax,
		     double * R,
		     double * Ez,
		     int nargs,
		     struct potentialArg * actionAngleArgs,
		     int order){
  int ii, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * JzInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct JzAdiabaticArg * params= (struct JzAdiabaticArg *) malloc ( nthreads * sizeof (struct JzAdiabaticArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii)							\
  shared(jz,zmax,JzInt,params,T,Ez,R)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(zmax+ii) == -9999.99 ){
      *(jz+ii)= 9999.99;
      continue;
    }
    if ( *(zmax+ii) < 0.000001 ){//circular
      *(jz+ii) = 0.;
      continue;
    }
    //Setup function
    (params+tid)->Ez= *(Ez+ii);
    (params+tid)->R= *(R+ii);
    (JzInt+tid)->function = &JzAdiabaticIntegrand;
    (JzInt+tid)->params = params+tid;
    //Integrate
    *(jz+ii)= gsl_integration_glfixed (JzInt+tid,0.,*(zmax+ii),T)
      * 2 * sqrt(2.) / M_PI;
  }
  free(JzInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
void calcRapRperi(int ndata,
		  double * rperi,
		  double * rap,
		  double * R,
		  double * ER,
		  double * Lz,
		  int nargs,
		  struct potentialArg * actionAngleArgs){
  int ii, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  double peps, meps;
  gsl_function * JRRoot= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct JRAdiabaticArg * params= (struct JRAdiabaticArg *) malloc ( nthreads * sizeof (struct JRAdiabaticArg) );
  //Setup solver
  int status;
  int iter, max_iter = 100;
  const gsl_root_fsolver_type *T;
  double R_lo, R_hi;
  struct pragmasolver *s= (struct pragmasolver *) malloc ( nthreads * sizeof (struct pragmasolver) );
  T = gsl_root_fsolver_brent;
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
    (s+tid)->s= gsl_root_fsolver_alloc (T);
  }
  UNUSED int chunk= CHUNKSIZE;
  gsl_set_error_handler_off();
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,iter,status,R_lo,R_hi,meps,peps)			\
  shared(rperi,rap,JRRoot,params,s,R,ER,Lz,max_iter)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    //Setup function
    (params+tid)->ER= *(ER+ii);
    (params+tid)->Lz22= 0.5 * *(Lz+ii) * *(Lz+ii);
    (JRRoot+tid)->params = params+tid;
    (JRRoot+tid)->function = &JRAdiabaticIntegrandSquared;
    //Find starting points for minimum
    if ( fabs(GSL_FN_EVAL(JRRoot+tid,*(R+ii))) < 0.0000001){ //we are at rap or rperi
      peps= GSL_FN_EVAL(JRRoot+tid,*(R+ii)+0.0000001);
      meps= GSL_FN_EVAL(JRRoot+tid,*(R+ii)-0.0000001);
      if ( fabs(peps) < 0.00000001 && fabs(meps) < 0.00000001 && peps*meps >= 0.) {//circular
	*(rperi+ii) = *(R+ii);
	*(rap+ii) = *(R+ii);
      }
      else if ( peps < 0. && meps > 0. ) {//umax
	*(rap+ii)= *(R+ii);
	R_lo= 0.9 * (*(R+ii) - 0.0000001);
	R_hi= *(R+ii) - 0.00000001;
	while ( GSL_FN_EVAL(JRRoot+tid,R_lo) >= 0. && R_lo > 0.000000001){
	  R_hi= R_lo; //this makes sure that brent evaluates using previous
	  R_lo*= 0.9;
	}
	//Find root
	status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, R_lo, R_hi);
	if (status == GSL_EINVAL) {
	  *(rperi+ii) = 0.;//Assume zero if below 0.000000001
	  continue;
	}
	iter= 0;
	do
	  {
	    iter++;
	    status = gsl_root_fsolver_iterate ((s+tid)->s);
	    R_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	    R_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	    status = gsl_root_test_interval (R_lo, R_hi,
					     9.9999999999999998e-13,
					     4.4408920985006262e-16);
	  }
	while (status == GSL_CONTINUE && iter < max_iter);
	// LCOV_EXCL_START
	if (status == GSL_EINVAL) {//Shouldn't ever get here
	  *(rperi+ii) = -9999.99;
	  *(rap+ii) = -9999.99;
	  continue;
	}
	// LCOV_EXCL_STOP
	*(rperi+ii) = gsl_root_fsolver_root ((s+tid)->s);
      }
      else if ( peps > 0. && meps < 0. ){//umin
	*(rperi+ii)= *(R+ii);
	R_lo= *(R+ii) + 0.0000001;
	R_hi= 1.1 * (*(R+ii) + 0.0000001);
	while ( GSL_FN_EVAL(JRRoot+tid,R_hi) >= 0. && R_hi < 37.5) {
	  R_lo= R_hi; //this makes sure that brent evaluates using previous
	  R_hi*= 1.1;
	}
	//Find root
	status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, R_lo, R_hi);
	if (status == GSL_EINVAL) {
	  *(rperi+ii) = -9999.99;
	  *(rap+ii) = -9999.99;
	  continue;
	}
	iter= 0;
	do
	  {
	    iter++;
	    status = gsl_root_fsolver_iterate ((s+tid)->s);
	    R_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	    R_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	    status = gsl_root_test_interval (R_lo, R_hi,
					     9.9999999999999998e-13,
					     4.4408920985006262e-16);
	  }
	while (status == GSL_CONTINUE && iter < max_iter);
	// LCOV_EXCL_START
	if (status == GSL_EINVAL) {//Shouldn't ever get here 
	  *(rperi+ii) = -9999.99;
	  *(rap+ii) = -9999.99;
	  continue;
	}
	// LCOV_EXCL_STOP
	*(rap+ii) = gsl_root_fsolver_root ((s+tid)->s);
      }
    }
    else {
      R_lo= 0.9 * *(R+ii);
      R_hi= *(R+ii);
      while ( GSL_FN_EVAL(JRRoot+tid,R_lo) >= 0. && R_lo > 0.000000001){
	R_hi= R_lo; //this makes sure that brent evaluates using previous
	R_lo*= 0.9;
      }
      R_hi= (R_lo < 0.9 * *(R+ii)) ? R_lo / 0.9 / 0.9: *(R+ii);
      //Find root
      status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, R_lo, R_hi);
      if (status == GSL_EINVAL) {
	*(rperi+ii) = 0.;//Assume zero if below 0.000000001
      } else {
	iter= 0;
	do
	  {
	    iter++;
	    status = gsl_root_fsolver_iterate ((s+tid)->s);
	    R_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	    R_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	    status = gsl_root_test_interval (R_lo, R_hi,
					     9.9999999999999998e-13,
					     4.4408920985006262e-16);
	  }
	while (status == GSL_CONTINUE && iter < max_iter);
	// LCOV_EXCL_START
	if (status == GSL_EINVAL) {//Shouldn't ever get here
	  *(rperi+ii) = -9999.99;
	  *(rap+ii) = -9999.99;
	  continue;
	}
	// LCOV_EXCL_STOP
	*(rperi+ii) = gsl_root_fsolver_root ((s+tid)->s);
      }
      //Find starting points for maximum
      R_lo= *(R+ii);
      R_hi= 1.1 * *(R+ii);
      while ( GSL_FN_EVAL(JRRoot+tid,R_hi) > 0. && R_hi < 37.5) {
	R_lo= R_hi; //this makes sure that brent evaluates using previous
	R_hi*= 1.1;
      }
      R_lo= (R_hi > 1.1 * *(R+ii)) ? R_hi / 1.1 / 1.1: *(R+ii);
      //Find root
      status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, R_lo, R_hi);
      if (status == GSL_EINVAL) {
	*(rperi+ii) = -9999.99;
	*(rap+ii) = -9999.99;
	continue;
      }
      iter= 0;
      do
	{
	  iter++;
	  status = gsl_root_fsolver_iterate ((s+tid)->s);
	  R_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	  R_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	  status = gsl_root_test_interval (R_lo, R_hi,
					   9.9999999999999998e-13,
					   4.4408920985006262e-16);
	}
      while (status == GSL_CONTINUE && iter < max_iter);
      // LCOV_EXCL_START
      if (status == GSL_EINVAL) {//Shouldn't ever get here 
	*(rperi+ii) = -9999.99;
	*(rap+ii) = -9999.99;
	continue;
      }
      // LCOV_EXCL_STOP
      *(rap+ii) = gsl_root_fsolver_root ((s+tid)->s);
    }
  }
  gsl_set_error_handler (NULL);
  for (tid=0; tid < nthreads; tid++)
    gsl_root_fsolver_free( (s+tid)->s);
  free(s);
  free(JRRoot);
  free(params);
}
void calcZmax(int ndata,
	      double * zmax,
	      double * z,
	      double * R,
	      double * Ez,
	      int nargs,
	      struct potentialArg * actionAngleArgs){
  int ii, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * JzRoot= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct JzAdiabaticArg * params= (struct JzAdiabaticArg *) malloc ( nthreads * sizeof (struct JzAdiabaticArg) );
  //Setup solver
  int status;
  int iter, max_iter = 100;
  const gsl_root_fsolver_type *T;
  double z_lo, z_hi;
  struct pragmasolver *s= (struct pragmasolver *) malloc ( nthreads * sizeof (struct pragmasolver) );
  T = gsl_root_fsolver_brent;
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
    (s+tid)->s= gsl_root_fsolver_alloc (T);
  }
  UNUSED int chunk= CHUNKSIZE;
  gsl_set_error_handler_off();
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,iter,status,z_lo,z_hi)				\
  shared(zmax,JzRoot,params,s,z,Ez,R,max_iter)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    //Setup function
    (params+tid)->Ez= *(Ez+ii);
    (params+tid)->R= *(R+ii);
    (JzRoot+tid)->function = &JzAdiabaticIntegrandSquared;
    (JzRoot+tid)->params = params+tid;
    //Find starting points for minimum
    if ( fabs(GSL_FN_EVAL(JzRoot+tid,*(z+ii))) < 0.0000001){ //we are at zmax
      *(zmax+ii)= fabs( *(z+ii) );
    }
    else {
      z_lo= fabs ( *(z+ii) );
      z_hi= ( *(z+ii) == 0. ) ? 0.1: 1.1 * fabs( *(z+ii) );
      while ( GSL_FN_EVAL(JzRoot+tid,z_hi) >= 0. && z_hi < 37.5) {
	z_lo= z_hi; //this makes sure that brent evaluates using previous
	z_hi*= 1.1;
      }
      //Find root
      status = gsl_root_fsolver_set ((s+tid)->s, JzRoot+tid, z_lo, z_hi);
      if (status == GSL_EINVAL) {
	*(zmax+ii) = -9999.99;
	continue;
      }
      iter= 0;
      do
	{
	  iter++;
	  status = gsl_root_fsolver_iterate ((s+tid)->s);
	  z_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	  z_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	  status = gsl_root_test_interval (z_lo, z_hi,
					   9.9999999999999998e-13,
					   4.4408920985006262e-16);
	}
      while (status == GSL_CONTINUE && iter < max_iter);
      // LCOV_EXCL_START
      if (status == GSL_EINVAL) {//Shouldn't ever get here
	*(zmax+ii) = -9999.99;
	continue;
      }
      // LCOV_EXCL_STOP
      *(zmax+ii) = gsl_root_fsolver_root ((s+tid)->s);
    }
  }
  gsl_set_error_handler (NULL);
  for (tid=0; tid < nthreads; tid++)
    gsl_root_fsolver_free( (s+tid)->s);
  free(s);
  free(JzRoot);
  free(params);
}
double JRAdiabaticIntegrand(double R,
			   void * p){
  return sqrt(JRAdiabaticIntegrandSquared(R,p));
}
double JRAdiabaticIntegrandSquared(double R,
				  void * p){
  struct JRAdiabaticArg * params= (struct JRAdiabaticArg *) p;
  return params->ER - evaluatePotentials(R,0.,params->nargs,params->actionAngleArgs) - params->Lz22 / R / R;
}
double JzAdiabaticIntegrand(double z,
			    void * p){
  return sqrt(JzAdiabaticIntegrandSquared(z,p));
}
double JzAdiabaticIntegrandSquared(double z,
				   void * p){
  struct JzAdiabaticArg * params= (struct JzAdiabaticArg *) p;
  return params->Ez - evaluateVerticalPotentials(params->R,z,
						 params->nargs,
						 params->actionAngleArgs);
}
double evaluateVerticalPotentials(double R, double z,
				  int nargs, 
				  struct potentialArg * actionAngleArgs){
  return evaluatePotentials(R,z,nargs,actionAngleArgs)
    -evaluatePotentials(R,0.,nargs,actionAngleArgs);
}
