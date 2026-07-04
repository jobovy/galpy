/*
  C code for Binney (2012)'s Staeckel approximation code
*/
#ifdef _WIN32
#include <Python.h>
#endif
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
struct JRStaeckelArg{
  double E;
  double Lz22delta;
  double I3U;
  double delta;
  double u0;
  double sinh2u0;
  double v0;
  double sin2v0;
  double potu0v0;
  int nargs;
  struct potentialArg * actionAngleArgs;
};
struct JzStaeckelArg{
  double E;
  double Lz22delta;
  double I3V;
  double delta;
  double u0;
  double cosh2u0;
  double sinh2u0;
  double potupi2;
  int nargs;
  struct potentialArg * actionAngleArgs;
};
struct dJRStaeckelArg{
  double E;
  double Lz22delta;
  double I3U;
  double delta;
  double u0;
  double sinh2u0;
  double v0;
  double sin2v0;
  double potu0v0;
  double umin;
  double umax;
  int nargs;
  struct potentialArg * actionAngleArgs;
};
struct dJzStaeckelArg{
  double E;
  double Lz22delta;
  double I3V;
  double delta;
  double u0;
  double cosh2u0;
  double sinh2u0;
  double potupi2;
  double vmin;
  int nargs;
  struct potentialArg * actionAngleArgs;
};
struct u0EqArg{
  double E;
  double Lz22delta;
  double delta;
  int nargs;
  struct potentialArg * actionAngleArgs;
};
struct dJzdU0StaeckelArg{
  double E;
  double Lz22delta;
  double I3V;
  double delta;
  double u0;
  double cosh2u0;
  double sinh2u0;
  double potupi2;
  double vmin;
  double dpotupi2du0; // = -calcRforce(delta*sinh(u0),0)*delta*cosh(u0)
  int nargs;
  struct potentialArg * actionAngleArgs;
};
/*
  Function Declarations
*/
EXPORT void calcu0(int,double *,double *,int,int *,double *,tfuncs_type_arr,
       int,double*,double *,int *);
EXPORT void actionAngleStaeckel_uminUmaxVmin(int,double *,double *,double *,double *,
				      double *,double *,int,int *,double *,tfuncs_type_arr,
				      int,double *,double *,
				      double *,double *,int *);
EXPORT void actionAngleStaeckel_actions(int,double *,double *,double *,double *,
				 double *,double *,int,int *,double *,tfuncs_type_arr,int,
				 double *,int,double *,double *,int *);
EXPORT void actionAngleStaeckel_actionsFreqsAngles(int,double *,double *,double *,
					    double *,double *,double *,
					    int,int *,double *,tfuncs_type_arr,
					    int,double *,int,double *,double *,
					    double *,double *,double *,
					    double *,double *,double *,int *);
EXPORT void actionAngleStaeckel_actionsFreqs(int,double *,double *,double *,double *,
				      double *,double *,int,int *,double *,tfuncs_type_arr,
				      int,double *,int,double *,double *,
				      double *,double *,double *,int *);
EXPORT void actionAngleStaeckel_actionsJac(int,double *,double *,double *,double *,
				    double *,double *,int,int *,double *,tfuncs_type_arr,
				    int,double *,int,int,double *,double *,double *,int *);
void calcAnglesStaeckel(int,double *,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			double *,int,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			int,struct potentialArg *,int);
void calcFreqsFromDerivsStaeckel(int,double *,double *,double *,
				 double *,double *,double *,
				 double *,double *,double *,double *);
void calcdI3dJFromDerivsStaeckel(int,double *,double *,double *,double *,
				 double *,double *,double *,double *);
void calcJRStaeckel(int,double *,double *,double *,double *,double *,double *,
		    int,double *,double *,double *,double *,double *,double *,
		    int,struct potentialArg *,int);
void calcJzStaeckel(int,double *,double *,double *,double *,double *,int,
		    double *,double *,double *,double *,double *,int,
		    struct potentialArg *,int);
void calcdJRStaeckel(int,double *,double *,double *,double *,double *,
		     double *,double *,double *,int,
		     double *,double *,double *,double *,double *,double *,int,
		    struct potentialArg *,int);
void calcdJzStaeckel(int,double *,double *,double *,double *,double *,
		     double *,double *,int,double *,double *,double *,double *,
		     double *,int,
		     struct potentialArg *,int);
void calcdJzdU0Staeckel(int,double *,double *,double *,double *,double *,int,
			double *,double *,double *,double *,double *,int,
			struct potentialArg *,int);
void calcUminUmax(int,double *,double *,double *,double *,double *,double *,
		  double *,int,double *,double *,double *,double *,double *,
		  double *,int,struct potentialArg *);
void calcVmin(int,double *,double *,double *,double *,double *,double *,int,
	      double *,double *,double *,double *,double *,int,
	      struct potentialArg *);
double JRStaeckelIntegrandSquared4dJR(double,void *);
double JzStaeckelIntegrandSquared4dJz(double,void *);
double JRStaeckel_dSdu(double,struct dJRStaeckelArg *);
double JzStaeckel_dSdv(double,struct dJzStaeckelArg *);
double JzStaeckel_dSdu0(double,struct dJzStaeckelArg *);
void calcd2JRStaeckel(int,double *,double *,double *,double *,double *,double *,
		      int,double *,double *,double *,double *,double *,double *,
		      int,struct potentialArg *,int);
void calcd2JzStaeckel(int,double *,double *,double *,double *,double *,int,
		      double *,double *,double *,double *,double *,int,
		      struct potentialArg *,int);
EXPORT void actionAngleStaeckel_actionsFreqsAnglesJac(int,double *,double *,
	double *,double *,double *,double *,int,int *,double *,tfuncs_type_arr,
	int,double *,int,int,double *,double *,double *,double *,double *,
	double *,double *,double *,double *,double *,int *);
void calcAnglePartialDerivU(struct dJRStaeckelArg *,gsl_integration_glfixed_table *,
	int,double,double,double,double,double *,double *,double *,double *,
	double *,double *);
void calcAnglePartialDerivV(struct dJzStaeckelArg *,gsl_integration_glfixed_table *,
	int,double,double,double,double,int,double *,double *,double *,double *,
	double *,double *,double *);
double JRStaeckelIntegrandSquared(double,void *);
double JRStaeckelIntegrand(double,void *);
double JzStaeckelIntegrandSquared(double,void *);
double JzStaeckelIntegrand(double,void *);
double dJRdEStaeckelIntegrand(double,void *);
double dJRdELowStaeckelIntegrand(double,void *);
double dJRdEHighStaeckelIntegrand(double,void *);
double dJRdLzStaeckelIntegrand(double,void *);
double dJRdLzLowStaeckelIntegrand(double,void *);
double dJRdLzHighStaeckelIntegrand(double,void *);
double dJRdI3StaeckelIntegrand(double,void *);
double dJRdI3LowStaeckelIntegrand(double,void *);
double dJRdI3HighStaeckelIntegrand(double,void *);
double dJzdEStaeckelIntegrand(double,void *);
double dJzdELowStaeckelIntegrand(double,void *);
double dJzdEHighStaeckelIntegrand(double,void *);
double dJzdLzStaeckelIntegrand(double,void *);
double dJzdLzLowStaeckelIntegrand(double,void *);
double dJzdLzHighStaeckelIntegrand(double,void *);
double dJzdI3StaeckelIntegrand(double,void *);
double dJzdI3LowStaeckelIntegrand(double,void *);
double dJzdI3HighStaeckelIntegrand(double,void *);
double dJzdU0StaeckelIntegrand(double,void *);
double dJzdU0LowStaeckelIntegrand(double,void *);
double dJzdU0HighStaeckelIntegrand(double,void *);
double u0Equation(double,void *);
double evaluatePotentials(double,double,int, struct potentialArg *);
double evaluatePotentialsUV(double,double,double,int,struct potentialArg *);
/*
  Actual functions, inlines first
*/
static inline void uv_to_Rz(double u, double v, double * R, double *z,
			    double delta){
  *R= delta * sinh(u) * sin(v);
  *z= delta * cosh(u) * cos(v);
}
static inline void Rz_to_uv_vec(int ndata,
				double *R,
				double *z,
				double *u,
				double *v,
				int ndelta,
				double * delta){
  int ii;
  double d12, d22, coshu, cosv,tdelta;
  int delta_stride= ndelta == 1 ? 0 : 1;
  for (ii=0; ii < ndata; ii++) {
    tdelta= *(delta+ii*delta_stride);
    d12= (*(z+ii)+tdelta)*(*(z+ii)+tdelta)+(*(R+ii))*(*(R+ii));
    d22= (*(z+ii)-tdelta)*(*(z+ii)-tdelta)+(*(R+ii))*(*(R+ii));
    coshu= 0.5/tdelta*(sqrt(d12)+sqrt(d22));
    cosv=  0.5/tdelta*(sqrt(d12)-sqrt(d22));
    *u++= acosh(coshu);
    *v++= acos(cosv);
  }
  u-= ndata;
  v-= ndata;
}
static inline void calcEL(int ndata,
			  double *R,
			  double *vR,
			  double *vT,
			  double *z,
			  double *vz,
			  double *E,
			  double *Lz,
			  int nargs,
			  struct potentialArg * actionAngleArgs){
  int ii;
  for (ii=0; ii < ndata; ii++){
    *(E+ii)= evaluatePotentials(*(R+ii),*(z+ii),
				nargs,actionAngleArgs)
      + 0.5 * *(vR+ii) * *(vR+ii)
      + 0.5 * *(vT+ii) * *(vT+ii)
      + 0.5 * *(vz+ii) * *(vz+ii);
    *(Lz+ii)= *(R+ii) * *(vT+ii);
  }
}
/*
  MAIN FUNCTIONS
 */
void calcu0(int ndata,
	    double *E,
	    double *Lz,
	    int npot,
	    int * pot_type,
	    double * pot_args,
      tfuncs_type_arr pot_tfuncs,
	    int ndelta,
	    double * delta,
	    double *u0,
	    int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
  //setup the function to be minimized
  gsl_function u0Eq;
  struct u0EqArg * params= (struct u0EqArg *) malloc ( sizeof (struct u0EqArg) );
  params->nargs= npot;
  params->actionAngleArgs= actionAngleArgs;
  //Setup solver
  int status;
  int iter, max_iter = 100;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  double u_guess, u_lo, u_hi;
  T = gsl_min_fminimizer_brent;
  s = gsl_min_fminimizer_alloc (T);
  u0Eq.function = &u0Equation;
  int delta_stride= ndelta == 1 ? 0 : 1;
  for (ii=0; ii < ndata; ii++){
    //Setup function
    params->delta= *(delta+ii*delta_stride);
    params->E= *(E+ii);
    params->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    u0Eq.params = params;
    //Find starting points for minimum
    u_guess= 1.;
    u_lo= 0.001;
    u_hi= 100.;
    gsl_set_error_handler_off();
    status = gsl_min_fminimizer_set (s, &u0Eq, u_guess, u_lo, u_hi);
    if (status == GSL_EINVAL) {
      *(u0+ii)= u_hi;
      gsl_set_error_handler (NULL);
      continue;
    }
    gsl_set_error_handler (NULL);
    iter= 0;
    do
      {
	iter++;
	status = gsl_min_fminimizer_iterate (s);
	u_guess = gsl_min_fminimizer_x_minimum (s);
	u_lo = gsl_min_fminimizer_x_lower (s);
	u_hi = gsl_min_fminimizer_x_upper (s);
	status = gsl_min_test_interval (u_lo, u_hi,
					 9.9999999999999998e-13,
					 4.4408920985006262e-16);
      }
    while (status == GSL_CONTINUE && iter < max_iter);
    *(u0+ii)= gsl_min_fminimizer_x_minimum (s);
  }
  gsl_min_fminimizer_free (s);
  free(params);
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  *err= status;
}
void actionAngleStaeckel_uminUmaxVmin(int ndata,
				      double *R,
				      double *vR,
				      double *vT,
				      double *z,
				      double *vz,
				      double *u0,
				      int npot,
				      int * pot_type,
				      double * pot_args,
              tfuncs_type_arr pot_tfuncs,
				      int ndelta,
				      double * delta,
				      double *umin,
				      double *umax,
				      double *vmin,
				      int * err){
  // Just copied this over from actionAngleStaeckel_actions below, not elegant
  // but does the job...
  int ii;
  double tdelta;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  double *coshux= (double *) malloc ( ndata * sizeof(double) );
  double *sinhux= (double *) malloc ( ndata * sizeof(double) );
  double *sinvx= (double *) malloc ( ndata * sizeof(double) );
  double *cosvx= (double *) malloc ( ndata * sizeof(double) );
  double *pux= (double *) malloc ( ndata * sizeof(double) );
  double *pvx= (double *) malloc ( ndata * sizeof(double) );
  double *sinh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *cosh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *v0= (double *) malloc ( ndata * sizeof(double) );
  double *sin2v0= (double *) malloc ( ndata * sizeof(double) );
  double *potu0v0= (double *) malloc ( ndata * sizeof(double) );
  double *potupi2= (double *) malloc ( ndata * sizeof(double) );
  double *I3U= (double *) malloc ( ndata * sizeof(double) );
  double *I3V= (double *) malloc ( ndata * sizeof(double) );
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    tdelta= *(delta+ii*delta_stride);
    *(coshux+ii)= cosh(*(ux+ii));
    *(sinhux+ii)= sinh(*(ux+ii));
    *(cosvx+ii)= cos(*(vx+ii));
    *(sinvx+ii)= sin(*(vx+ii));
    *(pux+ii)= tdelta * (*(vR+ii) * *(coshux+ii) * *(sinvx+ii)
			+ *(vz+ii) * *(sinhux+ii) * *(cosvx+ii));
    *(pvx+ii)= tdelta * (*(vR+ii) * *(sinhux+ii) * *(cosvx+ii)
			- *(vz+ii) * *(coshux+ii) * *(sinvx+ii));
    *(sinh2u0+ii)= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(cosh2u0+ii)= cosh(*(u0+ii)) * cosh(*(u0+ii));
    *(v0+ii)= 0.5 * M_PI; //*(vx+ii);
    *(sin2v0+ii)= sin(*(v0+ii)) * sin(*(v0+ii));
    *(potu0v0+ii)= evaluatePotentialsUV(*(u0+ii),*(v0+ii),tdelta,
					npot,actionAngleArgs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,actionAngleArgs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,actionAngleArgs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,actionAngleArgs);
  }
  //Calculate 'peri' and 'apo'centers
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,
	       sin2v0,potu0v0,npot,actionAngleArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,
	   npot,actionAngleArgs);
  //Free
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(E);
  free(Lz);
  free(ux);
  free(vx);
  free(coshux);
  free(sinhux);
  free(sinvx);
  free(cosvx);
  free(pux);
  free(pvx);
  free(sinh2u0);
  free(cosh2u0);
  free(v0);
  free(sin2v0);
  free(potu0v0);
  free(potupi2);
  free(I3U);
  free(I3V);
}
void actionAngleStaeckel_actions(int ndata,
				 double *R,
				 double *vR,
				 double *vT,
				 double *z,
				 double *vz,
				 double *u0,
				 int npot,
				 int * pot_type,
				 double * pot_args,
         tfuncs_type_arr pot_tfuncs,
				 int ndelta,
				 double * delta,
				 int order,
				 double *jr,
				 double *jz,
				 int * err){
  int ii;
  double tdelta;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  double *coshux= (double *) malloc ( ndata * sizeof(double) );
  double *sinhux= (double *) malloc ( ndata * sizeof(double) );
  double *sinvx= (double *) malloc ( ndata * sizeof(double) );
  double *cosvx= (double *) malloc ( ndata * sizeof(double) );
  double *pux= (double *) malloc ( ndata * sizeof(double) );
  double *pvx= (double *) malloc ( ndata * sizeof(double) );
  double *sinh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *cosh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *v0= (double *) malloc ( ndata * sizeof(double) );
  double *sin2v0= (double *) malloc ( ndata * sizeof(double) );
  double *potu0v0= (double *) malloc ( ndata * sizeof(double) );
  double *potupi2= (double *) malloc ( ndata * sizeof(double) );
  double *I3U= (double *) malloc ( ndata * sizeof(double) );
  double *I3V= (double *) malloc ( ndata * sizeof(double) );
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    tdelta= *(delta+ii*delta_stride);
    *(coshux+ii)= cosh(*(ux+ii));
    *(sinhux+ii)= sinh(*(ux+ii));
    *(cosvx+ii)= cos(*(vx+ii));
    *(sinvx+ii)= sin(*(vx+ii));
    *(pux+ii)= tdelta * (*(vR+ii) * *(coshux+ii) * *(sinvx+ii)
			+ *(vz+ii) * *(sinhux+ii) * *(cosvx+ii));
    *(pvx+ii)= tdelta * (*(vR+ii) * *(sinhux+ii) * *(cosvx+ii)
			- *(vz+ii) * *(coshux+ii) * *(sinvx+ii));
    *(sinh2u0+ii)= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(cosh2u0+ii)= cosh(*(u0+ii)) * cosh(*(u0+ii));
    *(v0+ii)= 0.5 * M_PI; //*(vx+ii);
    *(sin2v0+ii)= sin(*(v0+ii)) * sin(*(v0+ii));
    *(potu0v0+ii)= evaluatePotentialsUV(*(u0+ii),*(v0+ii),tdelta,
					npot,actionAngleArgs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,actionAngleArgs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,actionAngleArgs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,actionAngleArgs);
  }
  //Calculate 'peri' and 'apo'centers
  double *umin= (double *) malloc ( ndata * sizeof(double) );
  double *umax= (double *) malloc ( ndata * sizeof(double) );
  double *vmin= (double *) malloc ( ndata * sizeof(double) );
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,
	       sin2v0,potu0v0,npot,actionAngleArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,
	   npot,actionAngleArgs);
  //Calculate the actions
  calcJRStaeckel(ndata,jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,
		 potu0v0,npot,actionAngleArgs,order);
  calcJzStaeckel(ndata,jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,
		 potupi2,npot,actionAngleArgs,order);
  //Free
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(E);
  free(Lz);
  free(ux);
  free(vx);
  free(coshux);
  free(sinhux);
  free(sinvx);
  free(cosvx);
  free(pux);
  free(pvx);
  free(sinh2u0);
  free(cosh2u0);
  free(v0);
  free(sin2v0);
  free(potu0v0);
  free(potupi2);
  free(I3U);
  free(I3V);
  free(umin);
  free(umax);
  free(vmin);
}
// Stationarity residual g(u)= df/du of f(u)= E sinh^2 u - cosh^2 u Phi(u,pi/2)
// - Lz22delta/sinh^2 u; calcu0's reference u0 solves g=0. Used to get
// du0/d(E,Lz) by implicit differentiation on the useu0==2 path (forces only,
// so no R2deriv-in-C requirement; f'' is a well-conditioned central FD of g).
static inline double staeckelU0Stationarity(double u,double E,double Lz22delta,
					    double delta,int npot,
					    struct potentialArg * aaArgs){
  double s= sinh(u), c= cosh(u);
  double P= evaluatePotentialsUV(u,0.5*M_PI,delta,npot,aaArgs);
  double Pp= -calcRforce(delta*s,0.,0.,0.,npot,aaArgs)*delta*c;
  return 2.*E*s*c - 2.*s*c*P - c*c*Pp + 2.*Lz22delta*c/(s*s*s);
}
// actions jr, jz AND the full 2x5 Jacobian d(jr,jz)/d(R,vR,vT,z,vz) per object,
// assembled IN C: the six t^2-substituted action-derivative integrals
// dJ/d(E,Lz,I3) + a new dJz/du0 integral, chained through the analytic
// elementary derivatives d(E,Lz,I3Utilde,I3V,u0)/d(coords). I3Utilde =
// I3U-(sinh2u0+sin2v0)*potu0v0 makes J_R's u0/potu0v0 gauge terms drop out
// (dJr/dx = djrdE dE/dx + djrdLz dLz/dx + djrdI3 dI3Utilde/dx); J_z carries the
// extra dJz/du0*du0/dx because u0 enters its integrand via potentialStaeckel(u0,v).
// jac layout: [ii*10 + {0..4}]=dJr/d(R,vR,vT,z,vz), [ii*10 + {5..9}]=dJz/d(...).
// useu0 selects the reference-u0 mode: 0 -> u0 tracks ux (du0/dx=dux/dx);
// 1 -> user-fixed u0 kwarg (du0/dx=0); 2 -> u0=calcu0(E,Lz) reference, so
// du0/dx = du0/dE dE/dx + du0/dLz dLz/dx (exact useu0=True gradient).
void actionAngleStaeckel_actionsJac(int ndata,
				    double *R,
				    double *vR,
				    double *vT,
				    double *z,
				    double *vz,
				    double *u0,
				    int npot,
				    int * pot_type,
				    double * pot_args,
				    tfuncs_type_arr pot_tfuncs,
				    int ndelta,
				    double * delta,
				    int order,
				    int useu0,
				    double *jr,
				    double *jz,
				    double *jac,
				    int * err){
  int ii;
  double tdelta;
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  double *coshux= (double *) malloc ( ndata * sizeof(double) );
  double *sinhux= (double *) malloc ( ndata * sizeof(double) );
  double *sinvx= (double *) malloc ( ndata * sizeof(double) );
  double *cosvx= (double *) malloc ( ndata * sizeof(double) );
  double *pux= (double *) malloc ( ndata * sizeof(double) );
  double *pvx= (double *) malloc ( ndata * sizeof(double) );
  double *sinh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *cosh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *v0= (double *) malloc ( ndata * sizeof(double) );
  double *sin2v0= (double *) malloc ( ndata * sizeof(double) );
  double *potu0v0= (double *) malloc ( ndata * sizeof(double) );
  double *potupi2= (double *) malloc ( ndata * sizeof(double) );
  double *I3U= (double *) malloc ( ndata * sizeof(double) );
  double *I3V= (double *) malloc ( ndata * sizeof(double) );
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    tdelta= *(delta+ii*delta_stride);
    *(coshux+ii)= cosh(*(ux+ii));
    *(sinhux+ii)= sinh(*(ux+ii));
    *(cosvx+ii)= cos(*(vx+ii));
    *(sinvx+ii)= sin(*(vx+ii));
    *(pux+ii)= tdelta * (*(vR+ii) * *(coshux+ii) * *(sinvx+ii)
			+ *(vz+ii) * *(sinhux+ii) * *(cosvx+ii));
    *(pvx+ii)= tdelta * (*(vR+ii) * *(sinhux+ii) * *(cosvx+ii)
			- *(vz+ii) * *(coshux+ii) * *(sinvx+ii));
    *(sinh2u0+ii)= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(cosh2u0+ii)= cosh(*(u0+ii)) * cosh(*(u0+ii));
    *(v0+ii)= 0.5 * M_PI;
    *(sin2v0+ii)= sin(*(v0+ii)) * sin(*(v0+ii));
    *(potu0v0+ii)= evaluatePotentialsUV(*(u0+ii),*(v0+ii),tdelta,
					npot,actionAngleArgs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,actionAngleArgs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,actionAngleArgs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,actionAngleArgs);
  }
  double *umin= (double *) malloc ( ndata * sizeof(double) );
  double *umax= (double *) malloc ( ndata * sizeof(double) );
  double *vmin= (double *) malloc ( ndata * sizeof(double) );
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,
	       sin2v0,potu0v0,npot,actionAngleArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,
	   npot,actionAngleArgs);
  calcJRStaeckel(ndata,jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,
		 potu0v0,npot,actionAngleArgs,order);
  calcJzStaeckel(ndata,jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,
		 potupi2,npot,actionAngleArgs,order);
  // Action-derivative integrals: dJ/d(E,Lz,I3) (six) + the new dJz/du0.
  double *djrdE= (double *) malloc ( ndata * sizeof(double) );
  double *djrdLz= (double *) malloc ( ndata * sizeof(double) );
  double *djrdI3= (double *) malloc ( ndata * sizeof(double) );
  double *djzdE= (double *) malloc ( ndata * sizeof(double) );
  double *djzdLz= (double *) malloc ( ndata * sizeof(double) );
  double *djzdI3= (double *) malloc ( ndata * sizeof(double) );
  double *djzdU0= (double *) malloc ( ndata * sizeof(double) );
  calcdJRStaeckel(ndata,djrdE,djrdLz,djrdI3,umin,umax,E,Lz,I3U,ndelta,delta,u0,
		  sinh2u0,v0,sin2v0,potu0v0,npot,actionAngleArgs,order);
  calcdJzStaeckel(ndata,djzdE,djzdLz,djzdI3,vmin,E,Lz,I3V,ndelta,delta,u0,
		  cosh2u0,sinh2u0,potupi2,npot,actionAngleArgs,order);
  calcdJzdU0Staeckel(ndata,djzdU0,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,
		     potupi2,npot,actionAngleArgs,order);
  // Assemble the 2x5 Jacobian: chain the action-derivatives through the
  // analytic elementary d(E,Lz,I3Utilde,I3V,u0)/d(R,vR,vT,z,vz).
#pragma omp parallel for schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    int kk;
    tdelta= *(delta+ii*delta_stride);
    double shx= *(sinhux+ii), chx= *(coshux+ii);
    double svx= *(sinvx+ii), cvx= *(cosvx+ii);
    double tu0= *(u0+ii), sh0= sinh(tu0), ch0= cosh(tu0);
    double tE= *(E+ii), tLz= *(Lz+ii), tpux= *(pux+ii), tpvx= *(pvx+ii);
    double tvR= *(vR+ii), tvT= *(vT+ii), tvz= *(vz+ii);
    double D= shx*shx + svx*svx;
    // coordinate-transform derivatives (R = d*sinh(u)*sin(v), z = d*cosh(u)*cos(v))
    double dux_dR= chx*svx/(tdelta*D), dux_dz= shx*cvx/(tdelta*D);
    double dvx_dR= shx*cvx/(tdelta*D), dvx_dz= -chx*svx/(tdelta*D);
    double dux[5]= {dux_dR,0.,0.,dux_dz,0.};
    double dvx[5]= {dvx_dR,0.,0.,dvx_dz,0.};
    double dE[5]= {-calcRforce(*(R+ii),*(z+ii),0.,0.,npot,actionAngleArgs),
		   tvR,tvT,
		   -calczforce(*(R+ii),*(z+ii),0.,0.,npot,actionAngleArgs),tvz};
    double dLz[5]= {tvT,0.,*(R+ii),0.,0.};
    // du0/dx per reference-u0 mode (see useu0 doc above). Mode 2 chains
    // du0/d(E,Lz) (implicit diff of the stationarity residual g=df/du=0,
    // f''=du0 denom via central FD of g) onto the elementary dE,dLz.
    double du0[5];
    double du0dE= 0., du0dLz= 0.;
    if ( useu0 == 2 ){
      double L2= 0.5*tLz*tLz/(tdelta*tdelta);
      double hh= 1.e-5;
      double fpp= ( staeckelU0Stationarity(tu0+hh,tE,L2,tdelta,npot,actionAngleArgs)
		    -staeckelU0Stationarity(tu0-hh,tE,L2,tdelta,npot,actionAngleArgs) )
	/ ( 2.*hh );
      du0dE= -2.*sh0*ch0/fpp;
      du0dLz= -2.*ch0*tLz/(tdelta*tdelta*sh0*sh0*sh0)/fpp;
    }
    for (kk=0;kk<5;kk++){
      if ( useu0 == 0 ) du0[kk]= dux[kk];
      else if ( useu0 == 1 ) du0[kk]= 0.;
      else du0[kk]= du0dE*dE[kk] + du0dLz*dLz[kk];
    }
    // momentum derivatives
    double dpux_dux= tdelta*(tvR*shx*svx + tvz*chx*cvx);
    double dpux_dvx= tdelta*(tvR*chx*cvx - tvz*shx*svx);
    double dpvx_dux= tdelta*(tvR*chx*cvx - tvz*shx*svx);
    double dpvx_dvx= tdelta*(-tvR*shx*svx - tvz*chx*cvx);
    double dpux[5], dpvx[5];
    for (kk=0;kk<5;kk++){
      dpux[kk]= dpux_dux*dux[kk] + dpux_dvx*dvx[kk];
      dpvx[kk]= dpvx_dux*dux[kk] + dpvx_dvx*dvx[kk];
    }
    dpux[1]+= tdelta*chx*svx; dpux[4]+= tdelta*shx*cvx;
    dpvx[1]+= tdelta*shx*cvx; dpvx[4]+= -tdelta*chx*svx;
    // I3Utilde = E*shx^2 - pux^2/(2 d^2) - Lz^2/(2 d^2 shx^2) - (shx^2+1)*Phi(ux,pi/2)
    double Pux= evaluatePotentialsUV(*(ux+ii),0.5*M_PI,tdelta,npot,actionAngleArgs);
    double FRux= calcRforce(tdelta*shx,0.,0.,0.,npot,actionAngleArgs);
    double dPux_dux= -FRux*tdelta*chx; // dPhi(ux,pi/2)/dux (z-line derivative is 0)
    double dI3Ut_dE= shx*shx;
    double dI3Ut_dLz= -tLz/(tdelta*tdelta*shx*shx);
    double dI3Ut_dpux= -tpux/(tdelta*tdelta);
    double dI3Ut_dux= 2.*shx*chx*tE + tLz*tLz*chx/(tdelta*tdelta*shx*shx*shx)
      - 2.*shx*chx*Pux - (shx*shx+1.)*dPux_dux;
    // I3V = -E*svx^2 + pvx^2/(2 d^2) + Lz^2/(2 d^2 svx^2)
    //       - cosh2u0*potupi2 + (sinh2u0+svx^2)*Phi(u0,vx)
    double P0v= evaluatePotentialsUV(tu0,*(vx+ii),tdelta,npot,actionAngleArgs);
    double FRu0= calcRforce(tdelta*sh0,0.,0.,0.,npot,actionAngleArgs);
    double Rp= tdelta*sh0*svx, zp= tdelta*ch0*cvx;
    double FRp= calcRforce(Rp,zp,0.,0.,npot,actionAngleArgs);
    double Fzp= calczforce(Rp,zp,0.,0.,npot,actionAngleArgs);
    double dPu0_du0= -FRu0*tdelta*ch0;               // dPhi(u0,pi/2)/du0
    double dP0v_dvx= -FRp*tdelta*sh0*cvx + Fzp*tdelta*ch0*svx; // dPhi(u0,vx)/dvx
    double dP0v_du0= -FRp*tdelta*ch0*svx - Fzp*tdelta*sh0*cvx; // dPhi(u0,vx)/du0
    double dI3V_dE= -svx*svx;
    double dI3V_dLz= tLz/(tdelta*tdelta*svx*svx);
    double dI3V_dpvx= tpvx/(tdelta*tdelta);
    double dI3V_dvx= -2.*tE*svx*cvx - tLz*tLz*cvx/(tdelta*tdelta*svx*svx*svx)
      + 2.*svx*cvx*P0v + (sh0*sh0+svx*svx)*dP0v_dvx;
    double dI3V_du0= -2.*ch0*sh0* *(potupi2+ii) - ch0*ch0*dPu0_du0
      + 2.*sh0*ch0*P0v + (sh0*sh0+svx*svx)*dP0v_du0;
    double tdjrdE= *(djrdE+ii), tdjrdLz= *(djrdLz+ii), tdjrdI3= *(djrdI3+ii);
    double tdjzdE= *(djzdE+ii), tdjzdLz= *(djzdLz+ii), tdjzdI3= *(djzdI3+ii);
    double tdjzdU0= *(djzdU0+ii);
    for (kk=0;kk<5;kk++){
      double dI3Ut= dI3Ut_dE*dE[kk] + dI3Ut_dLz*dLz[kk]
	+ dI3Ut_dpux*dpux[kk] + dI3Ut_dux*dux[kk];
      double dI3V= dI3V_dE*dE[kk] + dI3V_dLz*dLz[kk] + dI3V_dpvx*dpvx[kk]
	+ dI3V_dvx*dvx[kk] + dI3V_du0*du0[kk];
      *(jac+ii*10+kk)= tdjrdE*dE[kk] + tdjrdLz*dLz[kk] + tdjrdI3*dI3Ut;
      *(jac+ii*10+5+kk)= tdjzdE*dE[kk] + tdjzdLz*dLz[kk] + tdjzdI3*dI3V
	+ tdjzdU0*du0[kk];
    }
  }
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(E); free(Lz); free(ux); free(vx); free(coshux); free(sinhux);
  free(sinvx); free(cosvx); free(pux); free(pvx); free(sinh2u0); free(cosh2u0);
  free(v0); free(sin2v0); free(potu0v0); free(potupi2); free(I3U); free(I3V);
  free(umin); free(umax); free(vmin);
  free(djrdE); free(djrdLz); free(djrdI3); free(djzdE); free(djzdLz);
  free(djzdI3); free(djzdU0);
  *err= 0;
}
// c=True C-native frequency Jacobian (#131): jr,jz,Omega{r,phi,z} + the (3x5)
// d(Omega)/d(R,vR,vT,z,vz). Omega is a 1st derivative of J, so dOmega/dcoord =
// sum_P (dOmega/dP)*(dP/dcoord); dOmega/dP composes the action Hessians
// (calcd2JR/Jz) via the quotient-rule partials of calcFreqsFromDerivsStaeckel,
// then chains through the SAME dP/dcoord block as actionsJac.
EXPORT void actionAngleStaeckel_actionsFreqsJac(int ndata,
    double *R,double *vR,double *vT,double *z,double *vz,double *u0,
    int npot,int *pot_type,double *pot_args,tfuncs_type_arr pot_tfuncs,
    int ndelta,double *delta,int order,int useu0,
    double *jr,double *jz,double *Or,double *Op,double *Oz,
    double *ojac,int *err){
  int ii; double tdelta;
  struct potentialArg * aaArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,aaArgs,&pot_type,&pot_args,&pot_tfuncs);
  double *E=malloc(ndata*sizeof(double)),*Lz=malloc(ndata*sizeof(double));
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,aaArgs);
  double *ux=malloc(ndata*sizeof(double)),*vx=malloc(ndata*sizeof(double));
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  double *shx=malloc(ndata*sizeof(double)),*chx=malloc(ndata*sizeof(double));
  double *svx=malloc(ndata*sizeof(double)),*cvx=malloc(ndata*sizeof(double));
  double *pux=malloc(ndata*sizeof(double)),*pvx=malloc(ndata*sizeof(double));
  double *sinh2u0=malloc(ndata*sizeof(double)),*cosh2u0=malloc(ndata*sizeof(double));
  double *v0=malloc(ndata*sizeof(double)),*sin2v0=malloc(ndata*sizeof(double));
  double *potu0v0=malloc(ndata*sizeof(double)),*potupi2=malloc(ndata*sizeof(double));
  double *I3U=malloc(ndata*sizeof(double)),*I3V=malloc(ndata*sizeof(double));
  int ds= ndelta==1?0:1;
  for (ii=0;ii<ndata;ii++){
    tdelta= *(delta+ii*ds);
    chx[ii]=cosh(ux[ii]); shx[ii]=sinh(ux[ii]); cvx[ii]=cos(vx[ii]); svx[ii]=sin(vx[ii]);
    pux[ii]=tdelta*(vR[ii]*chx[ii]*svx[ii]+vz[ii]*shx[ii]*cvx[ii]);
    pvx[ii]=tdelta*(vR[ii]*shx[ii]*cvx[ii]-vz[ii]*chx[ii]*svx[ii]);
    sinh2u0[ii]=sinh(u0[ii])*sinh(u0[ii]); cosh2u0[ii]=cosh(u0[ii])*cosh(u0[ii]);
    v0[ii]=0.5*M_PI; sin2v0[ii]=sin(v0[ii])*sin(v0[ii]);
    potu0v0[ii]=evaluatePotentialsUV(u0[ii],v0[ii],tdelta,npot,aaArgs);
    I3U[ii]=E[ii]*shx[ii]*shx[ii]-0.5*pux[ii]*pux[ii]/tdelta/tdelta
      -0.5*Lz[ii]*Lz[ii]/tdelta/tdelta/shx[ii]/shx[ii]
      -(shx[ii]*shx[ii]+sin2v0[ii])*evaluatePotentialsUV(ux[ii],v0[ii],tdelta,npot,aaArgs)
      +(sinh2u0[ii]+sin2v0[ii])*potu0v0[ii];
    potupi2[ii]=evaluatePotentialsUV(u0[ii],0.5*M_PI,tdelta,npot,aaArgs);
    I3V[ii]=-E[ii]*svx[ii]*svx[ii]+0.5*pvx[ii]*pvx[ii]/tdelta/tdelta
      +0.5*Lz[ii]*Lz[ii]/tdelta/tdelta/svx[ii]/svx[ii]-cosh2u0[ii]*potupi2[ii]
      +(sinh2u0[ii]+svx[ii]*svx[ii])*evaluatePotentialsUV(u0[ii],vx[ii],tdelta,npot,aaArgs);
  }
  double *umin=malloc(ndata*sizeof(double)),*umax=malloc(ndata*sizeof(double)),*vmin=malloc(ndata*sizeof(double));
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs);
  calcJRStaeckel(ndata,jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs,order);
  calcJzStaeckel(ndata,jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  double *djrdE=malloc(ndata*sizeof(double)),*djrdLz=malloc(ndata*sizeof(double)),*djrdI3=malloc(ndata*sizeof(double));
  double *djzdE=malloc(ndata*sizeof(double)),*djzdLz=malloc(ndata*sizeof(double)),*djzdI3=malloc(ndata*sizeof(double));
  double *djzdU0=malloc(ndata*sizeof(double)),*detA=malloc(ndata*sizeof(double));
  calcdJRStaeckel(ndata,djrdE,djrdLz,djrdI3,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs,order);
  calcdJzStaeckel(ndata,djzdE,djzdLz,djzdI3,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  calcdJzdU0Staeckel(ndata,djzdU0,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  calcFreqsFromDerivsStaeckel(ndata,Or,Op,Oz,detA,djrdE,djrdLz,djrdI3,djzdE,djzdLz,djzdI3);
  double *d2jr=malloc(ndata*9*sizeof(double)),*d2jz=malloc(ndata*12*sizeof(double));
  calcd2JRStaeckel(ndata,d2jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs,order);
  calcd2JzStaeckel(ndata,d2jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  for (ii=0;ii<ndata;ii++){
    int kk;
    tdelta= *(delta+ii*ds);
    double sh=shx[ii],ch=chx[ii],sv=svx[ii],cv=cvx[ii];
    double tu0=u0[ii],sh0=sinh(tu0),ch0=cosh(tu0);
    double tE=E[ii],tLz=Lz[ii],tpux=pux[ii],tpvx=pvx[ii],tvR=vR[ii],tvT=vT[ii],tvz=vz[ii];
    double D=sh*sh+sv*sv;
    double dux_dR=ch*sv/(tdelta*D),dux_dz=sh*cv/(tdelta*D);
    double dvx_dR=sh*cv/(tdelta*D),dvx_dz=-ch*sv/(tdelta*D);
    double dux[5]={dux_dR,0.,0.,dux_dz,0.},dvx[5]={dvx_dR,0.,0.,dvx_dz,0.};
    double dE[5]={-calcRforce(R[ii],z[ii],0.,0.,npot,aaArgs),tvR,tvT,
                  -calczforce(R[ii],z[ii],0.,0.,npot,aaArgs),tvz};
    double dLz[5]={tvT,0.,R[ii],0.,0.};
    double du0[5],du0dE=0.,du0dLz=0.;
    if ( useu0==2 ){
      double L2=0.5*tLz*tLz/(tdelta*tdelta),hh=1.e-5;
      double fpp=( staeckelU0Stationarity(tu0+hh,tE,L2,tdelta,npot,aaArgs)
                  -staeckelU0Stationarity(tu0-hh,tE,L2,tdelta,npot,aaArgs) )/(2.*hh);
      du0dE=-2.*sh0*ch0/fpp; du0dLz=-2.*ch0*tLz/(tdelta*tdelta*sh0*sh0*sh0)/fpp;
    }
    for (kk=0;kk<5;kk++) du0[kk]= useu0==0?dux[kk]:(useu0==1?0.:du0dE*dE[kk]+du0dLz*dLz[kk]);
    double dpux_dux=tdelta*(tvR*sh*sv+tvz*ch*cv),dpux_dvx=tdelta*(tvR*ch*cv-tvz*sh*sv);
    double dpvx_dux=tdelta*(tvR*ch*cv-tvz*sh*sv),dpvx_dvx=tdelta*(-tvR*sh*sv-tvz*ch*cv);
    double dpux[5],dpvx[5];
    for (kk=0;kk<5;kk++){ dpux[kk]=dpux_dux*dux[kk]+dpux_dvx*dvx[kk]; dpvx[kk]=dpvx_dux*dux[kk]+dpvx_dvx*dvx[kk]; }
    dpux[1]+=tdelta*ch*sv; dpux[4]+=tdelta*sh*cv; dpvx[1]+=tdelta*sh*cv; dpvx[4]+=-tdelta*ch*sv;
    double Pux=evaluatePotentialsUV(ux[ii],0.5*M_PI,tdelta,npot,aaArgs);
    double FRux=calcRforce(tdelta*sh,0.,0.,0.,npot,aaArgs),dPux_dux=-FRux*tdelta*ch;
    double dI3Ut_dE=sh*sh,dI3Ut_dLz=-tLz/(tdelta*tdelta*sh*sh),dI3Ut_dpux=-tpux/(tdelta*tdelta);
    double dI3Ut_dux=2.*sh*ch*tE+tLz*tLz*ch/(tdelta*tdelta*sh*sh*sh)-2.*sh*ch*Pux-(sh*sh+1.)*dPux_dux;
    double P0v=evaluatePotentialsUV(tu0,vx[ii],tdelta,npot,aaArgs);
    double FRu0=calcRforce(tdelta*sh0,0.,0.,0.,npot,aaArgs);
    double Rp=tdelta*sh0*sv,zp=tdelta*ch0*cv;
    double FRp=calcRforce(Rp,zp,0.,0.,npot,aaArgs),Fzp=calczforce(Rp,zp,0.,0.,npot,aaArgs);
    double dPu0_du0=-FRu0*tdelta*ch0;
    double dP0v_dvx=-FRp*tdelta*sh0*cv+Fzp*tdelta*ch0*sv,dP0v_du0=-FRp*tdelta*ch0*sv-Fzp*tdelta*sh0*cv;
    double dI3V_dE=-sv*sv,dI3V_dLz=tLz/(tdelta*tdelta*sv*sv),dI3V_dpvx=tpvx/(tdelta*tdelta);
    double dI3V_dvx=-2.*tE*sv*cv-tLz*tLz*cv/(tdelta*tdelta*sv*sv*sv)+2.*sv*cv*P0v+(sh0*sh0+sv*sv)*dP0v_dvx;
    double dI3V_du0=-2.*ch0*sh0*potupi2[ii]-ch0*ch0*dPu0_du0+2.*sh0*ch0*P0v+(sh0*sh0+sv*sv)*dP0v_du0;
    double dI3Ut[5],dI3V[5];
    for (kk=0;kk<5;kk++){
      dI3Ut[kk]=dI3Ut_dE*dE[kk]+dI3Ut_dLz*dLz[kk]+dI3Ut_dpux*dpux[kk]+dI3Ut_dux*dux[kk];
      dI3V[kk]=dI3V_dE*dE[kk]+dI3V_dLz*dLz[kk]+dI3V_dpvx*dpvx[kk]+dI3V_dvx*dvx[kk]+dI3V_du0*du0[kk];
    }
    // fused (5x5) rows: 0=jr,1=jz (action Jacobian, always differentiable, as
    // #1051 actionsJac), 2=Or,3=Op,4=Oz (freq Jacobian via the action Hessians).
    for (kk=0;kk<5;kk++){
      ojac[ii*25+kk]= djrdE[ii]*dE[kk]+djrdLz[ii]*dLz[kk]+djrdI3[ii]*dI3Ut[kk];
      ojac[ii*25+5+kk]= djzdE[ii]*dE[kk]+djzdLz[ii]*dLz[kk]+djzdI3[ii]*dI3V[kk]+djzdU0[ii]*du0[kk];
    }
    double a=djrdE[ii],b=djrdLz[ii],cJR=djrdI3[ii],dJZ=djzdE[ii],eJZ=djzdLz[ii],fJZ=djzdI3[ii];
    if ( a==9999.99 || dJZ==9999.99 || detA[ii]==0. ){
      for (kk=0;kk<15;kk++) ojac[ii*25+10+kk]=0.; continue; // zero freq rows (2,3,4)
    }
    double id=1./detA[ii],id2=id*id,N=cJR*eJZ-fJZ*b;
    double gOr[6]={-fJZ*fJZ*id2,0.,fJZ*dJZ*id2,fJZ*cJR*id2,0.,id-fJZ*a*id2};
    double gOp[6]={-N*fJZ*id2,-fJZ*id,eJZ*id+N*dJZ*id2,N*cJR*id2,cJR*id,-b*id-N*a*id2};
    double gOz[6]={cJR*fJZ*id2,0.,-id-cJR*dJZ*id2,-cJR*cJR*id2,0.,cJR*a*id2};
    double *gg[3]={gOr,gOp,gOz};
    double *D2R=d2jr+ii*9,*D2Z=d2jz+ii*12;
    for (int oi=0;oi<3;oi++){
      double *g=gg[oi];
      double dOdE=g[0]*D2R[0]+g[1]*D2R[3]+g[2]*D2R[6]+g[3]*D2Z[0]+g[4]*D2Z[4]+g[5]*D2Z[8];
      double dOdLz=g[0]*D2R[1]+g[1]*D2R[4]+g[2]*D2R[7]+g[3]*D2Z[1]+g[4]*D2Z[5]+g[5]*D2Z[9];
      double dOdI3U=g[0]*D2R[2]+g[1]*D2R[5]+g[2]*D2R[8];
      double dOdI3V=g[3]*D2Z[2]+g[4]*D2Z[6]+g[5]*D2Z[10];
      double dOdu0=g[3]*D2Z[3]+g[4]*D2Z[7]+g[5]*D2Z[11];
      for (kk=0;kk<5;kk++)
        ojac[ii*25+(oi+2)*5+kk]=dOdE*dE[kk]+dOdLz*dLz[kk]+dOdI3U*dI3Ut[kk]+dOdI3V*dI3V[kk]+dOdu0*du0[kk];
    }
  }
  free_potentialArgs(npot,aaArgs); free(aaArgs);
  free(E);free(Lz);free(ux);free(vx);free(shx);free(chx);free(svx);free(cvx);
  free(pux);free(pvx);free(sinh2u0);free(cosh2u0);free(v0);free(sin2v0);
  free(potu0v0);free(potupi2);free(I3U);free(I3V);free(umin);free(umax);free(vmin);
  free(djrdE);free(djrdLz);free(djrdI3);free(djzdE);free(djzdLz);free(djzdI3);
  free(djzdU0);free(detA);free(d2jr);free(d2jz);
  *err=0;
}
// ---- Route-A partial-range angle-Jacobian helpers for c=True (#131 PR-B) ----
// d(P_k)/d(coord) for the three u-side angle factor families k={sinh^2u,1,1/sinh^2u}
// (PE,PI,PL), plus the partial-integral VALUES Pval[k]. Woven t^2 form (u=base+
// sign*(mid*s)^2, s in GL[0,1] order): boundary + S^{-3/2} range + turning-point
// cancellation, with the turning-pt base motion dbase[c]= sum_Pk A[Pk]*dparam[Pk],
// A[Pk]=-SP_base[Pk]/Su_base (implicit d(tp)/dc, mirrors calcd2JR). u_c=(1-s^2)*
// dbase+s^2*dux interpolates base & current-position motion. 3 params {E,Lz,I3Ut}
// (u0 folded into I3Utilde -> dS/du0=0 on the u-side).
void calcAnglePartialDerivU(struct dJRStaeckelArg * p,
			    gsl_integration_glfixed_table * T,int order,
			    double base,double sign,double mid,double Ld2,
			    double * dE,double * dLz,double * dI3Ut,double * dux,
			    double * Pval,double * dP){
  int gi,k,c;
  for (k=0;k<3;k++){ Pval[k]= 0.; for (c=0;c<5;c++) dP[k*5+c]= 0.; }
  double shb= sinh(base), sinh2b= shb*shb;
  double SPbase[3]= { sinh2b, -Ld2/sinh2b, -1. };
  double Su_base= JRStaeckel_dSdu(base,p);
  double dbase[5],dmid[5];
  for (c=0;c<5;c++)
    dbase[c]= (-SPbase[0]/Su_base)*dE[c]+(-SPbase[1]/Su_base)*dLz[c]
      +(-SPbase[2]/Su_base)*dI3Ut[c];
  for (c=0;c<5;c++) dmid[c]= sign*(dux[c]-dbase[c])/(2.*mid);
  double si,wi;
  for (gi=0;gi<order;gi++){
    gsl_integration_glfixed_point(0.,1.,gi,&si,&wi,T);
    double t= mid*si, u= base+sign*t*t;
    double S= JRStaeckelIntegrandSquared4dJR(u,p);
    if ( S <= 0. ) continue;
    double sq= sqrt(S), S15= S*sq;
    double Su= JRStaeckel_dSdu(u,p);
    double shu= sinh(u), chu= cosh(u), sinh2u= shu*shu;
    double gv[3]= { sinh2u, 1., 1./sinh2u };
    double gu[3]= { 2.*shu*chu, 0., -2.*chu/(sinh2u*shu) };
    double SP[3]= { sinh2u, -Ld2/sinh2u, -1. };
    double s2= si*si, u_c[5], Sc[5];
    for (c=0;c<5;c++){
      u_c[c]= (1.-s2)*dbase[c]+s2*dux[c];
      Sc[c]= SP[0]*dE[c]+SP[1]*dLz[c]+SP[2]*dI3Ut[c];
    }
    for (k=0;k<3;k++){
      Pval[k]+= wi*2.*mid*mid*si*(gv[k]/sq);
      for (c=0;c<5;c++){
	double term1= 4.*mid*dmid[c]*si*(gv[k]/sq);
	double term2= 2.*mid*mid*si*( (gu[k]/sq)*u_c[c]
				      -0.5*(gv[k]/S15)*(Su*u_c[c]+Sc[c]) );
	dP[k*5+c]+= wi*(term1+term2);
      }
    }
  }
}
// v-side analogue; 4 params {E,Lz,I3V,u0} (S_v genuinely depends on u0). High-v
// panel base=pi/2 is a regular point (dbase=0, no turning-point cancellation).
// dvx here is the reflected endpoint motion (caller passes -dvx when vx>pi/2).
void calcAnglePartialDerivV(struct dJzStaeckelArg * p,
			    gsl_integration_glfixed_table * T,int order,
			    double base,double sign,double mid,double Ld2,
			    int high_panel,
			    double * dE,double * dLz,double * dI3V,double * du0,
			    double * dvx,double * Qval,double * dQ){
  int gi,k,c;
  for (k=0;k<3;k++){ Qval[k]= 0.; for (c=0;c<5;c++) dQ[k*5+c]= 0.; }
  double dbase[5],dmid[5];
  if ( high_panel ){
    for (c=0;c<5;c++) dbase[c]= 0.;
  } else {
    double svb= sin(base), sin2b= svb*svb;
    double SPbase[4]= { sin2b, -Ld2/sin2b, 1., JzStaeckel_dSdu0(base,p) };
    double Sv_base= JzStaeckel_dSdv(base,p);
    for (c=0;c<5;c++)
      dbase[c]= (-SPbase[0]/Sv_base)*dE[c]+(-SPbase[1]/Sv_base)*dLz[c]
	+(-SPbase[2]/Sv_base)*dI3V[c]+(-SPbase[3]/Sv_base)*du0[c];
  }
  for (c=0;c<5;c++) dmid[c]= sign*(dvx[c]-dbase[c])/(2.*mid);
  double si,wi;
  for (gi=0;gi<order;gi++){
    gsl_integration_glfixed_point(0.,1.,gi,&si,&wi,T);
    double t= mid*si, v= base+sign*t*t;
    double S= JzStaeckelIntegrandSquared4dJz(v,p);
    if ( S <= 0. ) continue;
    double sq= sqrt(S), S15= S*sq;
    double Sv= JzStaeckel_dSdv(v,p);
    double sv= sin(v), cv= cos(v), sin2v= sv*sv;
    double gv[3]= { sin2v, 1., 1./sin2v };
    double gu[3]= { 2.*sv*cv, 0., -2.*cv/(sin2v*sv) };
    double SP[4]= { sin2v, -Ld2/sin2v, 1., JzStaeckel_dSdu0(v,p) };
    double s2= si*si, v_c[5], Sc[5];
    for (c=0;c<5;c++){
      v_c[c]= (1.-s2)*dbase[c]+s2*dvx[c];
      Sc[c]= SP[0]*dE[c]+SP[1]*dLz[c]+SP[2]*dI3V[c]+SP[3]*du0[c];
    }
    for (k=0;k<3;k++){
      Qval[k]+= wi*2.*mid*mid*si*(gv[k]/sq);
      for (c=0;c<5;c++){
	double term1= 4.*mid*dmid[c]*si*(gv[k]/sq);
	double term2= 2.*mid*mid*si*( (gu[k]/sq)*v_c[c]
				      -0.5*(gv[k]/S15)*(Sv*v_c[c]+Sc[c]) );
	dQ[k*5+c]+= wi*(term1+term2);
      }
    }
  }
}
// c=True C-native ANGLE Jacobian (#131 PR-B): emits jr,jz,Omega{r,phi,z} + the
// three angles (values, angles fmod-wrapped as calcAnglesStaeckel, anglephi WITHOUT
// phi) + ojac (N*25, byte-identical to actionsFreqsJac) + ajac (N*15 = 3x5
// d(angler,anglephi,anglez)/d(R,vR,vT,z,vz)). The angle rows chain the outer
// dOmega/d(dI3dJ) quotient-rule gradients through the action Hessians (as the freq
// rows) PLUS the novel partial-range integral derivatives (calcAnglePartialDeriv).
EXPORT void actionAngleStaeckel_actionsFreqsAnglesJac(int ndata,
    double *R,double *vR,double *vT,double *z,double *vz,double *u0,
    int npot,int *pot_type,double *pot_args,tfuncs_type_arr pot_tfuncs,
    int ndelta,double *delta,int order,int useu0,
    double *jr,double *jz,double *Or,double *Op,double *Oz,
    double *Angler,double *Anglephi,double *Anglez,
    double *ojac,double *ajac,int *err){
  int ii; double tdelta;
  struct potentialArg * aaArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,aaArgs,&pot_type,&pot_args,&pot_tfuncs);
  double *E=malloc(ndata*sizeof(double)),*Lz=malloc(ndata*sizeof(double));
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,aaArgs);
  double *ux=malloc(ndata*sizeof(double)),*vx=malloc(ndata*sizeof(double));
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  double *shx=malloc(ndata*sizeof(double)),*chx=malloc(ndata*sizeof(double));
  double *svx=malloc(ndata*sizeof(double)),*cvx=malloc(ndata*sizeof(double));
  double *pux=malloc(ndata*sizeof(double)),*pvx=malloc(ndata*sizeof(double));
  double *sinh2u0=malloc(ndata*sizeof(double)),*cosh2u0=malloc(ndata*sizeof(double));
  double *v0=malloc(ndata*sizeof(double)),*sin2v0=malloc(ndata*sizeof(double));
  double *potu0v0=malloc(ndata*sizeof(double)),*potupi2=malloc(ndata*sizeof(double));
  double *I3U=malloc(ndata*sizeof(double)),*I3V=malloc(ndata*sizeof(double));
  int ds= ndelta==1?0:1;
  for (ii=0;ii<ndata;ii++){
    tdelta= *(delta+ii*ds);
    chx[ii]=cosh(ux[ii]); shx[ii]=sinh(ux[ii]); cvx[ii]=cos(vx[ii]); svx[ii]=sin(vx[ii]);
    pux[ii]=tdelta*(vR[ii]*chx[ii]*svx[ii]+vz[ii]*shx[ii]*cvx[ii]);
    pvx[ii]=tdelta*(vR[ii]*shx[ii]*cvx[ii]-vz[ii]*chx[ii]*svx[ii]);
    sinh2u0[ii]=sinh(u0[ii])*sinh(u0[ii]); cosh2u0[ii]=cosh(u0[ii])*cosh(u0[ii]);
    v0[ii]=0.5*M_PI; sin2v0[ii]=sin(v0[ii])*sin(v0[ii]);
    potu0v0[ii]=evaluatePotentialsUV(u0[ii],v0[ii],tdelta,npot,aaArgs);
    I3U[ii]=E[ii]*shx[ii]*shx[ii]-0.5*pux[ii]*pux[ii]/tdelta/tdelta
      -0.5*Lz[ii]*Lz[ii]/tdelta/tdelta/shx[ii]/shx[ii]
      -(shx[ii]*shx[ii]+sin2v0[ii])*evaluatePotentialsUV(ux[ii],v0[ii],tdelta,npot,aaArgs)
      +(sinh2u0[ii]+sin2v0[ii])*potu0v0[ii];
    potupi2[ii]=evaluatePotentialsUV(u0[ii],0.5*M_PI,tdelta,npot,aaArgs);
    I3V[ii]=-E[ii]*svx[ii]*svx[ii]+0.5*pvx[ii]*pvx[ii]/tdelta/tdelta
      +0.5*Lz[ii]*Lz[ii]/tdelta/tdelta/svx[ii]/svx[ii]-cosh2u0[ii]*potupi2[ii]
      +(sinh2u0[ii]+svx[ii]*svx[ii])*evaluatePotentialsUV(u0[ii],vx[ii],tdelta,npot,aaArgs);
  }
  double *umin=malloc(ndata*sizeof(double)),*umax=malloc(ndata*sizeof(double)),*vmin=malloc(ndata*sizeof(double));
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs);
  calcJRStaeckel(ndata,jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs,order);
  calcJzStaeckel(ndata,jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  double *djrdE=malloc(ndata*sizeof(double)),*djrdLz=malloc(ndata*sizeof(double)),*djrdI3=malloc(ndata*sizeof(double));
  double *djzdE=malloc(ndata*sizeof(double)),*djzdLz=malloc(ndata*sizeof(double)),*djzdI3=malloc(ndata*sizeof(double));
  double *djzdU0=malloc(ndata*sizeof(double)),*detA=malloc(ndata*sizeof(double));
  calcdJRStaeckel(ndata,djrdE,djrdLz,djrdI3,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs,order);
  calcdJzStaeckel(ndata,djzdE,djzdLz,djzdI3,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  calcdJzdU0Staeckel(ndata,djzdU0,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  calcFreqsFromDerivsStaeckel(ndata,Or,Op,Oz,detA,djrdE,djrdLz,djrdI3,djzdE,djzdLz,djzdI3);
  double *dI3dJR=malloc(ndata*sizeof(double)),*dI3dJz=malloc(ndata*sizeof(double)),*dI3dLz=malloc(ndata*sizeof(double));
  calcdI3dJFromDerivsStaeckel(ndata,dI3dJR,dI3dJz,dI3dLz,detA,djrdE,djzdE,djrdLz,djzdLz);
  double *d2jr=malloc(ndata*9*sizeof(double)),*d2jz=malloc(ndata*12*sizeof(double));
  calcd2JRStaeckel(ndata,d2jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,npot,aaArgs,order);
  calcd2JzStaeckel(ndata,d2jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,npot,aaArgs,order);
  // angle VALUES (reused value path -> byte-identical to actionsFreqsAngles_c)
  calcAnglesStaeckel(ndata,Angler,Anglephi,Anglez,Or,Op,Oz,dI3dJR,dI3dJz,dI3dLz,
		     djrdE,djrdLz,djrdI3,djzdE,djzdLz,djzdI3,ux,vx,pux,pvx,
		     umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,potu0v0,
		     vmin,I3V,cosh2u0,potupi2,npot,aaArgs,order);
  gsl_integration_glfixed_table * Tang= gsl_integration_glfixed_table_alloc(order);
  for (ii=0;ii<ndata;ii++){
    int kk;
    tdelta= *(delta+ii*ds);
    double sh=shx[ii],ch=chx[ii],sv=svx[ii],cv=cvx[ii];
    double tu0=u0[ii],sh0=sinh(tu0),ch0=cosh(tu0);
    double tE=E[ii],tLz=Lz[ii],tpux=pux[ii],tpvx=pvx[ii],tvR=vR[ii],tvT=vT[ii],tvz=vz[ii];
    double D=sh*sh+sv*sv;
    double dux_dR=ch*sv/(tdelta*D),dux_dz=sh*cv/(tdelta*D);
    double dvx_dR=sh*cv/(tdelta*D),dvx_dz=-ch*sv/(tdelta*D);
    double dux[5]={dux_dR,0.,0.,dux_dz,0.},dvx[5]={dvx_dR,0.,0.,dvx_dz,0.};
    double dE[5]={-calcRforce(R[ii],z[ii],0.,0.,npot,aaArgs),tvR,tvT,
                  -calczforce(R[ii],z[ii],0.,0.,npot,aaArgs),tvz};
    double dLz[5]={tvT,0.,R[ii],0.,0.};
    double du0[5],du0dE=0.,du0dLz=0.;
    if ( useu0==2 ){
      double L2=0.5*tLz*tLz/(tdelta*tdelta),hh=1.e-5;
      double fpp=( staeckelU0Stationarity(tu0+hh,tE,L2,tdelta,npot,aaArgs)
                  -staeckelU0Stationarity(tu0-hh,tE,L2,tdelta,npot,aaArgs) )/(2.*hh);
      du0dE=-2.*sh0*ch0/fpp; du0dLz=-2.*ch0*tLz/(tdelta*tdelta*sh0*sh0*sh0)/fpp;
    }
    for (kk=0;kk<5;kk++) du0[kk]= useu0==0?dux[kk]:(useu0==1?0.:du0dE*dE[kk]+du0dLz*dLz[kk]);
    double dpux_dux=tdelta*(tvR*sh*sv+tvz*ch*cv),dpux_dvx=tdelta*(tvR*ch*cv-tvz*sh*sv);
    double dpvx_dux=tdelta*(tvR*ch*cv-tvz*sh*sv),dpvx_dvx=tdelta*(-tvR*sh*sv-tvz*ch*cv);
    double dpux[5],dpvx[5];
    for (kk=0;kk<5;kk++){ dpux[kk]=dpux_dux*dux[kk]+dpux_dvx*dvx[kk]; dpvx[kk]=dpvx_dux*dux[kk]+dpvx_dvx*dvx[kk]; }
    dpux[1]+=tdelta*ch*sv; dpux[4]+=tdelta*sh*cv; dpvx[1]+=tdelta*sh*cv; dpvx[4]+=-tdelta*ch*sv;
    double Pux=evaluatePotentialsUV(ux[ii],0.5*M_PI,tdelta,npot,aaArgs);
    double FRux=calcRforce(tdelta*sh,0.,0.,0.,npot,aaArgs),dPux_dux=-FRux*tdelta*ch;
    double dI3Ut_dE=sh*sh,dI3Ut_dLz=-tLz/(tdelta*tdelta*sh*sh),dI3Ut_dpux=-tpux/(tdelta*tdelta);
    double dI3Ut_dux=2.*sh*ch*tE+tLz*tLz*ch/(tdelta*tdelta*sh*sh*sh)-2.*sh*ch*Pux-(sh*sh+1.)*dPux_dux;
    double P0v=evaluatePotentialsUV(tu0,vx[ii],tdelta,npot,aaArgs);
    double FRu0=calcRforce(tdelta*sh0,0.,0.,0.,npot,aaArgs);
    double Rp=tdelta*sh0*sv,zp=tdelta*ch0*cv;
    double FRp=calcRforce(Rp,zp,0.,0.,npot,aaArgs),Fzp=calczforce(Rp,zp,0.,0.,npot,aaArgs);
    double dPu0_du0=-FRu0*tdelta*ch0;
    double dP0v_dvx=-FRp*tdelta*sh0*cv+Fzp*tdelta*ch0*sv,dP0v_du0=-FRp*tdelta*ch0*sv-Fzp*tdelta*sh0*cv;
    double dI3V_dE=-sv*sv,dI3V_dLz=tLz/(tdelta*tdelta*sv*sv),dI3V_dpvx=tpvx/(tdelta*tdelta);
    double dI3V_dvx=-2.*tE*sv*cv-tLz*tLz*cv/(tdelta*tdelta*sv*sv*sv)+2.*sv*cv*P0v+(sh0*sh0+sv*sv)*dP0v_dvx;
    double dI3V_du0=-2.*ch0*sh0*potupi2[ii]-ch0*ch0*dPu0_du0+2.*sh0*ch0*P0v+(sh0*sh0+sv*sv)*dP0v_du0;
    double dI3Ut[5],dI3V_[5];
    for (kk=0;kk<5;kk++){
      dI3Ut[kk]=dI3Ut_dE*dE[kk]+dI3Ut_dLz*dLz[kk]+dI3Ut_dpux*dpux[kk]+dI3Ut_dux*dux[kk];
      dI3V_[kk]=dI3V_dE*dE[kk]+dI3V_dLz*dLz[kk]+dI3V_dpvx*dpvx[kk]+dI3V_dvx*dvx[kk]+dI3V_du0*du0[kk];
    }
    // ojac action rows (0,1) -- byte-identical to actionsFreqsJac
    for (kk=0;kk<5;kk++){
      ojac[ii*25+kk]= djrdE[ii]*dE[kk]+djrdLz[ii]*dLz[kk]+djrdI3[ii]*dI3Ut[kk];
      ojac[ii*25+5+kk]= djzdE[ii]*dE[kk]+djzdLz[ii]*dLz[kk]+djzdI3[ii]*dI3V_[kk]+djzdU0[ii]*du0[kk];
    }
    double a=djrdE[ii],b=djrdLz[ii],cJR=djrdI3[ii],dJZ=djzdE[ii],eJZ=djzdLz[ii],fJZ=djzdI3[ii];
    if ( a==9999.99 || dJZ==9999.99 || detA[ii]==0. ){
      for (kk=0;kk<15;kk++) ojac[ii*25+10+kk]=0.;
      for (kk=0;kk<15;kk++) ajac[ii*15+kk]=0.;
      continue; // zero freq + angle rows
    }
    double id=1./detA[ii],id2=id*id,N=cJR*eJZ-fJZ*b,Mm=a*eJZ-dJZ*b;
    double gOr[6]={-fJZ*fJZ*id2,0.,fJZ*dJZ*id2,fJZ*cJR*id2,0.,id-fJZ*a*id2};
    double gOp[6]={-N*fJZ*id2,-fJZ*id,eJZ*id+N*dJZ*id2,N*cJR*id2,cJR*id,-b*id-N*a*id2};
    double gOz[6]={cJR*fJZ*id2,0.,-id-cJR*dJZ*id2,-cJR*cJR*id2,0.,cJR*a*id2};
    // quotient-rule gradients of dI3dJR=-dJZ/detA, dI3dJz=a/detA,
    // dI3dLz=-(a*eJZ-dJZ*b)/detA in {a,b,cJR,dJZ,eJZ,fJZ} (Mm=a*eJZ-dJZ*b)
    double gI3R[6]={fJZ*dJZ*id2,0.,-dJZ*dJZ*id2,-id-dJZ*cJR*id2,0.,dJZ*a*id2};
    double gI3z[6]={id-a*fJZ*id2,0.,a*dJZ*id2,a*cJR*id2,0.,-a*a*id2};
    double gI3Lz[6]={-eJZ*id+Mm*fJZ*id2,dJZ*id,-Mm*dJZ*id2,b*id-Mm*cJR*id2,-a*id,Mm*a*id2};
    double *gall[6]={gOr,gOp,gOz,gI3R,gI3z,gI3Lz};
    double *D2R=d2jr+ii*9,*D2Z=d2jz+ii*12;
    double drow[6][5];
    for (int oi=0;oi<6;oi++){
      double *g=gall[oi];
      double dOdE=g[0]*D2R[0]+g[1]*D2R[3]+g[2]*D2R[6]+g[3]*D2Z[0]+g[4]*D2Z[4]+g[5]*D2Z[8];
      double dOdLz=g[0]*D2R[1]+g[1]*D2R[4]+g[2]*D2R[7]+g[3]*D2Z[1]+g[4]*D2Z[5]+g[5]*D2Z[9];
      double dOdI3U=g[0]*D2R[2]+g[1]*D2R[5]+g[2]*D2R[8];
      double dOdI3V=g[3]*D2Z[2]+g[4]*D2Z[6]+g[5]*D2Z[10];
      double dOdu0=g[3]*D2Z[3]+g[4]*D2Z[7]+g[5]*D2Z[11];
      for (kk=0;kk<5;kk++)
	drow[oi][kk]=dOdE*dE[kk]+dOdLz*dLz[kk]+dOdI3U*dI3Ut[kk]+dOdI3V*dI3V_[kk]+dOdu0*du0[kk];
    }
    // ojac freq rows (2,3,4) -- byte-identical to actionsFreqsJac
    for (int oi=0;oi<3;oi++)
      for (kk=0;kk<5;kk++) ojac[ii*25+(oi+2)*5+kk]=drow[oi][kk];
    // ----- angle Jacobian rows (ajac) -----
    double tumin=umin[ii],tumax=umax[ii],tux=ux[ii];
    double midpt_u= tumin+0.5*(tumax-tumin);
    int high_u= tux>midpt_u;
    double base_u= high_u? tumax:tumin, sign_u= high_u? -1.:1.;
    double mid_u= sqrt( high_u? (tumax-tux) : (tux-tumin) );
    double Ku= high_u? M_PI : ( tpux>0.? 0. : 2.*M_PI );
    double su= high_u? (tpux>0.? -1.:1.) : (tpux>0.? 1.:-1.);
    double tvmin=vmin[ii],tvx=vx[ii];
    double midpt_v= tvmin+0.5*(0.5*M_PI-tvmin);
    int low_v= (tvx<midpt_v) || (tvx>(M_PI-midpt_v));
    int above= tvx>0.5*M_PI;
    double base_v= low_v? tvmin:0.5*M_PI, sign_v= low_v? 1.:-1.;
    double mid_v= low_v? ( above? sqrt(fabs(M_PI-tvx-tvmin)) : sqrt(fabs(tvx-tvmin)) )
                       : sqrt(fabs(0.5*M_PI-tvx));
    double Kv,sv2;
    if ( low_v ){
      if ( tpvx>0. ){ Kv= above? M_PI:0.; sv2= above? -1.:1.; }
      else { Kv= above? M_PI:2.*M_PI; sv2= above? 1.:-1.; }
    } else {
      if ( tpvx>0. ){ Kv= 0.5*M_PI; sv2= above? 1.:-1.; }
      else { Kv= 1.5*M_PI; sv2= above? -1.:1.; }
    }
    // near-turning-point guard (z=0 -> mid_v=0; position at a panel base) -> zero
    if ( mid_u<1.e-7 || mid_v<1.e-7 ){
      for (kk=0;kk<15;kk++) ajac[ii*15+kk]=0.;
      continue;
    }
    double Ld2= tLz/(tdelta*tdelta);
    double dvx_eff[5];
    for (kk=0;kk<5;kk++) dvx_eff[kk]= above? -dvx[kk]:dvx[kk];
    struct dJRStaeckelArg pu;
    pu.E=tE; pu.Lz22delta=0.5*tLz*tLz/(tdelta*tdelta); pu.I3U=I3U[ii]; pu.delta=tdelta;
    pu.u0=tu0; pu.sinh2u0=sinh2u0[ii]; pu.v0=v0[ii]; pu.sin2v0=sin2v0[ii];
    pu.potu0v0=potu0v0[ii]; pu.umin=tumin; pu.umax=tumax; pu.nargs=npot; pu.actionAngleArgs=aaArgs;
    struct dJzStaeckelArg pv;
    pv.E=tE; pv.Lz22delta=0.5*tLz*tLz/(tdelta*tdelta); pv.I3V=I3V[ii]; pv.delta=tdelta;
    pv.u0=tu0; pv.cosh2u0=cosh2u0[ii]; pv.sinh2u0=sinh2u0[ii]; pv.potupi2=potupi2[ii];
    pv.vmin=tvmin; pv.nargs=npot; pv.actionAngleArgs=aaArgs;
    double Pval[3],Qval[3],dPu[15],dQv[15];
    calcAnglePartialDerivU(&pu,Tang,order,base_u,sign_u,mid_u,Ld2,dE,dLz,dI3Ut,dux,Pval,dPu);
    calcAnglePartialDerivV(&pv,Tang,order,base_v,sign_v,mid_v,Ld2,!low_v,dE,dLz,dI3V_,du0,dvx_eff,Qval,dQv);
    double pr=tdelta/sqrt(2.), aL=tLz/tdelta/sqrt(2.), aLc=1./tdelta/sqrt(2.);
    double Or1=Ku*djrdE[ii]+su*pr*Pval[0], I3r1=Ku*djrdI3[ii]-su*pr*Pval[1], PLv=Pval[2];
    double Or2=Kv*djzdE[ii]+sv2*pr*Qval[0], I3r2=Kv*djzdI3[ii]+sv2*pr*Qval[1], QLv=Qval[2];
    double Or_sum=Or1+Or2, I3_sum=I3r1+I3r2;
    // full-range action-Hessian chains d(djXdY)/dc
    double dOr_sum[5],dI3_sum[5],daphi_u[5],dphitmp[5];
    for (kk=0;kk<5;kk++){
      double ddjrdE= D2R[0]*dE[kk]+D2R[1]*dLz[kk]+D2R[2]*dI3Ut[kk];
      double ddjrdLz= D2R[3]*dE[kk]+D2R[4]*dLz[kk]+D2R[5]*dI3Ut[kk];
      double ddjrdI3= D2R[6]*dE[kk]+D2R[7]*dLz[kk]+D2R[8]*dI3Ut[kk];
      double ddjzdE= D2Z[0]*dE[kk]+D2Z[1]*dLz[kk]+D2Z[2]*dI3V_[kk]+D2Z[3]*du0[kk];
      double ddjzdLz= D2Z[4]*dE[kk]+D2Z[5]*dLz[kk]+D2Z[6]*dI3V_[kk]+D2Z[7]*du0[kk];
      double ddjzdI3= D2Z[8]*dE[kk]+D2Z[9]*dLz[kk]+D2Z[10]*dI3V_[kk]+D2Z[11]*du0[kk];
      dOr_sum[kk]= Ku*ddjrdE+Kv*ddjzdE+su*pr*dPu[0*5+kk]+sv2*pr*dQv[0*5+kk];
      dI3_sum[kk]= Ku*ddjrdI3+Kv*ddjzdI3-su*pr*dPu[1*5+kk]+sv2*pr*dQv[1*5+kk];
      daphi_u[kk]= Ku*ddjrdLz-aL*su*dPu[2*5+kk]-aLc*dLz[kk]*su*PLv;
      dphitmp[kk]= Kv*ddjzdLz-aL*sv2*dQv[2*5+kk]-aLc*dLz[kk]*sv2*QLv;
    }
    double Omr=Or[ii],Omp=Op[ii],Omz=Oz[ii];
    double dI3JR=dI3dJR[ii],dI3Jz=dI3dJz[ii],dI3Lz2=dI3dLz[ii];
    for (kk=0;kk<5;kk++){
      ajac[ii*15+0*5+kk]= drow[0][kk]*Or_sum+Omr*dOr_sum[kk]+drow[3][kk]*I3_sum+dI3JR*dI3_sum[kk];
      ajac[ii*15+1*5+kk]= daphi_u[kk]+dphitmp[kk]+drow[1][kk]*Or_sum+Omp*dOr_sum[kk]
	+drow[5][kk]*I3_sum+dI3Lz2*dI3_sum[kk];
      ajac[ii*15+2*5+kk]= drow[2][kk]*Or_sum+Omz*dOr_sum[kk]+drow[4][kk]*I3_sum+dI3Jz*dI3_sum[kk];
    }
  }
  gsl_integration_glfixed_table_free(Tang);
  free_potentialArgs(npot,aaArgs); free(aaArgs);
  free(E);free(Lz);free(ux);free(vx);free(shx);free(chx);free(svx);free(cvx);
  free(pux);free(pvx);free(sinh2u0);free(cosh2u0);free(v0);free(sin2v0);
  free(potu0v0);free(potupi2);free(I3U);free(I3V);free(umin);free(umax);free(vmin);
  free(djrdE);free(djrdLz);free(djrdI3);free(djzdE);free(djzdLz);free(djzdI3);
  free(djzdU0);free(detA);free(dI3dJR);free(dI3dJz);free(dI3dLz);free(d2jr);free(d2jz);
  *err=0;
}
void calcJRStaeckel(int ndata,
		    double * jr,
		    double * umin,
		    double * umax,
		    double * E,
		    double * Lz,
		    double * I3U,
		    int ndelta,
		    double * delta,
		    double * u0,
		    double * sinh2u0,
		    double * v0,
		    double * sin2v0,
		    double * potu0v0,
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
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) malloc ( nthreads * sizeof (struct JRStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii)							\
  shared(jr,umin,umax,JRInt,params,T,delta,E,Lz,I3U,u0,sinh2u0,v0,sin2v0,potu0v0)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(umin+ii) == -9999.99 || *(umax+ii) == -9999.99 ){
      *(jr+ii)= 9999.99;
      continue;
    }
    if ( (*(umax+ii) - *(umin+ii)) / *(umax+ii) < 0.000001 ){//circular
      *(jr+ii) = 0.;
      continue;
    }
    //Setup function
    (params+tid)->delta= *(delta+ii*delta_stride);
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (params+tid)->I3U= *(I3U+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->v0= *(v0+ii);
    (params+tid)->sin2v0= *(sin2v0+ii);
    (params+tid)->potu0v0= *(potu0v0+ii);
    (JRInt+tid)->function = &JRStaeckelIntegrand;
    (JRInt+tid)->params = params+tid;
    //Integrate
    *(jr+ii)= gsl_integration_glfixed (JRInt+tid,*(umin+ii),*(umax+ii),T)
      * sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
  }
  free(JRInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
void calcJzStaeckel(int ndata,
		    double * jz,
		    double * vmin,
		    double * E,
		    double * Lz,
		    double * I3V,
		    int ndelta,
		    double * delta,
		    double * u0,
		    double * cosh2u0,
		    double * sinh2u0,
		    double * potupi2,
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
  struct JzStaeckelArg * params= (struct JzStaeckelArg *) malloc ( nthreads * sizeof (struct JzStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii)							\
  shared(jz,vmin,JzInt,params,T,delta,E,Lz,I3V,u0,cosh2u0,sinh2u0,potupi2)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(vmin+ii) == -9999.99 ){
      *(jz+ii)= 9999.99;
      continue;
    }
    if ( (0.5 * M_PI - *(vmin+ii)) / M_PI * 2. < 0.000001 ){//circular
      *(jz+ii) = 0.;
      continue;
    }
    //Setup function
    (params+tid)->delta= *(delta+ii*delta_stride);
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (params+tid)->I3V= *(I3V+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->cosh2u0= *(cosh2u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->potupi2= *(potupi2+ii);
    (JzInt+tid)->function = &JzStaeckelIntegrand;
    (JzInt+tid)->params = params+tid;
    //Integrate
    *(jz+ii)= gsl_integration_glfixed (JzInt+tid,*(vmin+ii),M_PI/2.,T)
      * 2 * sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
  }
  free(JzInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
void actionAngleStaeckel_actionsFreqs(int ndata,
				      double *R,
				      double *vR,
				      double *vT,
				      double *z,
				      double *vz,
				      double *u0,
				      int npot,
				      int * pot_type,
				      double * pot_args,
              tfuncs_type_arr pot_tfuncs,
				      int ndelta,
				      double * delta,
				      int order,
				      double *jr,
				      double *jz,
				      double *Omegar,
				      double *Omegaphi,
				      double *Omegaz,
				      int * err){
  int ii;
  double tdelta;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  double *coshux= (double *) malloc ( ndata * sizeof(double) );
  double *sinhux= (double *) malloc ( ndata * sizeof(double) );
  double *sinvx= (double *) malloc ( ndata * sizeof(double) );
  double *cosvx= (double *) malloc ( ndata * sizeof(double) );
  double *pux= (double *) malloc ( ndata * sizeof(double) );
  double *pvx= (double *) malloc ( ndata * sizeof(double) );
  double *sinh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *cosh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *v0= (double *) malloc ( ndata * sizeof(double) );
  double *sin2v0= (double *) malloc ( ndata * sizeof(double) );
  double *potu0v0= (double *) malloc ( ndata * sizeof(double) );
  double *potupi2= (double *) malloc ( ndata * sizeof(double) );
  double *I3U= (double *) malloc ( ndata * sizeof(double) );
  double *I3V= (double *) malloc ( ndata * sizeof(double) );
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    tdelta= *(delta+ii*delta_stride);
    *(coshux+ii)= cosh(*(ux+ii));
    *(sinhux+ii)= sinh(*(ux+ii));
    *(cosvx+ii)= cos(*(vx+ii));
    *(sinvx+ii)= sin(*(vx+ii));
    *(pux+ii)= tdelta * (*(vR+ii) * *(coshux+ii) * *(sinvx+ii)
			+ *(vz+ii) * *(sinhux+ii) * *(cosvx+ii));
    *(pvx+ii)= tdelta * (*(vR+ii) * *(sinhux+ii) * *(cosvx+ii)
			- *(vz+ii) * *(coshux+ii) * *(sinvx+ii));
    *(sinh2u0+ii)= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(cosh2u0+ii)= cosh(*(u0+ii)) * cosh(*(u0+ii));
    *(v0+ii)= 0.5 * M_PI; //*(vx+ii);
    *(sin2v0+ii)= sin(*(v0+ii)) * sin(*(v0+ii));
    *(potu0v0+ii)= evaluatePotentialsUV(*(u0+ii),*(v0+ii),tdelta,
					npot,actionAngleArgs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,actionAngleArgs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,actionAngleArgs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,actionAngleArgs);
  }
  //Calculate 'peri' and 'apo'centers
  double *umin= (double *) malloc ( ndata * sizeof(double) );
  double *umax= (double *) malloc ( ndata * sizeof(double) );
  double *vmin= (double *) malloc ( ndata * sizeof(double) );
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,
	       sin2v0,potu0v0,npot,actionAngleArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,
	   npot,actionAngleArgs);
  //Calculate the actions
  calcJRStaeckel(ndata,jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,
		 potu0v0,npot,actionAngleArgs,order);
  calcJzStaeckel(ndata,jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,
		 potupi2,npot,actionAngleArgs,order);
  //Calculate the derivatives of the actions wrt the integrals of motion
  double *dJRdE= (double *) malloc ( ndata * sizeof(double) );
  double *dJRdLz= (double *) malloc ( ndata * sizeof(double) );
  double *dJRdI3= (double *) malloc ( ndata * sizeof(double) );
  double *dJzdE= (double *) malloc ( ndata * sizeof(double) );
  double *dJzdLz= (double *) malloc ( ndata * sizeof(double) );
  double *dJzdI3= (double *) malloc ( ndata * sizeof(double) );
  double *detA= (double *) malloc ( ndata * sizeof(double) );
  calcdJRStaeckel(ndata,dJRdE,dJRdLz,dJRdI3,
		  umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,
		  potu0v0,npot,actionAngleArgs,order);
  calcdJzStaeckel(ndata,dJzdE,dJzdLz,dJzdI3,
		  vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,
		  potupi2,npot,actionAngleArgs,order);
  calcFreqsFromDerivsStaeckel(ndata,Omegar,Omegaphi,Omegaz,detA,
			      dJRdE,dJRdLz,dJRdI3,
			      dJzdE,dJzdLz,dJzdI3);
  //Free
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(E);
  free(Lz);
  free(ux);
  free(vx);
  free(coshux);
  free(sinhux);
  free(sinvx);
  free(cosvx);
  free(pux);
  free(pvx);
  free(sinh2u0);
  free(cosh2u0);
  free(v0);
  free(sin2v0);
  free(potu0v0);
  free(potupi2);
  free(I3U);
  free(I3V);
  free(umin);
  free(umax);
  free(vmin);
  free(dJRdE);
  free(dJRdLz);
  free(dJRdI3);
  free(dJzdE);
  free(detA);
  free(dJzdLz);
  free(dJzdI3);
}
void actionAngleStaeckel_actionsFreqsAngles(int ndata,
					    double *R,
					    double *vR,
					    double *vT,
					    double *z,
					    double *vz,
					    double *u0,
					    int npot,
					    int * pot_type,
					    double * pot_args,
              tfuncs_type_arr pot_tfuncs,
					    int ndelta,
					    double * delta,
					    int order,
					    double *jr,
					    double *jz,
					    double *Omegar,
					    double *Omegaphi,
					    double *Omegaz,
					    double *Angler,
					    double *Anglephi,
					    double *Anglez,
					    int * err){
  int ii;
  double tdelta;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  double *coshux= (double *) malloc ( ndata * sizeof(double) );
  double *sinhux= (double *) malloc ( ndata * sizeof(double) );
  double *sinvx= (double *) malloc ( ndata * sizeof(double) );
  double *cosvx= (double *) malloc ( ndata * sizeof(double) );
  double *pux= (double *) malloc ( ndata * sizeof(double) );
  double *pvx= (double *) malloc ( ndata * sizeof(double) );
  double *sinh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *cosh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *v0= (double *) malloc ( ndata * sizeof(double) );
  double *sin2v0= (double *) malloc ( ndata * sizeof(double) );
  double *potu0v0= (double *) malloc ( ndata * sizeof(double) );
  double *potupi2= (double *) malloc ( ndata * sizeof(double) );
  double *I3U= (double *) malloc ( ndata * sizeof(double) );
  double *I3V= (double *) malloc ( ndata * sizeof(double) );
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    tdelta= *(delta+ii*delta_stride);
    *(coshux+ii)= cosh(*(ux+ii));
    *(sinhux+ii)= sinh(*(ux+ii));
    *(cosvx+ii)= cos(*(vx+ii));
    *(sinvx+ii)= sin(*(vx+ii));
    *(pux+ii)= tdelta * (*(vR+ii) * *(coshux+ii) * *(sinvx+ii)
			+ *(vz+ii) * *(sinhux+ii) * *(cosvx+ii));
    *(pvx+ii)= tdelta * (*(vR+ii) * *(sinhux+ii) * *(cosvx+ii)
			- *(vz+ii) * *(coshux+ii) * *(sinvx+ii));
    *(sinh2u0+ii)= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(cosh2u0+ii)= cosh(*(u0+ii)) * cosh(*(u0+ii));
    *(v0+ii)= 0.5 * M_PI; //*(vx+ii);
    *(sin2v0+ii)= sin(*(v0+ii)) * sin(*(v0+ii));
    *(potu0v0+ii)= evaluatePotentialsUV(*(u0+ii),*(v0+ii),tdelta,
					npot,actionAngleArgs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,actionAngleArgs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,actionAngleArgs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,actionAngleArgs);
  }
  //Calculate 'peri' and 'apo'centers
  double *umin= (double *) malloc ( ndata * sizeof(double) );
  double *umax= (double *) malloc ( ndata * sizeof(double) );
  double *vmin= (double *) malloc ( ndata * sizeof(double) );
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,
	       sin2v0,potu0v0,npot,actionAngleArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,
	   npot,actionAngleArgs);
  //Calculate the actions
  calcJRStaeckel(ndata,jr,umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,
		 potu0v0,npot,actionAngleArgs,order);
  calcJzStaeckel(ndata,jz,vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,
		 potupi2,npot,actionAngleArgs,order);
  //Calculate the derivatives of the actions wrt the integrals of motion
  double *dJRdE= (double *) malloc ( ndata * sizeof(double) );
  double *dJRdLz= (double *) malloc ( ndata * sizeof(double) );
  double *dJRdI3= (double *) malloc ( ndata * sizeof(double) );
  double *dJzdE= (double *) malloc ( ndata * sizeof(double) );
  double *dJzdLz= (double *) malloc ( ndata * sizeof(double) );
  double *dJzdI3= (double *) malloc ( ndata * sizeof(double) );
  double *detA= (double *) malloc ( ndata * sizeof(double) );
  calcdJRStaeckel(ndata,dJRdE,dJRdLz,dJRdI3,
		  umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,
		  potu0v0,npot,actionAngleArgs,order);
  calcdJzStaeckel(ndata,dJzdE,dJzdLz,dJzdI3,
		  vmin,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,
		  potupi2,npot,actionAngleArgs,order);
  calcFreqsFromDerivsStaeckel(ndata,Omegar,Omegaphi,Omegaz,detA,
			      dJRdE,dJRdLz,dJRdI3,
			      dJzdE,dJzdLz,dJzdI3);
  double *dI3dJR= (double *) malloc ( ndata * sizeof(double) );
  double *dI3dJz= (double *) malloc ( ndata * sizeof(double) );
  double *dI3dLz= (double *) malloc ( ndata * sizeof(double) );
  calcdI3dJFromDerivsStaeckel(ndata,dI3dJR,dI3dJz,dI3dLz,detA,
			      dJRdE,dJzdE,dJRdLz,dJzdLz);
  calcAnglesStaeckel(ndata,Angler,Anglephi,Anglez,
		     Omegar,Omegaphi,Omegaz,dI3dJR,dI3dJz,dI3dLz,
		     dJRdE,dJRdLz,dJRdI3,
		     dJzdE,dJzdLz,dJzdI3,
		     ux,vx,pux,pvx,
		     umin,umax,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,sin2v0,
		     potu0v0,
		     vmin,I3V,cosh2u0,potupi2,
		     npot,actionAngleArgs,order);
  //Free
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(E);
  free(Lz);
  free(ux);
  free(vx);
  free(coshux);
  free(sinhux);
  free(sinvx);
  free(cosvx);
  free(pux);
  free(pvx);
  free(sinh2u0);
  free(cosh2u0);
  free(v0);
  free(sin2v0);
  free(potu0v0);
  free(potupi2);
  free(I3U);
  free(I3V);
  free(umin);
  free(umax);
  free(vmin);
  free(dJRdE);
  free(dJRdLz);
  free(dJRdI3);
  free(dJzdE);
  free(dJzdLz);
  free(dJzdI3);
  free(detA);
  free(dI3dJR);
  free(dI3dJz);
  free(dI3dLz);
}
void calcFreqsFromDerivsStaeckel(int ndata,
				 double * Omegar,
				 double * Omegaphi,
				 double * Omegaz,
				 double * detA,
				 double * djrdE,
				 double * djrdLz,
				 double * djrdI3,
				 double * djzdE,
				 double * djzdLz,
				 double * djzdI3){
  int ii;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)			\
  private(ii)							\
  shared(Omegar,Omegaphi,Omegaz,djrdE,djrdLz,djrdI3,djzdE,djzdLz,djzdI3,detA)
  for (ii=0; ii < ndata; ii++){
    if ( *(djrdE+ii) == 9999.99 || *(djzdE+ii) == 9999.99 ) {
      *(Omegar+ii)= 9999.99;
      *(Omegaz+ii)= 9999.99;
      *(Omegaphi+ii)= 9999.99;
    } else {
      //First calculate the determinant of the relevant matrix
      *(detA+ii)= *(djrdE+ii) * *(djzdI3+ii) - *(djzdE+ii) * *(djrdI3+ii);
      //Then calculate the frequencies
      *(Omegar+ii)= *(djzdI3+ii) / *(detA+ii);
      *(Omegaz+ii)= - *(djrdI3+ii) / *(detA+ii);
      *(Omegaphi+ii)= ( *(djrdI3+ii) * *(djzdLz+ii) - *(djzdI3+ii) * *(djrdLz+ii)) / *(detA+ii);
    }
  }
}
void calcdI3dJFromDerivsStaeckel(int ndata,
				 double * dI3dJR,
				 double * dI3dJz,
				 double * dI3dLz,
				 double * detA,
				 double * djrdE,
				 double * djzdE,
				 double * djrdLz,
				 double * djzdLz){
  int ii;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)			\
  private(ii)							\
  shared(djrdE,djzdE,djrdLz,djzdLz,dI3dJR,dI3dJz,dI3dLz,detA)
  for (ii=0; ii < ndata; ii++){
    *(dI3dJR+ii)= - *(djzdE+ii) / *(detA+ii);
    *(dI3dJz+ii)= *(djrdE+ii) / *(detA+ii);
    *(dI3dLz+ii)= -( *(djrdE+ii) * *(djzdLz+ii) - *(djzdE+ii) * *(djrdLz+ii) ) / *(detA+ii);
  }
}
void calcdJRStaeckel(int ndata,
		     double * djrdE,
		     double * djrdLz,
		     double * djrdI3,
		     double * umin,
		     double * umax,
		     double * E,
		     double * Lz,
		     double * I3U,
		     int ndelta,
		     double * delta,
		     double * u0,
		     double * sinh2u0,
		     double * v0,
		     double * sin2v0,
		     double * potu0v0,
		     int nargs,
		     struct potentialArg * actionAngleArgs,
		     int order){
  int ii, tid, nthreads;
  double mid;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * dJRInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) malloc ( nthreads * sizeof (struct dJRStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,mid)							\
  shared(djrdE,djrdLz,djrdI3,umin,umax,dJRInt,params,T,delta,E,Lz,I3U,u0,sinh2u0,v0,sin2v0,potu0v0)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(umin+ii) == -9999.99 || *(umax+ii) == -9999.99 ){
      *(djrdE+ii)= 9999.99;
      *(djrdLz+ii)= 9999.99;
      *(djrdI3+ii)= 9999.99;
      continue;
    }
    if ( (*(umax+ii) - *(umin+ii)) / *(umax+ii) < 0.000001 ){//circular
      *(djrdE+ii) = 0.;
      *(djrdLz+ii) = 0.;
      *(djrdI3+ii) = 0.;
      continue;
    }
    //Setup function
    (params+tid)->delta= *(delta+ii*delta_stride);
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (params+tid)->I3U= *(I3U+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->v0= *(v0+ii);
    (params+tid)->sin2v0= *(sin2v0+ii);
    (params+tid)->potu0v0= *(potu0v0+ii);
    (params+tid)->umin= *(umin+ii);
    (params+tid)->umax= *(umax+ii);
    (dJRInt+tid)->function = &dJRdELowStaeckelIntegrand;
    (dJRInt+tid)->params = params+tid;
    mid= sqrt( 0.5 * ( *(umax+ii) - *(umin+ii) ) );
    //Integrate to get djrdE
    *(djrdE+ii)= gsl_integration_glfixed (dJRInt+tid,0.,mid,T);
    (dJRInt+tid)->function = &dJRdEHighStaeckelIntegrand;
    *(djrdE+ii)+= gsl_integration_glfixed (dJRInt+tid,0.,mid,T);
    *(djrdE+ii)*= *(delta+ii*delta_stride) / M_PI / sqrt(2.);
    //then calculate djrdLz
    (dJRInt+tid)->function = &dJRdLzLowStaeckelIntegrand;
    *(djrdLz+ii)= gsl_integration_glfixed (dJRInt+tid,0.,mid,T);
    (dJRInt+tid)->function = &dJRdLzHighStaeckelIntegrand;
    *(djrdLz+ii)+= gsl_integration_glfixed (dJRInt+tid,0.,mid,T);
    *(djrdLz+ii)*= - *(Lz+ii) / M_PI / sqrt(2.) / *(delta+ii*delta_stride);
    //then calculate djrdI3
    (dJRInt+tid)->function = &dJRdI3LowStaeckelIntegrand;
    *(djrdI3+ii)= gsl_integration_glfixed (dJRInt+tid,0.,mid,T);
    (dJRInt+tid)->function = &dJRdI3HighStaeckelIntegrand;
    *(djrdI3+ii)+= gsl_integration_glfixed (dJRInt+tid,0.,mid,T);
    *(djrdI3+ii)*= - *(delta+ii*delta_stride) / M_PI / sqrt(2.);
  }
  free(dJRInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
void calcdJzStaeckel(int ndata,
		     double * djzdE,
		     double * djzdLz,
		     double * djzdI3,
		     double * vmin,
		     double * E,
		     double * Lz,
		     double * I3V,
		     int ndelta,
		     double * delta,
		     double * u0,
		     double * cosh2u0,
		     double * sinh2u0,
		     double * potupi2,
		     int nargs,
		     struct potentialArg * actionAngleArgs,
		     int order){
  int ii, tid, nthreads;
  double mid;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * dJzInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) malloc ( nthreads * sizeof (struct dJzStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,mid)							\
  shared(djzdE,djzdLz,djzdI3,vmin,dJzInt,params,T,delta,E,Lz,I3V,u0,cosh2u0,sinh2u0,potupi2)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(vmin+ii) == -9999.99 ){
      *(djzdE+ii)= 9999.99;
      *(djzdLz+ii)= 9999.99;
      *(djzdI3+ii)= 9999.99;
      continue;
    }
    if ( (0.5 * M_PI - *(vmin+ii)) / M_PI * 2. < 0.000001 ){//circular
      *(djzdE+ii) = 0.;
      *(djzdLz+ii) = 0.;
      *(djzdI3+ii) = 0.;
      continue;
    }
    //Setup function
    (params+tid)->delta= *(delta+ii*delta_stride);
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (params+tid)->I3V= *(I3V+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->cosh2u0= *(cosh2u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->potupi2= *(potupi2+ii);
    (params+tid)->vmin= *(vmin+ii);
    //First calculate dJzdE
    (dJzInt+tid)->function = &dJzdELowStaeckelIntegrand;
    (dJzInt+tid)->params = params+tid;
    mid= sqrt( 0.5 * (M_PI/2. - *(vmin+ii) ) );
    //BOVY: pv does not vanish at pi/2, so no need to break up the integral
    //Integrate
    *(djzdE+ii)= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    (dJzInt+tid)->function = &dJzdEHighStaeckelIntegrand;
    *(djzdE+ii)+= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    *(djzdE+ii)*= sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
    //Then calculate dJzdLz
    (dJzInt+tid)->function = &dJzdLzLowStaeckelIntegrand;
    //Integrate
    *(djzdLz+ii)= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    (dJzInt+tid)->function = &dJzdLzHighStaeckelIntegrand;
    *(djzdLz+ii)+= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    *(djzdLz+ii)*= - *(Lz+ii) * sqrt(2.) / M_PI / *(delta+ii*delta_stride);
    //Then calculate dJzdI3
    (dJzInt+tid)->function = &dJzdI3LowStaeckelIntegrand;
    //Integrate
    *(djzdI3+ii)= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    (dJzInt+tid)->function = &dJzdI3HighStaeckelIntegrand;
    *(djzdI3+ii)+= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    *(djzdI3+ii)*= sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
  }
  free(dJzInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
// dJz/du0 (E,Lz,I3V held fixed): u0 enters the J_z integrand directly through
// potentialStaeckel(u0,v) [and cosh2u0/sinh2u0/potupi2], so d(sqrt S_z)/du0 =
// (ddV/du0)/(2 sqrt S_z) with ddV/du0 a force evaluation along the u0-line.
// Same t^2-substituted low(vmin)+high(pi/2) GL panels and sqrt(2)*delta/pi
// prefactor as calcdJzStaeckel; degenerate (planar/unbound) -> 0.
void calcdJzdU0Staeckel(int ndata,
			double * djzdU0,
			double * vmin,
			double * E,
			double * Lz,
			double * I3V,
			int ndelta,
			double * delta,
			double * u0,
			double * cosh2u0,
			double * sinh2u0,
			double * potupi2,
			int nargs,
			struct potentialArg * actionAngleArgs,
			int order){
  int ii, tid, nthreads;
  double mid;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * dJzInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct dJzdU0StaeckelArg * params= (struct dJzdU0StaeckelArg *) malloc ( nthreads * sizeof (struct dJzdU0StaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,mid)							\
  shared(djzdU0,vmin,dJzInt,params,T,delta,E,Lz,I3V,u0,cosh2u0,sinh2u0,potupi2)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(vmin+ii) == -9999.99 ){
      *(djzdU0+ii)= 0.;
      continue;
    }
    if ( (0.5 * M_PI - *(vmin+ii)) / M_PI * 2. < 0.000001 ){//planar
      *(djzdU0+ii) = 0.;
      continue;
    }
    double td= *(delta+ii*delta_stride);
    double ch0= cosh(*(u0+ii)), sh0= sinh(*(u0+ii));
    (params+tid)->delta= td;
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / td / td;
    (params+tid)->I3V= *(I3V+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->cosh2u0= *(cosh2u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->potupi2= *(potupi2+ii);
    (params+tid)->vmin= *(vmin+ii);
    (params+tid)->dpotupi2du0= -calcRforce(td*sh0,0.,0.,0.,nargs,actionAngleArgs)*td*ch0;
    (dJzInt+tid)->function = &dJzdU0LowStaeckelIntegrand;
    (dJzInt+tid)->params = params+tid;
    mid= sqrt( 0.5 * (M_PI/2. - *(vmin+ii) ) );
    *(djzdU0+ii)= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    (dJzInt+tid)->function = &dJzdU0HighStaeckelIntegrand;
    *(djzdU0+ii)+= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    *(djzdU0+ii)*= sqrt(2.) * td / M_PI;
  }
  free(dJzInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
// ---- Route-A analytic action Hessians for c=True freq Jacobians (#131) ----
// d(dJ/dP0)/dP via a fixed-domain theta-map u=c-r*cos(theta), theta in [0,pi]
// (P-independent domain -> pure under-integral, no boundary terms). The
// turning-point sensitivity A_tp=-S_P(tp)/S_u(tp) enters via c_P,r_P; the
// integrand stays finite at the turning points (S_u*u_P+S_P vanishes there).
double JRStaeckel_dSdu(double u, struct dJRStaeckelArg * p){
  // dS_R/du (S_R=JRStaeckelIntegrandSquared); needs the force at (u,v0)
  double shu= sinh(u), chu= cosh(u), sinh2u= shu*shu, s2u= 2.*shu*chu;
  double sv0= sin(p->v0), cv0= cos(p->v0);
  double R= p->delta*shu*sv0, z= p->delta*chu*cv0;
  double FR= calcRforce(R,z,0.,0.,p->nargs,p->actionAngleArgs);
  double Fz= calczforce(R,z,0.,0.,p->nargs,p->actionAngleArgs);
  double Phi= evaluatePotentialsUV(u,p->v0,p->delta,p->nargs,p->actionAngleArgs);
  double dPhidu= -FR*p->delta*chu*sv0 - Fz*p->delta*shu*cv0;
  double ddUdu= s2u*Phi + (sinh2u+p->sin2v0)*dPhidu;
  return p->E*s2u - ddUdu + p->Lz22delta*2.*chu/(sinh2u*shu);
}
double JzStaeckel_dSdv(double v, struct dJzStaeckelArg * p){
  double sv= sin(v), cv= cos(v), sin2v= sv*sv, s2v= 2.*sv*cv;
  double sh0= sinh(p->u0), ch0= cosh(p->u0);
  double R= p->delta*sh0*sv, z= p->delta*ch0*cv;
  double FR= calcRforce(R,z,0.,0.,p->nargs,p->actionAngleArgs);
  double Fz= calczforce(R,z,0.,0.,p->nargs,p->actionAngleArgs);
  double Phi= evaluatePotentialsUV(p->u0,v,p->delta,p->nargs,p->actionAngleArgs);
  double dPhidv= -FR*p->delta*sh0*cv + Fz*p->delta*ch0*sv;
  double ddVdv= -( s2v*Phi + (p->sinh2u0+sin2v)*dPhidv );
  return p->E*s2v + ddVdv + p->Lz22delta*2.*cv/(sin2v*sv);
}
double JzStaeckel_dSdu0(double v, struct dJzStaeckelArg * p){
  // dS_z/du0 (= ddV/du0), same expression as the dJzdU0 integrand's numerator
  double sv= sin(v), cv= cos(v), sin2v= sv*sv;
  double sh0= sinh(p->u0), ch0= cosh(p->u0);
  double R= p->delta*sh0*sv, z= p->delta*ch0*cv;
  double FR= calcRforce(R,z,0.,0.,p->nargs,p->actionAngleArgs);
  double Fz= calczforce(R,z,0.,0.,p->nargs,p->actionAngleArgs);
  double FRu0= calcRforce(p->delta*sh0,0.,0.,0.,p->nargs,p->actionAngleArgs);
  double dpotupi2du0= -FRu0*p->delta*ch0;
  double Phi_u0v= evaluatePotentialsUV(p->u0,v,p->delta,p->nargs,p->actionAngleArgs);
  double dPhi_du0= -FR*p->delta*ch0*sv - Fz*p->delta*sh0*cv;
  return 2.*ch0*sh0*p->potupi2 + p->cosh2u0*dpotupi2du0
    - 2.*sh0*ch0*Phi_u0v - (p->sinh2u0+sin2v)*dPhi_du0;
}
void calcd2JRStaeckel(int ndata,
		      double * d2jr,   // ndata*9: [ii*9+P0*3+P]=d(djr_P0)/dP, {E,Lz,I3U}
		      double * umin, double * umax,
		      double * E, double * Lz, double * I3U,
		      int ndelta, double * delta, double * u0, double * sinh2u0,
		      double * v0, double * sin2v0, double * potu0v0,
		      int nargs, struct potentialArg * actionAngleArgs, int order){
  int ii, tid, nthreads;
#ifdef _OPENMP
  nthreads= omp_get_max_threads();
#else
  nthreads= 1;
#endif
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) malloc ( nthreads * sizeof (struct dJRStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (2*order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(tid,ii) \
  shared(d2jr,umin,umax,E,Lz,I3U,delta,u0,sinh2u0,v0,sin2v0,potu0v0,params,T)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    int jj, P0, Pk, gi;
    for (jj=0; jj < 9; jj++) *(d2jr+ii*9+jj)= 0.;
    if ( *(umin+ii) == -9999.99 || *(umax+ii) == -9999.99 ) continue;
    if ( (*(umax+ii) - *(umin+ii)) / *(umax+ii) < 0.000001 ) continue; //circular
    double dlt= *(delta+ii*delta_stride);
    (params+tid)->delta= dlt;
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / dlt / dlt;
    (params+tid)->I3U= *(I3U+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->v0= *(v0+ii);
    (params+tid)->sin2v0= *(sin2v0+ii);
    (params+tid)->potu0v0= *(potu0v0+ii);
    (params+tid)->umin= *(umin+ii);
    (params+tid)->umax= *(umax+ii);
    double umn= *(umin+ii), umx= *(umax+ii);
    double Ld2= *(Lz+ii) / dlt / dlt; // dS/dLz = -Ld2/sinh^2u
    double shmn= sinh(umn), shmx= sinh(umx);
    double Su_min= JRStaeckel_dSdu(umn,params+tid), Su_max= JRStaeckel_dSdu(umx,params+tid);
    double SPmin[3]= { shmn*shmn, -Ld2/(shmn*shmn), -1. };
    double SPmax[3]= { shmx*shmx, -Ld2/(shmx*shmx), -1. };
    double cP[3], rP[3];
    for (Pk=0; Pk < 3; Pk++){
      double Amn= -SPmin[Pk]/Su_min, Amx= -SPmax[Pk]/Su_max;
      cP[Pk]= 0.5*(Amn+Amx); rP[Pk]= 0.5*(Amx-Amn);
    }
    double cc= 0.5*(umn+umx), rr= 0.5*(umx-umn);
    double H[9], Ifirst[3];
    for (jj=0; jj < 9; jj++) H[jj]= 0.;
    for (Pk=0; Pk < 3; Pk++) Ifirst[Pk]= 0.;
    double xi, wi;
    for (gi=0; gi < 2*order; gi++){
      gsl_integration_glfixed_point(0.,M_PI,gi,&xi,&wi,T);
      double costh= cos(xi), sinth= sin(xi);
      double u= cc - rr*costh;
      double S= JRStaeckelIntegrandSquared4dJR(u,params+tid);
      if ( S <= 0. ) continue;
      double sq= sqrt(S), S15= S*sq;
      double Su= JRStaeckel_dSdu(u,params+tid);
      double shu= sinh(u), chu= cosh(u), sinh2u= shu*shu;
      double SP[3]= { sinh2u, -Ld2/sinh2u, -1. };
      double gv[3]= { sinh2u, 1./sinh2u, 1. };
      double gu[3]= { 2.*shu*chu, -2.*chu/(sinh2u*shu), 0. };
      for (P0=0; P0 < 3; P0++){
	Ifirst[P0]+= wi*( gv[P0]/sq*rr*sinth );
	for (Pk=0; Pk < 3; Pk++){
	  double uP= cP[Pk]-rP[Pk]*costh;
	  H[P0*3+Pk]+= wi*( rr*sinth*( gu[P0]*uP/sq - 0.5*gv[P0]*(Su*uP+SP[Pk])/S15 )
			    + rP[Pk]*gv[P0]/sq*sinth );
	}
      }
    }
    double prefr= dlt / M_PI / sqrt(2.);
    double pref[3]= { prefr, - *(Lz+ii) / M_PI / sqrt(2.) / dlt, -prefr };
    for (P0=0; P0 < 3; P0++)
      for (Pk=0; Pk < 3; Pk++)
	*(d2jr+ii*9+P0*3+Pk)= H[P0*3+Pk]*pref[P0];
    // pref(Lz) depends on Lz -> product-rule term on d(djrdLz)/dLz
    *(d2jr+ii*9+1*3+1)+= (-1. / M_PI / sqrt(2.) / dlt)*Ifirst[1];
  }
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
void calcd2JzStaeckel(int ndata,
		      double * d2jz,   // ndata*12: [ii*12+P0*4+P]=d(djz_P0)/dP, {E,Lz,I3V,u0}
		      double * vmin,
		      double * E, double * Lz, double * I3V,
		      int ndelta, double * delta, double * u0, double * cosh2u0,
		      double * sinh2u0, double * potupi2,
		      int nargs, struct potentialArg * actionAngleArgs, int order){
  int ii, tid, nthreads;
#ifdef _OPENMP
  nthreads= omp_get_max_threads();
#else
  nthreads= 1;
#endif
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) malloc ( nthreads * sizeof (struct dJzStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
  }
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (2*order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(tid,ii) \
  shared(d2jz,vmin,E,Lz,I3V,delta,u0,cosh2u0,sinh2u0,potupi2,params,T)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    int jj, P0, Pk, gi;
    for (jj=0; jj < 12; jj++) *(d2jz+ii*12+jj)= 0.;
    if ( *(vmin+ii) == -9999.99 ) continue;
    if ( (M_PI/2. - *(vmin+ii)) < 0.0000001 ) continue; //planar (J_z=0)
    double dlt= *(delta+ii*delta_stride);
    (params+tid)->delta= dlt;
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / dlt / dlt;
    (params+tid)->I3V= *(I3V+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->cosh2u0= *(cosh2u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->potupi2= *(potupi2+ii);
    (params+tid)->vmin= *(vmin+ii);
    double vmn= *(vmin+ii), vhi= M_PI/2.;
    double Ld2= *(Lz+ii) / dlt / dlt;
    double svmn= sin(vmn);
    double Sv_min= JzStaeckel_dSdv(vmn,params+tid);
    // S_P at vmin for {E,Lz,I3V,u0}
    double SPmin[4]= { svmn*svmn, -Ld2/(svmn*svmn), 1., JzStaeckel_dSdu0(vmn,params+tid) };
    double cP[4], rP[4];
    for (Pk=0; Pk < 4; Pk++){
      double Amn= -SPmin[Pk]/Sv_min; // A at vmin; A at pi/2 = 0 (fixed endpoint)
      cP[Pk]= 0.5*Amn; rP[Pk]= -0.5*Amn;
    }
    double cc= 0.5*(vmn+vhi), rr= 0.5*(vhi-vmn);
    double H[12], Ifirst[3];
    for (jj=0; jj < 12; jj++) H[jj]= 0.;
    for (P0=0; P0 < 3; P0++) Ifirst[P0]= 0.;
    double xi, wi;
    for (gi=0; gi < 2*order; gi++){
      gsl_integration_glfixed_point(0.,M_PI,gi,&xi,&wi,T);
      double costh= cos(xi), sinth= sin(xi);
      double v= cc - rr*costh;
      double S= JzStaeckelIntegrandSquared4dJz(v,params+tid);
      if ( S <= 0. ) continue;
      double sq= sqrt(S), S15= S*sq;
      double Sv= JzStaeckel_dSdv(v,params+tid);
      double sv= sin(v), cv= cos(v), sin2v= sv*sv;
      double SP[4]= { sin2v, -Ld2/sin2v, 1., JzStaeckel_dSdu0(v,params+tid) };
      double gv[3]= { sin2v, 1./sin2v, 1. };
      double gvd[3]= { 2.*sv*cv, -2.*cv/(sin2v*sv), 0. };
      for (P0=0; P0 < 3; P0++){
	Ifirst[P0]+= wi*( gv[P0]/sq*rr*sinth );
	for (Pk=0; Pk < 4; Pk++){
	  double vP= cP[Pk]-rP[Pk]*costh;
	  H[P0*4+Pk]+= wi*( rr*sinth*( gvd[P0]*vP/sq - 0.5*gv[P0]*(Sv*vP+SP[Pk])/S15 )
			    + rP[Pk]*gv[P0]/sq*sinth );
	}
      }
    }
    double prefz= sqrt(2.) * dlt / M_PI;
    double pref[3]= { prefz, - *(Lz+ii) * sqrt(2.) / M_PI / dlt, prefz };
    for (P0=0; P0 < 3; P0++)
      for (Pk=0; Pk < 4; Pk++)
	*(d2jz+ii*12+P0*4+Pk)= H[P0*4+Pk]*pref[P0];
    // pref(Lz) depends on Lz -> product-rule term on d(djzdLz)/dLz
    *(d2jz+ii*12+1*4+1)+= (-sqrt(2.) / M_PI / dlt)*Ifirst[1];
  }
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
void calcAnglesStaeckel(int ndata,
			double * Angler,
			double * Anglephi,
			double * Anglez,
			double * Omegar,
			double * Omegaphi,
			double * Omegaz,
			double * dI3dJR,
			double * dI3dJz,
			double * dI3dLz,
			double * dJRdE,
			double * dJRdLz,
			double * dJRdI3,
			double * dJzdE,
			double * dJzdLz,
			double * dJzdI3,
			double * ux,
			double * vx,
			double * pux,
			double * pvx,
			double * umin,
			double * umax,
			double * E,
			double * Lz,
			double * I3U,
			int ndelta,
			double * delta,
			double * u0,
			double * sinh2u0,
			double * v0,
			double * sin2v0,
			double * potu0v0,
			double * vmin,
			double * I3V,
			double * cosh2u0,
			double * potupi2,
			int nargs,
			struct potentialArg * actionAngleArgs,
			int order){
  int ii, tid, nthreads;
  double Or1, Or2, I3r1, I3r2,phitmp;
  double mid, midpoint;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * AngleuInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  gsl_function * AnglevInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct dJRStaeckelArg * paramsu= (struct dJRStaeckelArg *) malloc ( nthreads * sizeof (struct dJRStaeckelArg) );
  struct dJzStaeckelArg * paramsv= (struct dJzStaeckelArg *) malloc ( nthreads * sizeof (struct dJzStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (paramsu+tid)->nargs= nargs;
    (paramsu+tid)->actionAngleArgs= actionAngleArgs;
    (paramsv+tid)->nargs= nargs;
    (paramsv+tid)->actionAngleArgs= actionAngleArgs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,mid,midpoint,Or1,Or2,I3r1,I3r2,phitmp)			\
  shared(Angler,Anglephi,Anglez,Omegar,Omegaz,dI3dJR,dI3dJz,umin,umax,AngleuInt,AnglevInt,paramsu,paramsv,T,delta,E,Lz,I3U,u0,sinh2u0,v0,sin2v0,potu0v0,vmin,I3V,cosh2u0,potupi2)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    if ( *(umin+ii) == -9999.99 || *(umax+ii) == -9999.99 ){
      *(Angler+ii)= 9999.99;
      *(Anglephi+ii)= 9999.99;
      *(Anglez+ii)= 9999.99;
      continue;
    }
    if ( (*(umax+ii) - *(umin+ii)) / *(umax+ii) < 0.000001 ){//circular
      *(Angler+ii) = 0.;
      *(Anglephi+ii) = 0.;
      *(Anglez+ii) = 0.;
      continue;
    }
    //Setup u function
    (paramsu+tid)->delta= *(delta+ii*delta_stride);
    (paramsu+tid)->E= *(E+ii);
    (paramsu+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (paramsu+tid)->I3U= *(I3U+ii);
    (paramsu+tid)->u0= *(u0+ii);
    (paramsu+tid)->sinh2u0= *(sinh2u0+ii);
    (paramsu+tid)->v0= *(v0+ii);
    (paramsu+tid)->sin2v0= *(sin2v0+ii);
    (paramsu+tid)->potu0v0= *(potu0v0+ii);
    (paramsu+tid)->umin= *(umin+ii);
    (paramsu+tid)->umax= *(umax+ii);
    (AngleuInt+tid)->params = paramsu+tid;
    midpoint= *(umin+ii)+ 0.5 * ( *(umax+ii) - *(umin+ii) );
    if ( *(pux+ii) > 0. ) {
      if ( *(ux+ii) > midpoint ) {
	mid= sqrt( ( *(umax+ii) - *(ux+ii) ) );
	(AngleuInt+tid)->function = &dJRdEHighStaeckelIntegrand;
	Or1= gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	(AngleuInt+tid)->function = &dJRdI3HighStaeckelIntegrand;
	I3r1= -gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	(AngleuInt+tid)->function = &dJRdLzHighStaeckelIntegrand;
	*(Anglephi+ii)= M_PI * *(dJRdLz+ii) + *(Lz+ii) * gsl_integration_glfixed (AngleuInt+tid,0.,mid,T) / *(delta+ii*delta_stride) / sqrt(2.);
	Or1*= *(delta+ii*delta_stride) / sqrt(2.);
	I3r1*= *(delta+ii*delta_stride) / sqrt(2.);
	Or1= M_PI * *(dJRdE+ii) - Or1;
	I3r1= M_PI * *(dJRdI3+ii) - I3r1;
      }
      else {
	mid= sqrt( ( *(ux+ii) - *(umin+ii) ) );
	(AngleuInt+tid)->function = &dJRdELowStaeckelIntegrand;
	Or1= gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	(AngleuInt+tid)->function = &dJRdI3LowStaeckelIntegrand;
	I3r1= -gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	(AngleuInt+tid)->function = &dJRdLzLowStaeckelIntegrand;
	*(Anglephi+ii)= - *(Lz+ii) * gsl_integration_glfixed (AngleuInt+tid,0.,mid,T) / *(delta+ii*delta_stride) / sqrt(2.);
	Or1*= *(delta+ii*delta_stride) / sqrt(2.);
	I3r1*= *(delta+ii*delta_stride) / sqrt(2.);
      }
    }
    else {
      if ( *(ux+ii) > midpoint ) {
	mid= sqrt( ( *(umax+ii) - *(ux+ii) ) );
	(AngleuInt+tid)->function = &dJRdEHighStaeckelIntegrand;
	Or1= gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	Or1*= *(delta+ii*delta_stride) / sqrt(2.);
	Or1= M_PI * *(dJRdE+ii) + Or1;
	(AngleuInt+tid)->function = &dJRdI3HighStaeckelIntegrand;
	I3r1= -gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	I3r1*= *(delta+ii*delta_stride) / sqrt(2.);
	I3r1= M_PI * *(dJRdI3+ii) + I3r1;
	(AngleuInt+tid)->function = &dJRdLzHighStaeckelIntegrand;
	*(Anglephi+ii)= M_PI * *(dJRdLz+ii) - *(Lz+ii) * gsl_integration_glfixed (AngleuInt+tid,0.,mid,T) / *(delta+ii*delta_stride) / sqrt(2.);
      }
      else {
	mid= sqrt( ( *(ux+ii) - *(umin+ii) ) );
	(AngleuInt+tid)->function = &dJRdELowStaeckelIntegrand;
	Or1= gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	Or1*= *(delta+ii*delta_stride) / sqrt(2.);
	Or1= 2. * M_PI * *(dJRdE+ii) - Or1;
	(AngleuInt+tid)->function = &dJRdI3LowStaeckelIntegrand;
	I3r1= -gsl_integration_glfixed (AngleuInt+tid,0.,mid,T);
	I3r1*= *(delta+ii*delta_stride) / sqrt(2.);
	I3r1= 2. * M_PI * *(dJRdI3+ii) - I3r1;
	(AngleuInt+tid)->function = &dJRdLzLowStaeckelIntegrand;
	*(Anglephi+ii)= 2. * M_PI * *(dJRdLz+ii) + *(Lz+ii) * gsl_integration_glfixed (AngleuInt+tid,0.,mid,T) / *(delta+ii*delta_stride) / sqrt(2.);
      }
    }
    //Setup v function
    (paramsv+tid)->delta= *(delta+ii*delta_stride);
    (paramsv+tid)->E= *(E+ii);
    (paramsv+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (paramsv+tid)->I3V= *(I3V+ii);
    (paramsv+tid)->u0= *(u0+ii);
    (paramsv+tid)->cosh2u0= *(cosh2u0+ii);
    (paramsv+tid)->sinh2u0= *(sinh2u0+ii);
    (paramsv+tid)->potupi2= *(potupi2+ii);
    (paramsv+tid)->vmin= *(vmin+ii);
    (AnglevInt+tid)->params = paramsv+tid;
    midpoint= *(vmin+ii)+ 0.5 * ( 0.5 * M_PI - *(vmin+ii) );
    if ( *(pvx+ii) > 0. ) {
      if ( *(vx+ii) < midpoint || *(vx+ii) > (M_PI - midpoint) ) {
	mid = ( *(vx+ii) > 0.5 * M_PI ) ? sqrt( (M_PI - *(vx+ii) - *(vmin+ii))): sqrt( *(vx+ii) - *(vmin+ii));
	(AnglevInt+tid)->function = &dJzdELowStaeckelIntegrand;
	Or2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	Or2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdI3LowStaeckelIntegrand;
	I3r2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	I3r2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdLzLowStaeckelIntegrand;
	phitmp= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	phitmp*= - *(Lz+ii) / *(delta+ii*delta_stride) / sqrt(2.);
	if ( *(vx+ii) > 0.5 * M_PI ) {
	  Or2= M_PI * *(dJzdE+ii) - Or2;
	  I3r2= M_PI * *(dJzdI3+ii) - I3r2;
	  phitmp= M_PI * *(dJzdLz+ii) - phitmp;
	}
      }
      else {
	mid= sqrt( fabs ( 0.5 * M_PI - *(vx+ii) ) );
	(AnglevInt+tid)->function = &dJzdEHighStaeckelIntegrand;
	Or2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	Or2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdI3HighStaeckelIntegrand;
	I3r2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	I3r2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdLzHighStaeckelIntegrand;
	phitmp= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	phitmp*= - *(Lz+ii) / *(delta+ii*delta_stride) / sqrt(2.);
	if ( *(vx+ii) > 0.5 * M_PI ) {
	  Or2= 0.5 * M_PI * *(dJzdE+ii) + Or2;
	  I3r2= 0.5 * M_PI * *(dJzdI3+ii) + I3r2;
	  phitmp= 0.5 * M_PI * *(dJzdLz+ii) + phitmp;
	}
	else {
	  Or2= 0.5 * M_PI * *(dJzdE+ii) - Or2;
	  I3r2= 0.5 * M_PI * *(dJzdI3+ii) - I3r2;
	  phitmp= 0.5 * M_PI * *(dJzdLz+ii) - phitmp;
	}
      }
    }
    else {
      if ( *(vx+ii) < midpoint || *(vx+ii) > (M_PI - midpoint)) {
	mid = ( *(vx+ii) > 0.5 * M_PI ) ? sqrt( (M_PI - *(vx+ii) - *(vmin+ii))): sqrt( *(vx+ii) - *(vmin+ii));
	(AnglevInt+tid)->function = &dJzdELowStaeckelIntegrand;
	Or2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	Or2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdI3LowStaeckelIntegrand;
	I3r2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	I3r2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdLzLowStaeckelIntegrand;
	phitmp= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	phitmp*= - *(Lz+ii) / *(delta+ii*delta_stride) / sqrt(2.);
	if ( *(vx+ii) < 0.5 * M_PI ) {
	  Or2= 2. * M_PI * *(dJzdE+ii) - Or2;
	  I3r2= 2. * M_PI * *(dJzdI3+ii) - I3r2;
	  phitmp= 2. * M_PI * *(dJzdLz+ii) - phitmp;
	}
	else {
	  Or2= M_PI * *(dJzdE+ii) + Or2;
	  I3r2= M_PI * *(dJzdI3+ii) + I3r2;
	  phitmp= M_PI * *(dJzdLz+ii) + phitmp;
	}
      }
      else {
	mid= sqrt( fabs ( 0.5 * M_PI - *(vx+ii) ) );
	(AnglevInt+tid)->function = &dJzdEHighStaeckelIntegrand;
	Or2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	Or2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdI3HighStaeckelIntegrand;
	I3r2= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	I3r2*= *(delta+ii*delta_stride) / sqrt(2.);
	(AnglevInt+tid)->function = &dJzdLzHighStaeckelIntegrand;
	phitmp= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	phitmp*= - *(Lz+ii) / *(delta+ii*delta_stride) / sqrt(2.);
	if ( *(vx+ii) < 0.5 * M_PI ) {
	  Or2= 1.5 * M_PI * *(dJzdE+ii) + Or2;
	  I3r2= 1.5 * M_PI * *(dJzdI3+ii) + I3r2;
	  phitmp= 1.5 * M_PI * *(dJzdLz+ii) + phitmp;
	}
	else {
	  Or2= 1.5 * M_PI * *(dJzdE+ii) - Or2;
	  I3r2= 1.5 * M_PI * *(dJzdI3+ii) - I3r2;
	  phitmp= 1.5 * M_PI * *(dJzdLz+ii) - phitmp;
	}
      }
    }
    *(Angler+ii)= *(Omegar+ii) * ( Or1 + Or2 )
      + *(dI3dJR+ii) * ( I3r1 + I3r2 );
    // In Binney (2012) Anglez starts at zmax/vmin and v_z < 0 / v_v > 0;
    // Put this on the same system as Isochrone and Spherical angles +pi/2
    *(Anglez+ii)= *(Omegaz+ii) * ( Or1 + Or2 )
      + *(dI3dJz+ii) * ( I3r1 + I3r2 ) + 0.5 * M_PI;
    *(Anglephi+ii)+= phitmp;
    *(Anglephi+ii)+= *(Omegaphi+ii) * ( Or1 + Or2 )
      + *(dI3dLz+ii) * ( I3r1 + I3r2 );
    *(Angler+ii)= fmod(*(Angler+ii),2. * M_PI);
    *(Anglez+ii)= fmod(*(Anglez+ii),2. * M_PI);
    while ( *(Angler+ii) < 0. )
      *(Angler+ii)+= 2. * M_PI;
    while ( *(Anglez+ii) < 0. )
      *(Anglez+ii)+= 2. * M_PI;
    while ( *(Angler+ii) > 2. * M_PI )
      *(Angler+ii)-= 2. * M_PI;
    while ( *(Anglez+ii) > 2. * M_PI )
      *(Anglez+ii)-= 2. * M_PI;
  }
  free(AngleuInt);
  free(AnglevInt);
  free(paramsu);
  free(paramsv);
  gsl_integration_glfixed_table_free ( T );
}
void calcUminUmax(int ndata,
		  double * umin,
		  double * umax,
		  double * ux,
		  double * pux,
		  double * E,
		  double * Lz,
		  double * I3U,
		  int ndelta,
		  double * delta,
		  double * u0,
		  double * sinh2u0,
		  double * v0,
		  double * sin2v0,
		  double * potu0v0,
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
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) malloc ( nthreads * sizeof (struct JRStaeckelArg) );
  //Setup solver
  int status;
  int iter, max_iter = 100;
  const gsl_root_fsolver_type *T;
  struct pragmasolver *s= (struct pragmasolver *) malloc ( nthreads * sizeof (struct pragmasolver) );;
  double u_lo, u_hi;
  T = gsl_root_fsolver_brent;
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
    (s+tid)->s= gsl_root_fsolver_alloc (T);
  }
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
  gsl_set_error_handler_off();
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,iter,status,u_lo,u_hi,meps,peps)				\
  shared(umin,umax,JRRoot,params,s,ux,delta,E,Lz,I3U,u0,sinh2u0,v0,sin2v0,potu0v0,max_iter)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    //Setup function
    (params+tid)->delta= *(delta+ii*delta_stride);
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (params+tid)->I3U= *(I3U+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->v0= *(v0+ii);
    (params+tid)->sin2v0= *(sin2v0+ii);
    (params+tid)->potu0v0= *(potu0v0+ii);
    (JRRoot+tid)->function = &JRStaeckelIntegrandSquared;
    (JRRoot+tid)->params = params+tid;
    //Find starting points for minimum
    peps= GSL_FN_EVAL(JRRoot+tid,*(ux+ii)+0.000001);
    meps= GSL_FN_EVAL(JRRoot+tid,*(ux+ii)-0.000001);
    if ( fabs(GSL_FN_EVAL(JRRoot+tid,*(ux+ii))) < 0.0000001 && peps*meps < 0. ){ //we are at umin or umax
      if ( peps < 0. && meps > 0. ) {//umax
	*(umax+ii)= *(ux+ii);
	u_lo= 0.9 * (*(ux+ii) - 0.000001);
	u_hi= *(ux+ii) - 0.0000001;
	while ( GSL_FN_EVAL(JRRoot+tid,u_lo) >= 0. && u_lo > 0.000000001){
	  u_hi= u_lo; //this makes sure that brent evaluates using previous
	  u_lo*= 0.9;
	}
	//Find root
	status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, u_lo, u_hi);
	if (status == GSL_EINVAL) {
	  *(umin+ii) = 0.;//Assume zero if below 0.000000001
	} else {
	  iter= 0;
	  do
	    {
	      iter++;
	      status = gsl_root_fsolver_iterate ((s+tid)->s);
	      u_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	      u_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	      status = gsl_root_test_interval (u_lo, u_hi,
					       9.9999999999999998e-13,
					       4.4408920985006262e-16);
	    }
	  while (status == GSL_CONTINUE && iter < max_iter);
// LCOV_EXCL_START
	  if (status == GSL_EINVAL) {//Shouldn't ever get here
	    *(umin+ii) = -9999.99;
	    *(umax+ii) = -9999.99;
	    continue;
	  }
// LCOV_EXCL_STOP
	  *(umin+ii) = gsl_root_fsolver_root ((s+tid)->s);
	}
      }
      else {// JB: Should catch all: if ( peps > 0. && meps < 0. ){//umin
	*(umin+ii)= *(ux+ii);
	u_lo= *(ux+ii) + 0.000001;
	u_hi= 1.1 * (*(ux+ii) + 0.000001);
	while ( GSL_FN_EVAL(JRRoot+tid,u_hi) >= 0. && u_hi < asinh(37.5/ *(delta+ii*delta_stride))) {
	  u_lo= u_hi; //this makes sure that brent evaluates using previous
	  u_hi*= 1.1;
	}
	//Find root
	status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, u_lo, u_hi);
	if (status == GSL_EINVAL) {
	  *(umin+ii) = -9999.99;
	  *(umax+ii) = -9999.99;
	  continue;
	}
	iter= 0;
	do
	  {
	    iter++;
	    status = gsl_root_fsolver_iterate ((s+tid)->s);
	    u_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	    u_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	    status = gsl_root_test_interval (u_lo, u_hi,
					     9.9999999999999998e-13,
					     4.4408920985006262e-16);
	  }
	while (status == GSL_CONTINUE && iter < max_iter);
// LCOV_EXCL_START
	if (status == GSL_EINVAL) {//Shouldn't ever get here
	  *(umin+ii) = -9999.99;
	  *(umax+ii) = -9999.99;
	  continue;
	}
// LCOV_EXCL_STOP
	*(umax+ii) = gsl_root_fsolver_root ((s+tid)->s);
      }
    }
    else if ( fabs(peps) < 0.00000001 && fabs(meps) < 0.00000001 && peps <= 0 && meps <= 0 ) {//circular
	*(umin+ii) = *(ux+ii);
	*(umax+ii) = *(ux+ii);
    }
    else {
      u_lo= 0.9 * *(ux+ii);
      u_hi= *(ux+ii);
      while ( GSL_FN_EVAL(JRRoot+tid,u_lo) >= 0. && u_lo > 0.000000001){
	u_hi= u_lo; //this makes sure that brent evaluates using previous
	u_lo*= 0.9;
      }
      u_hi= (u_lo < 0.9 * *(ux+ii)) ? u_lo / 0.9 / 0.9: *(ux+ii);
      //Find root
      status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, u_lo, u_hi);
      if (status == GSL_EINVAL) {
	*(umin+ii) = 0.;//Assume zero if below 0.000000001
      } else {
	iter= 0;
	do
	  {
	    iter++;
	    status = gsl_root_fsolver_iterate ((s+tid)->s);
	    u_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	    u_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	    status = gsl_root_test_interval (u_lo, u_hi,
					     9.9999999999999998e-13,
					     4.4408920985006262e-16);
	  }
	while (status == GSL_CONTINUE && iter < max_iter);
// LCOV_EXCL_START
	if (status == GSL_EINVAL) {//Shouldn't ever get here
	  *(umin+ii) = -9999.99;
	  *(umax+ii) = -9999.99;
	  continue;
	}
// LCOV_EXCL_STOP
	*(umin+ii) = gsl_root_fsolver_root ((s+tid)->s);
      }
      //Find starting points for maximum
      u_lo= *(ux+ii);
      u_hi= 1.1 * *(ux+ii);
      while ( GSL_FN_EVAL(JRRoot+tid,u_hi) > 0. && u_hi < asinh(37.5/ *(delta+ii*delta_stride))) {
	u_lo= u_hi; //this makes sure that brent evaluates using previous
	u_hi*= 1.1;
      }
      u_lo= (u_hi > 1.1 * *(ux+ii)) ? u_hi / 1.1 / 1.1: *(ux+ii);
      //Find root
      status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid, u_lo, u_hi);
      if (status == GSL_EINVAL) {
	*(umin+ii) = -9999.99;
	*(umax+ii) = -9999.99;
	continue;
      }
      iter= 0;
      do
	{
	  iter++;
	  status = gsl_root_fsolver_iterate ((s+tid)->s);
	  u_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	  u_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	  status = gsl_root_test_interval (u_lo, u_hi,
					   9.9999999999999998e-13,
					   4.4408920985006262e-16);
	}
      while (status == GSL_CONTINUE && iter < max_iter);
// LCOV_EXCL_START
      if (status == GSL_EINVAL) {//Shouldn't ever get here
	*(umin+ii) = -9999.99;
	*(umax+ii) = -9999.99;
	continue;
      }
// LCOV_EXCL_STOP
      *(umax+ii) = gsl_root_fsolver_root ((s+tid)->s);
    }
  }
  gsl_set_error_handler (NULL);
  for (tid=0; tid < nthreads; tid++)
    gsl_root_fsolver_free( (s+tid)->s);
  free(s);
  free(JRRoot);
  free(params);
}
void calcVmin(int ndata,
	      double * vmin,
	      double * vx,
	      double * pvx,
	      double * E,
	      double * Lz,
	      double * I3V,
	      int ndelta,
	      double * delta,
	      double * u0,
	      double * cosh2u0,
	      double * sinh2u0,
	      double * potupi2,
	      int nargs,
	      struct potentialArg * actionAngleArgs){
  int ii, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * JzRoot= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct JzStaeckelArg * params= (struct JzStaeckelArg *) malloc ( nthreads * sizeof (struct JzStaeckelArg) );
  //Setup solver
  int status;
  int iter, max_iter = 100;
  const gsl_root_fsolver_type *T;
  struct pragmasolver *s= (struct pragmasolver *) malloc ( nthreads * sizeof (struct pragmasolver) );;
  double v_lo, v_hi;
  T = gsl_root_fsolver_brent;
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs;
    (s+tid)->s= gsl_root_fsolver_alloc (T);
  }
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
  gsl_set_error_handler_off();
#pragma omp parallel for schedule(static,chunk)				\
  private(tid,ii,iter,status,v_lo,v_hi)				\
  shared(vmin,JzRoot,params,s,vx,delta,E,Lz,I3V,u0,cosh2u0,sinh2u0,potupi2,max_iter)
  for (ii=0; ii < ndata; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    //Setup function
    (params+tid)->delta= *(delta+ii*delta_stride);
    (params+tid)->E= *(E+ii);
    (params+tid)->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / *(delta+ii*delta_stride) / *(delta+ii*delta_stride);
    (params+tid)->I3V= *(I3V+ii);
    (params+tid)->u0= *(u0+ii);
    (params+tid)->cosh2u0= *(cosh2u0+ii);
    (params+tid)->sinh2u0= *(sinh2u0+ii);
    (params+tid)->potupi2= *(potupi2+ii);
    (JzRoot+tid)->function = &JzStaeckelIntegrandSquared;
    (JzRoot+tid)->params = params+tid;
    //Find starting points for minimum
    if ( fabs(GSL_FN_EVAL(JzRoot+tid,*(vx+ii))) < 0.0000001) //we are at vmin
      *(vmin+ii)= ( *(vx+ii) > 0.5 * M_PI ) ? M_PI - *(vx+ii): *(vx+ii);
    else {
      if ( *(vx+ii) > 0.5 * M_PI ){
	v_lo= 0.9 * ( M_PI - *(vx+ii) );
	v_hi= M_PI - *(vx+ii);
      }
      else {
	v_lo= 0.9 * *(vx+ii);
	v_hi= *(vx+ii);
      }
      while ( GSL_FN_EVAL(JzRoot+tid,v_lo) >= 0. && v_lo > 0.000000001){
	v_hi= v_lo; //this makes sure that brent evaluates using previous
	v_lo*= 0.9;
      }
      //Find root
      status = gsl_root_fsolver_set ((s+tid)->s, JzRoot+tid, v_lo, v_hi);
      if (status == GSL_EINVAL) {
	*(vmin+ii) = -9999.99;
	continue;
      }
      iter= 0;
      do
	{
	  iter++;
	  status = gsl_root_fsolver_iterate ((s+tid)->s);
	  v_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	  v_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	  status = gsl_root_test_interval (v_lo, v_hi,
					   9.9999999999999998e-13,
					   4.4408920985006262e-16);
	}
      while (status == GSL_CONTINUE && iter < max_iter);
// LCOV_EXCL_START
      if (status == GSL_EINVAL) {//Shouldn't ever get here
	*(vmin+ii) = -9999.99;
	continue;
      }
// LCOV_EXCL_STOP
      *(vmin+ii) = gsl_root_fsolver_root ((s+tid)->s);
      fflush(stdout);
    }
  }
  gsl_set_error_handler (NULL);
  for (tid=0; tid < nthreads; tid++)
    gsl_root_fsolver_free( (s+tid)->s);
  free(s);
  free(JzRoot);
  free(params);
}

double JRStaeckelIntegrand(double u,
			   void * p){
  double out= JRStaeckelIntegrandSquared(u,p);
  if ( out <= 0.) return 0.;
  else return sqrt(out);
}
double JRStaeckelIntegrandSquared(double u,
				  void * p){
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) p;
  double sinh2u= sinh(u) * sinh(u);
  double dU= (sinh2u+params->sin2v0)
    *evaluatePotentialsUV(u,params->v0,params->delta,
			  params->nargs,params->actionAngleArgs)
    - (params->sinh2u0+params->sin2v0)*params->potu0v0;
  return params->E * sinh2u - params->I3U - dU  - params->Lz22delta / sinh2u;
}
double JRStaeckelIntegrandSquared4dJR(double u,
				      void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double sinh2u= sinh(u) * sinh(u);
  double dU= (sinh2u+params->sin2v0)
    *evaluatePotentialsUV(u,params->v0,params->delta,
			  params->nargs,params->actionAngleArgs)
    - (params->sinh2u0+params->sin2v0)*params->potu0v0;
  return params->E * sinh2u - params->I3U - dU  - params->Lz22delta / sinh2u;
}

double JzStaeckelIntegrand(double v,
			   void * p){
  double out= JzStaeckelIntegrandSquared(v,p);
  if ( out <= 0. ) return 0.;
  else return sqrt(out);
}
double JzStaeckelIntegrandSquared(double v,
				  void * p){
  struct JzStaeckelArg * params= (struct JzStaeckelArg *) p;
  double sin2v= sin(v) * sin(v);
  double dV= params->cosh2u0 * params->potupi2
    - (params->sinh2u0+sin2v)
    *evaluatePotentialsUV(params->u0,v,params->delta,
			  params->nargs,params->actionAngleArgs);
  return params->E * sin2v + params->I3V + dV  - params->Lz22delta / sin2v;
}
double JzStaeckelIntegrandSquared4dJz(double v,
				      void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double sin2v= sin(v) * sin(v);
  double dV= params->cosh2u0 * params->potupi2
    - (params->sinh2u0+sin2v)
    *evaluatePotentialsUV(params->u0,v,params->delta,
			  params->nargs,params->actionAngleArgs);
  return params->E * sin2v + params->I3V + dV  - params->Lz22delta / sin2v;
}
double dJRdELowStaeckelIntegrand(double t,
				 void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u= params->umin + t * t;
  return 2. * t * dJRdEStaeckelIntegrand(u,p);
}
double dJRdEHighStaeckelIntegrand(double t,
				 void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u= params->umax - t * t;
  return 2. * t * dJRdEStaeckelIntegrand(u,p);
}
double dJRdEStaeckelIntegrand(double u,
			      void * p){
  double out= JRStaeckelIntegrandSquared4dJR(u,p);
  if ( out <= 0. ) return 0.;
  else return sinh(u)*sinh(u)/sqrt(out);
}
double dJRdLzLowStaeckelIntegrand(double t,
				  void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u= params->umin + t * t;
  return 2. * t * dJRdLzStaeckelIntegrand(u,p);
}
double dJRdLzHighStaeckelIntegrand(double t,
				   void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u= params->umax - t * t;
  return 2. * t * dJRdLzStaeckelIntegrand(u,p);
}
double dJRdLzStaeckelIntegrand(double u,
			      void * p){
  double out= JRStaeckelIntegrandSquared4dJR(u,p);
  if ( out <= 0. ) return 0.;
  else return 1./sinh(u)/sinh(u)/sqrt(out);
}
double dJRdI3LowStaeckelIntegrand(double t,
				  void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u= params->umin + t * t;
  return 2. * t * dJRdI3StaeckelIntegrand(u,p);
}
double dJRdI3HighStaeckelIntegrand(double t,
				   void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u= params->umax - t * t;
  return 2. * t * dJRdI3StaeckelIntegrand(u,p);
}
double dJRdI3StaeckelIntegrand(double u,
			      void * p){
  double out= JRStaeckelIntegrandSquared4dJR(u,p);
  if ( out <= 0. ) return 0.;
  else return 1./sqrt(out);
}

double dJzdELowStaeckelIntegrand(double t,
				 void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double v= params->vmin + t * t;
  return 2. * t * dJzdEStaeckelIntegrand(v,p);
}
double dJzdEHighStaeckelIntegrand(double t,
				 void * p){
  double v= M_PI/2. - t * t;
  return 2. * t * dJzdEStaeckelIntegrand(v,p);
}
double dJzdEStaeckelIntegrand(double v,
			      void * p){
  double out= JzStaeckelIntegrandSquared4dJz(v,p);
  if ( out <= 0. ) return 0.;
  else return sin(v)*sin(v)/sqrt(out);
}
double dJzdLzLowStaeckelIntegrand(double t,
				  void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double v= params->vmin + t * t;
  return 2. * t * dJzdLzStaeckelIntegrand(v,p);
}
double dJzdLzHighStaeckelIntegrand(double t,
				   void * p){
  double v= M_PI/2. - t * t;
  return 2. * t * dJzdLzStaeckelIntegrand(v,p);
}
double dJzdLzStaeckelIntegrand(double v,
			      void * p){
  double out= JzStaeckelIntegrandSquared4dJz(v,p);
  if ( out <= 0. ) return 0.;
  else return 1./sin(v)/sin(v)/sqrt(out);
}
double dJzdI3LowStaeckelIntegrand(double t,
				  void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double v= params->vmin + t * t;
  return 2. * t * dJzdI3StaeckelIntegrand(v,p);
}
double dJzdI3HighStaeckelIntegrand(double t,
				   void * p){
  double v= M_PI/2. - t * t;
  return 2. * t * dJzdI3StaeckelIntegrand(v,p);
}
double dJzdI3StaeckelIntegrand(double v,
			       void * p){
  double out= JzStaeckelIntegrandSquared4dJz(v,p);
  if ( out <= 0. ) return 0.;
  else return 1./sqrt(out);
}
double dJzdU0LowStaeckelIntegrand(double t,
				  void * p){
  struct dJzdU0StaeckelArg * params= (struct dJzdU0StaeckelArg *) p;
  double v= params->vmin + t * t;
  return 2. * t * dJzdU0StaeckelIntegrand(v,p);
}
double dJzdU0HighStaeckelIntegrand(double t,
				   void * p){
  double v= M_PI/2. - t * t;
  return 2. * t * dJzdU0StaeckelIntegrand(v,p);
}
double dJzdU0StaeckelIntegrand(double v,
			       void * p){
  struct dJzdU0StaeckelArg * params= (struct dJzdU0StaeckelArg *) p;
  double sin2v= sin(v) * sin(v);
  double phi_u0_v= evaluatePotentialsUV(params->u0,v,params->delta,
					params->nargs,params->actionAngleArgs);
  double dV= params->cosh2u0 * params->potupi2
    - (params->sinh2u0+sin2v) * phi_u0_v;
  double out= params->E * sin2v + params->I3V + dV
    - params->Lz22delta / sin2v;
  if ( out <= 0. ) return 0.;
  double ch0= cosh(params->u0), sh0= sinh(params->u0);
  double sv= sin(v), cv= cos(v);
  double Rv= params->delta * sh0 * sv;
  double zv= params->delta * ch0 * cv;
  double FRv= calcRforce(Rv,zv,0.,0.,params->nargs,params->actionAngleArgs);
  double Fzv= calczforce(Rv,zv,0.,0.,params->nargs,params->actionAngleArgs);
  double dPhi_du0= -FRv * params->delta * ch0 * sv
    - Fzv * params->delta * sh0 * cv; // dPhi(u0,v)/du0
  double ddVdu0= 2. * ch0 * sh0 * params->potupi2
    + params->cosh2u0 * params->dpotupi2du0
    - 2. * sh0 * ch0 * phi_u0_v
    - (params->sinh2u0+sin2v) * dPhi_du0;
  return ddVdu0/sqrt(out);
}
double u0Equation(double u, void * p){
  struct u0EqArg * params= (struct u0EqArg *) p;
  double sinh2u= sinh(u) * sinh(u);
  double cosh2u= cosh(u) * cosh(u);
  double dU= cosh2u * evaluatePotentialsUV(u,0.5*M_PI,params->delta,
				    params->nargs,params->actionAngleArgs);
  return -(params->E*sinh2u-dU-params->Lz22delta/sinh2u);
}
double evaluatePotentialsUV(double u, double v, double delta,
			    int nargs,
			    struct potentialArg * actionAngleArgs){
  double R,z;
  R= delta * sinh(u) * sin(v);
  z= delta * cosh(u) * cos(v);
  return evaluatePotentials(R,z,nargs,actionAngleArgs);
}
