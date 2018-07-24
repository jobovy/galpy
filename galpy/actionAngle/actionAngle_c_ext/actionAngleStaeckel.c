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
#ifdef _WIN32
// On Windows, *need* to define this function to allow the package to be imported
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_galpy_actionAngle_c(void) { // Python 3
  return NULL;
}
#else
PyMODINIT_FUNC initgalpy_actionAngle_c(void) {} // Python 2
#endif
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
/*
  Function Declarations
*/
EXPORT void calcu0(int,double *,double *,int,int *,double *,int,double*,
		   double *,int *);
EXPORT void actionAngleStaeckel_uminUmaxVmin(int,double *,double *,double *,double *,
				      double *,double *,int,int *,double *,
				      int,double *,double *,
				      double *,double *,int *);
EXPORT void actionAngleStaeckel_actions(int,double *,double *,double *,double *,
				 double *,double *,int,int *,double *,int,
				 double *,int,double *,double *,int *);
EXPORT void actionAngleStaeckel_actionsFreqsAngles(int,double *,double *,double *,
					    double *,double *,double *,
					    int,int *,double *,
					    int,double *,int,double *,double *,
					    double *,double *,double *,
					    double *,double *,double *,int *);
EXPORT void actionAngleStaeckel_actionsFreqs(int,double *,double *,double *,double *,
				      double *,double *,int,int *,double *,
				      int,double *,int,double *,double *,
				      double *,double *,double *,int *);
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
void calcUminUmax(int,double *,double *,double *,double *,double *,double *,
		  double *,int,double *,double *,double *,double *,double *,
		  double *,int,struct potentialArg *);
void calcVmin(int,double *,double *,double *,double *,double *,double *,int,
	      double *,double *,double *,double *,double *,int,
	      struct potentialArg *);
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
	    int ndelta,
	    double * delta,
	    double *u0,
	    int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args);
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
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args);
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
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args);
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
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args);
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
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args);
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
    if ( fabs(GSL_FN_EVAL(JRRoot+tid,*(ux+ii))) < 0.0000001){ //we are at umin or umax
      peps= GSL_FN_EVAL(JRRoot+tid,*(ux+ii)+0.000001);
      meps= GSL_FN_EVAL(JRRoot+tid,*(ux+ii)-0.000001);
      if ( fabs(peps) < 0.00000001 && fabs(meps) < 0.00000001 ) {//circular
	*(umin+ii) = *(ux+ii);
	*(umax+ii) = *(ux+ii);
      }
      else if ( peps < 0. && meps > 0. ) {//umax
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
      else if ( peps > 0. && meps < 0. ){//umin
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
  uv_to_Rz(u,v,&R,&z,delta);
  return evaluatePotentials(R,z,nargs,actionAngleArgs);
}
