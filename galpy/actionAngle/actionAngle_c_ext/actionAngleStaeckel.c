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
//Per-thread potential structs: potentials may keep per-instance caches in
//their args (cf. OblateStaeckelWrapperPotential), so every OpenMP loop over
//points must use its own thread's parsed copy rather than sharing one
#ifdef _OPENMP
#define AA_TARGS(args,n) ((args) + omp_get_thread_num() * (n))
#else
#define AA_TARGS(args,n) (args)
#endif
#include <integrateFullOrbit.h>
#include <actionAngle.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// see the chi-anomaly helpers below for what these control
#define STAECKEL_CHI_EDGE 1.e-6
// An orbit approaching the axis (umin small) picks up the potential's inner
// power-law cusp: S ~ A - B u^0.2 at umin -> 0, i.e. chi^0.4 in the anomaly.
// A single fixed-order rule converges only algebraically against that (the
// default rule floors at ~4.5e-5 there), so ONLY for such orbits the radial
// action is integrated with a geometrically graded composite rule that packs
// panels toward the cusp. Every other orbit keeps the cheap single-panel rule.
#define STAECKEL_NEARAXIS 0.2
#define STAECKEL_CHI_NPANELS 24
// chi of the midpoint of [vmin, pi/2]: there y = 1/4, so chi = 2 asin(1/2).
// A constant, so it needs no per-iteration assignment inside the OpenMP
// loops -- where it would have been a shared write, being absent from the
// private() clauses.
#define STAECKEL_CHI_MID ( M_PI / 3. )
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
  double umin;
  double umax;
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
  double vmin;
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
  int wd; // weight the dJR/dI3 integrands by sqrt(2) dp_u/ddelta sqrt(S)
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
  int wu0; // weight the dJz/dI3 integrands by dV/du0 (1) or by
           // sqrt(2) dp_v/ddelta sqrt(S) (2)
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
EXPORT void calcu0(int,double *,double *,int,int *,double *,tfuncs_type_arr,
       int,double*,double *,int *);
EXPORT void actionAngleStaeckel_uminUmaxVmin(int,double *,double *,double *,double *,
				      double *,double *,int,int *,double *,tfuncs_type_arr,
				      int,double *,int,double *,
				      double *,double *,int *);
EXPORT void actionAngleStaeckel_actions(int,double *,double *,double *,double *,
				 double *,double *,int,int *,double *,tfuncs_type_arr,int,
				 double *,int,int,double *,double *,int *);
EXPORT void actionAngleStaeckel_actionsFreqsAngles(int,double *,double *,double *,
					    double *,double *,double *,
					    int,int *,double *,tfuncs_type_arr,
					    int,double *,int,int,double *,double *,
					    double *,double *,double *,
					    double *,double *,double *,double *,int *);
EXPORT void actionAngleStaeckel_actionsFreqs(int,double *,double *,double *,double *,
				      double *,double *,int,int *,double *,tfuncs_type_arr,
				      int,double *,int,int,double *,double *,
				      double *,double *,double *,int *);
void calcAnglesStaeckel(int,double *,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			double *,int,double *,double *,double *,double *,
			double *,double *,double *,double *,double *,double *,
			int,struct potentialArg *,int,int,double *);
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
double JRLowStaeckelIntegrand(double,void *);
double JRLowStaeckelIntegrandDirect(double,void *);
static inline double glfixed_graded(gsl_function *,double,double,
				    gsl_integration_glfixed_table *);
double JRHighStaeckelIntegrand(double,void *);
double JzStaeckelIntegrandSquared(double,void *);
double JzStaeckelIntegrand(double,void *);
double JzLowStaeckelIntegrand(double,void *);
double JzHighStaeckelIntegrand(double,void *);
double dJRdELowStaeckelIntegrand(double,void *);
double dJRdEHighStaeckelIntegrand(double,void *);
double dJRdLzLowStaeckelIntegrand(double,void *);
double dJRdLzHighStaeckelIntegrand(double,void *);
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
static inline double SuTermsStaeckel(double,double,double,double,double,
				     double,double,double,double,int,
				     struct potentialArg *);
static inline double dSuduStaeckel(double,double,double,double,double,double,
				   int,struct potentialArg *);
static inline double wdUStaeckel(double,double,struct dJRStaeckelArg *);
static inline double wdVStaeckel(double,double,struct dJzStaeckelArg *);
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
  double tR, tz, tdelta, A, S, sinh2u, sin2v, d1, d2;
  int delta_stride= ndelta == 1 ? 0 : 1;
  // sinh^2 u and sin^2 v without cancellation, as Rz_to_uv in
  // OblateStaeckelWrapperPotential.c and galpy.util.coords
  for (ii=0; ii < ndata; ii++) {
    tdelta= *(delta+ii*delta_stride);
    tR= *(R+ii);
    tz= *(z+ii);
    A= tR * tR + ( tz - tdelta ) * ( tz + tdelta );
    S= sqrt( A * A + 4. * tdelta * tdelta * tR * tR );
    if ( A < 0. ) {
      sinh2u= 2. * tR * tR / ( S - A );
      sin2v= ( S - A ) / ( 2. * tdelta * tdelta );
    } else {
      sinh2u= ( A + S ) / ( 2. * tdelta * tdelta );
      sin2v= ( S + A > 0. ) ? 2. * tR * tR / ( S + A ) : 0.;
    }
    d1= sqrt( ( tz + tdelta ) * ( tz + tdelta ) + tR * tR );
    d2= sqrt( ( tz - tdelta ) * ( tz - tdelta ) + tR * tR );
    *u++= asinh( sqrt( sinh2u ) );
    *v++= atan2( sqrt( sin2v ) , 2. * tz / ( d1 + d2 ) );
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
// The energy in the Staeckel model that the approximation with (delta, u0)
// stands for, Phi = [U(u) - V(v)] / (sinh^2 u + sin^2 v) with U(u) =
// cosh^2 u Phi(u,pi/2) and V(v) = cosh^2 u0 Phi(u0,pi/2) - (sinh^2 u0 +
// sin^2 v) Phi(u0,v), in place of that in the potential itself: with it,
// the actions, frequencies, and angles are those of this Staeckel model
static inline void calcEModel(int ndata,
			      double *vR,
			      double *vT,
			      double *vz,
			      double *ux,
			      double *vx,
			      double *u0,
			      int ndelta,
			      double * delta,
			      double *E,
			      int nargs,
			      struct potentialArg * actionAngleArgs){
  int ii;
  double tdelta, sinh2u, sin2v, sinh2u0;
  int delta_stride= ndelta == 1 ? 0 : 1;
#ifdef _OPENMP
  int nthreads= ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#endif
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk) private(ii,tdelta,sinh2u,sin2v,sinh2u0)
  for (ii=0; ii < ndata; ii++){
    struct potentialArg * targs= AA_TARGS(actionAngleArgs,nargs);
    tdelta= *(delta+ii*delta_stride);
    sinh2u= sinh(*(ux+ii)) * sinh(*(ux+ii));
    sin2v= sin(*(vx+ii)) * sin(*(vx+ii));
    sinh2u0= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(E+ii)= ( ( sinh2u + 1. )
	       * evaluatePotentialsUV(*(ux+ii),0.5 * M_PI,tdelta,nargs,targs)
	       - ( sinh2u0 + 1. )
	       * evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,nargs,targs)
	       + ( sinh2u0 + sin2v )
	       * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,nargs,targs) )
      / ( sinh2u + sin2v )
      + 0.5 * *(vR+ii) * *(vR+ii)
      + 0.5 * *(vT+ii) * *(vT+ii)
      + 0.5 * *(vz+ii) * *(vz+ii);
  }
}
// The reference curve of the Staeckel model of the torus through each
// point, for a curve that follows the tori (actionAngleStaeckelInverse):
// u0='torus', delta sinh(u0) = 2 / (1/R_- + 1/R_+) at the turning points
// of the torus's u libration. The
// energy and the third integral of a point are those in its model, which
// depend on u0: one application of the rule at the current u0. With warm,
// the turning points are those handed in moved to the current model by
// Newton's method (they move little between applications), and solved for
// anew only where that does not converge
static void modelRuleu0(int ndata,
			double *R,
			double *vR,
			double *vT,
			double *z,
			double *vz,
			double *u0,
			double *u0new,
			double *umin,
			double *umax,
			int warm,
			int ndelta,
			double * delta,
			int nargs,
			struct potentialArg * actionAngleArgs){
  int ii, jj, kk, nfull= 0;
  double tdelta, shux, pux, sm, sp, x, dx, Lz22delta;
  int delta_stride= ndelta == 1 ? 0 : 1;
  double *work= (double *) malloc ( 9 * ndata * sizeof(double) );
  double *E= work, *Lz= work+ndata, *ux= work+2*ndata, *vx= work+3*ndata;
  double *I3U= work+4*ndata, *sinh2u0= work+5*ndata, *v0= work+6*ndata;
  double *sin2v0= work+7*ndata, *potu0v0= work+8*ndata;
  int *full= (int *) malloc ( ndata * sizeof(int) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  calcEModel(ndata,vR,vT,vz,ux,vx,u0,ndelta,delta,E,nargs,actionAngleArgs);
#ifdef _OPENMP
  int nthreads= ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#endif
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk) private(ii,jj,kk,tdelta,shux,pux,x,dx,Lz22delta)
  for (ii=0; ii < ndata; ii++){
    struct potentialArg * targs= AA_TARGS(actionAngleArgs,nargs);
    tdelta= *(delta+ii*delta_stride);
    *(Lz+ii)= *(R+ii) * *(vT+ii);
    Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta;
    *(sinh2u0+ii)= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(v0+ii)= 0.5 * M_PI;
    *(sin2v0+ii)= 1.;
    *(potu0v0+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					nargs,targs);
    shux= sinh(*(ux+ii));
    pux= tdelta * (*(vR+ii) * cosh(*(ux+ii)) * sin(*(vx+ii))
		   + *(vz+ii) * shux * cos(*(vx+ii)));
    *(I3U+ii)= *(E+ii) * shux * shux
      - 0.5 * pux * pux / tdelta / tdelta
      - Lz22delta / shux / shux
      - ( shux * shux + 1. )
      * evaluatePotentialsUV(*(ux+ii),0.5 * M_PI,tdelta,nargs,targs)
      + ( *(sinh2u0+ii) + 1. ) * *(potu0v0+ii);
    *(full+ii)= 1 - warm;
    for (jj=0; jj < 2 && ! *(full+ii); jj++){
      x= jj == 0 ? *(umin+ii) : *(umax+ii);
      // (a step below 1e-6 leaves its square)
      dx= 1.;
      for (kk=0; kk < 4 && fabs(dx) > 1.e-6 * x; kk++){
	dx= SuTermsStaeckel(x,*(E+ii),*(I3U+ii),Lz22delta,tdelta,0.5 * M_PI,
			    1.,*(sinh2u0+ii),*(potu0v0+ii),nargs,targs)
	  / dSuduStaeckel(x,*(E+ii),Lz22delta,tdelta,0.5 * M_PI,1.,
			  nargs,targs);
	x-= dx;
      }
      if ( ! ( fabs(dx) <= 1.e-6 * x ) ) *(full+ii)= 1;
      else if ( jj == 0 ) *(umin+ii)= x;
      else *(umax+ii)= x;
    }
    if ( ! *(full+ii) && ! ( *(umin+ii) < *(umax+ii) ) ) *(full+ii)= 1;
  }
  for (ii=0; ii < ndata; ii++) nfull+= *(full+ii);
  if ( nfull == ndata )
    calcUminUmax(ndata,umin,umax,ux,ux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,
		 sin2v0,potu0v0,nargs,actionAngleArgs);
  else if ( nfull > 0 ){
    // (the compressed arrays of the points to solve anew)
    double *sub= (double *) malloc ( 11 * nfull * sizeof(double) );
    double *arr[9]= {ux,E,Lz,I3U,u0,sinh2u0,v0,sin2v0,potu0v0};
    for (ii=0, kk=0; ii < ndata; ii++)
      if ( *(full+ii) ){
	for (jj=0; jj < 9; jj++) *(sub+jj*nfull+kk)= *(arr[jj]+ii);
	*(sub+9*nfull+kk++)= *(delta+ii*delta_stride);
      }
    // (the last block doubles as umin, the first as umax once ux is used)
    double *smin= sub+10*nfull, *smax= (double *) malloc ( nfull * sizeof(double) );
    calcUminUmax(nfull,smin,smax,sub,sub,sub+nfull,sub+2*nfull,sub+3*nfull,
		 nfull,sub+9*nfull,sub+4*nfull,sub+5*nfull,sub+6*nfull,
		 sub+7*nfull,sub+8*nfull,nargs,actionAngleArgs);
    for (ii=0, kk=0; ii < ndata; ii++)
      if ( *(full+ii) ){
	*(umin+ii)= *(smin+kk);
	*(umax+ii)= *(smax+kk++);
      }
    free(sub);
    free(smax);
  }
  for (ii=0; ii < ndata; ii++){
    sm= sinh(*(umin+ii));
    sp= sinh(*(umax+ii));
    *(u0new+ii)= asinh(2. * sm * sp / ( sm + sp ));
  }
  free(work);
  free(full);
}
// The fixed point of that rule for each point, by iterating it with
// Aitken's acceleration until the distance left (that of the geometric
// series of the remaining steps, at the rate of the last two) is below
// MODEL_TOL: from the u0 handed in where that is finite, else from the
// point's own u
#define MODEL_TOL 1.e-9
#define MODEL_MAXITER 10
#define MODEL_NWORK 12
static void calcModelu0(int ndata,
			double *R,
			double *vR,
			double *vT,
			double *z,
			double *vz,
			double *u0,
			int ndelta,
			double * delta,
			int nargs,
			struct potentialArg * actionAngleArgs){
  int ii, jj, kk, iter, n= ndata, m;
  double d1, d2, den, acc;
  int delta_stride= ndelta == 1 ? 0 : 1;
  int *idx= (int *) malloc ( ndata * sizeof(int) );
  // the points still iterated, compressed: (R,vR,vT,z,vz,delta), three
  // iterates, the turning points, and the rate of convergence
  double *work= (double *) malloc ( MODEL_NWORK * ndata * sizeof(double) );
  double *in[5]= {R,vR,vT,z,vz};
  double *cdelta= work+5*ndata, *c0= work+6*ndata, *c1= work+7*ndata;
  double *c2= work+8*ndata, *cmin= work+9*ndata, *cmax= work+10*ndata;
  double *rate= work+11*ndata;
  for (ii=0; ii < ndata; ii++){
    *(idx+ii)= ii;
    for (jj=0; jj < 5; jj++) *(work+jj*ndata+ii)= *(in[jj]+ii);
    *(cdelta+ii)= *(delta+ii*delta_stride);
    *(c0+ii)= isfinite(*(u0+ii)) ? *(u0+ii)
      : asinh(*(R+ii) / *(cdelta+ii));
    *(rate+ii)= 0.9;
  }
  for (iter=0; iter < MODEL_MAXITER && n > 0; iter++){
    modelRuleu0(n,work,work+ndata,work+2*ndata,work+3*ndata,work+4*ndata,
		c0,c1,cmin,cmax,iter > 0,n,cdelta,nargs,actionAngleArgs);
    // done where this step alone leaves less than the tolerance
    for (kk=0, m=0; kk < n; kk++){
      d1= fabs(*(c1+kk) - *(c0+kk));
      if ( ! isfinite(d1)
	   || d1 * *(rate+kk) / ( 1. - *(rate+kk) ) <= MODEL_TOL )
	*(u0+*(idx+kk))= *(c1+kk);
      else {
	for (jj=0; jj < MODEL_NWORK; jj++)
	  *(work+jj*ndata+m)= *(work+jj*ndata+kk);
	*(idx+m++)= *(idx+kk);
      }
    }
    n= m;
    if ( n == 0 ) break;
    modelRuleu0(n,work,work+ndata,work+2*ndata,work+3*ndata,work+4*ndata,
		c1,c2,cmin,cmax,1,n,cdelta,nargs,actionAngleArgs);
    for (kk=0, m=0; kk < n; kk++){
      d1= fabs(*(c1+kk) - *(c0+kk));
      d2= fabs(*(c2+kk) - *(c1+kk));
      *(rate+kk)= d2 / ( d1 > 0. ? d1 : 1. );
      if ( ! ( *(rate+kk) < 0.9 ) ) *(rate+kk)= 0.9;
      den= *(c2+kk) - 2. * *(c1+kk) + *(c0+kk);
      acc= *(c2+kk) - ( *(c2+kk) - *(c1+kk) ) * ( *(c2+kk) - *(c1+kk) ) / den;
      acc= ( den != 0. && isfinite(acc) ) ? acc : *(c2+kk);
      *(u0+*(idx+kk))= acc;
      if ( isfinite(d2)
	   && d2 * *(rate+kk) / ( 1. - *(rate+kk) ) > MODEL_TOL ){
	*(c0+kk)= acc;
	for (jj=0; jj < MODEL_NWORK; jj++)
	  *(work+jj*ndata+m)= *(work+jj*ndata+kk);
	*(idx+m++)= *(idx+kk);
      }
    }
    n= m;
  }
  free(idx);
  free(work);
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
					 4.4408920985006262e-16,
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
				      int model,
				      double *umin,
				      double *umax,
				      double *vmin,
				      int * err){
  // Just copied this over from actionAngleStaeckel_actions below, not elegant
  // but does the job...
  int ii;
  double tdelta;
  //Set up the potentials
  // one parsed copy of the potentials per OpenMP thread (see AA_TARGS above)
#ifdef _OPENMP
  // cap copies at the point count, as the orbit integrators cap their team
  int aa_nthreads= ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  int aa_nthreads= 1;
#endif
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( aa_nthreads * npot * sizeof (struct potentialArg) );
  int aa_tid;
  int * aa_pot_type;
  double * aa_pot_args;
  tfuncs_type_arr aa_pot_tfuncs;
  for (aa_tid=0; aa_tid < aa_nthreads; aa_tid++) {
    aa_pot_type= pot_type;
    aa_pot_args= pot_args;
    aa_pot_tfuncs= pot_tfuncs;
    parse_leapFuncArgs_Full(npot,actionAngleArgs+aa_tid*npot,
			    &aa_pot_type,&aa_pot_args,&aa_pot_tfuncs);
  }
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  if ( model >= 2 )
    calcModelu0(ndata,R,vR,vT,z,vz,u0,ndelta,delta,npot,actionAngleArgs);
  if ( model )
    calcEModel(ndata,vR,vT,vz,ux,vx,u0,ndelta,delta,E,npot,actionAngleArgs);
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
#pragma omp parallel for num_threads(aa_nthreads) schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    struct potentialArg * targs= AA_TARGS(actionAngleArgs,npot);
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
					npot,targs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,targs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,targs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,targs);
  }
  //Calculate 'peri' and 'apo'centers
  calcUminUmax(ndata,umin,umax,ux,pux,E,Lz,I3U,ndelta,delta,u0,sinh2u0,v0,
	       sin2v0,potu0v0,npot,actionAngleArgs);
  calcVmin(ndata,vmin,vx,pvx,E,Lz,I3V,ndelta,delta,u0,cosh2u0,sinh2u0,potupi2,
	   npot,actionAngleArgs);
  //Free
  free_potentialArgs(aa_nthreads*npot,actionAngleArgs);
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
				 int model,
				 int order,
				 double *jr,
				 double *jz,
				 int * err){
  int ii;
  double tdelta;
  //Set up the potentials
  // one parsed copy of the potentials per OpenMP thread (see AA_TARGS above)
#ifdef _OPENMP
  // cap copies at the point count, as the orbit integrators cap their team
  int aa_nthreads= ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  int aa_nthreads= 1;
#endif
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( aa_nthreads * npot * sizeof (struct potentialArg) );
  int aa_tid;
  int * aa_pot_type;
  double * aa_pot_args;
  tfuncs_type_arr aa_pot_tfuncs;
  for (aa_tid=0; aa_tid < aa_nthreads; aa_tid++) {
    aa_pot_type= pot_type;
    aa_pot_args= pot_args;
    aa_pot_tfuncs= pot_tfuncs;
    parse_leapFuncArgs_Full(npot,actionAngleArgs+aa_tid*npot,
			    &aa_pot_type,&aa_pot_args,&aa_pot_tfuncs);
  }
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  if ( model >= 2 )
    calcModelu0(ndata,R,vR,vT,z,vz,u0,ndelta,delta,npot,actionAngleArgs);
  if ( model )
    calcEModel(ndata,vR,vT,vz,ux,vx,u0,ndelta,delta,E,npot,actionAngleArgs);
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
#pragma omp parallel for num_threads(aa_nthreads) schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    struct potentialArg * targs= AA_TARGS(actionAngleArgs,npot);
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
					npot,targs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,targs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,targs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,targs);
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
  free_potentialArgs(aa_nthreads*npot,actionAngleArgs);
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
  double mid;
#ifdef _OPENMP
  nthreads = ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * JRInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) malloc ( nthreads * sizeof (struct JRStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk)				\
  private(tid,ii,mid)							\
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
    (params+tid)->umin= *(umin+ii);
    (params+tid)->umax= *(umax+ii);
    (JRInt+tid)->params = params+tid;
    // each half runs to the midpoint of the oscillation, y = 1/2
    mid= 0.5 * M_PI;
    //Integrate. Near the axis the low half straddles the inner cusp, so it uses
    //the graded composite rule on the model-free integrand; every other orbit
    //keeps the cheap single-panel rule. The high half is a normal turning point.
    if ( *(umin+ii) < STAECKEL_NEARAXIS ){
      (JRInt+tid)->function = &JRLowStaeckelIntegrandDirect;
      *(jr+ii)= glfixed_graded (JRInt+tid,0.,mid,T);
    } else {
      (JRInt+tid)->function = &JRLowStaeckelIntegrand;
      *(jr+ii)= gsl_integration_glfixed (JRInt+tid,0.,mid,T);
    }
    (JRInt+tid)->function = &JRHighStaeckelIntegrand;
    *(jr+ii)+= gsl_integration_glfixed (JRInt+tid,0.,mid,T);
    *(jr+ii)*= sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
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
  double mid;
#ifdef _OPENMP
  nthreads = ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * JzInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct JzStaeckelArg * params= (struct JzStaeckelArg *) malloc ( nthreads * sizeof (struct JzStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk)				\
  private(tid,ii,mid)							\
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
    (params+tid)->vmin= *(vmin+ii);
    (JzInt+tid)->function = &JzLowStaeckelIntegrand;
    (JzInt+tid)->params = params+tid;
    // the chi half runs to the midpoint of [vmin, pi/2], y = 1/4; the
    // midplane half is not a turning point and keeps its t^2 form
    mid= sqrt( 0.5 * ( M_PI/2. - *(vmin+ii) ) );
    //Integrate
    *(jz+ii)= gsl_integration_glfixed (JzInt+tid,0.,STAECKEL_CHI_MID,T);
    (JzInt+tid)->function = &JzHighStaeckelIntegrand;
    *(jz+ii)+= gsl_integration_glfixed (JzInt+tid,0.,mid,T);
    *(jz+ii)*= 2 * sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
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
				      int model,
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
  // one parsed copy of the potentials per OpenMP thread (see AA_TARGS above)
#ifdef _OPENMP
  // cap copies at the point count, as the orbit integrators cap their team
  int aa_nthreads= ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  int aa_nthreads= 1;
#endif
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( aa_nthreads * npot * sizeof (struct potentialArg) );
  int aa_tid;
  int * aa_pot_type;
  double * aa_pot_args;
  tfuncs_type_arr aa_pot_tfuncs;
  for (aa_tid=0; aa_tid < aa_nthreads; aa_tid++) {
    aa_pot_type= pot_type;
    aa_pot_args= pot_args;
    aa_pot_tfuncs= pot_tfuncs;
    parse_leapFuncArgs_Full(npot,actionAngleArgs+aa_tid*npot,
			    &aa_pot_type,&aa_pot_args,&aa_pot_tfuncs);
  }
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
#pragma omp parallel for num_threads(aa_nthreads) schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    struct potentialArg * targs= AA_TARGS(actionAngleArgs,npot);
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
					npot,targs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,targs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,targs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,targs);
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
  free_potentialArgs(aa_nthreads*npot,actionAngleArgs);
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
					    int model,
					    int order,
					    double *jr,
					    double *jz,
					    double *Omegar,
					    double *Omegaphi,
					    double *Omegaz,
					    double *Angler,
					    double *Anglephi,
					    double *Anglez,
					    double * ddelta,
					    int * err){
  int ii;
  double tdelta;
  //Set up the potentials
  // one parsed copy of the potentials per OpenMP thread (see AA_TARGS above)
#ifdef _OPENMP
  // cap copies at the point count, as the orbit integrators cap their team
  int aa_nthreads= ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  int aa_nthreads= 1;
#endif
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( aa_nthreads * npot * sizeof (struct potentialArg) );
  int aa_tid;
  int * aa_pot_type;
  double * aa_pot_args;
  tfuncs_type_arr aa_pot_tfuncs;
  for (aa_tid=0; aa_tid < aa_nthreads; aa_tid++) {
    aa_pot_type= pot_type;
    aa_pot_args= pot_args;
    aa_pot_tfuncs= pot_tfuncs;
    parse_leapFuncArgs_Full(npot,actionAngleArgs+aa_tid*npot,
			    &aa_pot_type,&aa_pot_args,&aa_pot_tfuncs);
  }
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,ndelta,delta);
  if ( model >= 2 )
    calcModelu0(ndata,R,vR,vT,z,vz,u0,ndelta,delta,npot,actionAngleArgs);
  if ( model )
    calcEModel(ndata,vR,vT,vz,ux,vx,u0,ndelta,delta,E,npot,actionAngleArgs);
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
#pragma omp parallel for num_threads(aa_nthreads) schedule(static,chunk) private(ii,tdelta)
  for (ii=0; ii < ndata; ii++){
    struct potentialArg * targs= AA_TARGS(actionAngleArgs,npot);
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
					npot,targs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / tdelta / tdelta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinhux+ii) / *(sinhux+ii)
      - ( *(sinhux+ii) * *(sinhux+ii) + *(sin2v0+ii))
      *evaluatePotentialsUV(*(ux+ii),*(v0+ii),tdelta,
			    npot,targs)
      + ( *(sinh2u0+ii) + *(sin2v0+ii) )* *(potu0v0+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(u0+ii),0.5 * M_PI,tdelta,
					npot,targs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / tdelta / tdelta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / *(sinvx+ii) / *(sinvx+ii)
      - *(cosh2u0+ii) * *(potupi2+ii)
      + ( *(sinh2u0+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(u0+ii),*(vx+ii),tdelta,
			     npot,targs);
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
		     npot,actionAngleArgs,order,model == 2,ddelta);
  //Free
  free_potentialArgs(aa_nthreads*npot,actionAngleArgs);
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
  nthreads = ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * dJRInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) malloc ( nthreads * sizeof (struct dJRStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
    (params+tid)->wd= 0;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk)				\
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
    // each half runs to the midpoint of the oscillation, y = 1/2
    mid= 0.5 * M_PI;
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
  nthreads = ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * dJzInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) malloc ( nthreads * sizeof (struct dJzStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (params+tid)->nargs= nargs;
    (params+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
    (params+tid)->wu0= 0;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk)				\
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
    *(djzdE+ii)= gsl_integration_glfixed (dJzInt+tid,0.,STAECKEL_CHI_MID,T);
    (dJzInt+tid)->function = &dJzdEHighStaeckelIntegrand;
    *(djzdE+ii)+= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    *(djzdE+ii)*= sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
    //Then calculate dJzdLz
    (dJzInt+tid)->function = &dJzdLzLowStaeckelIntegrand;
    //Integrate
    *(djzdLz+ii)= gsl_integration_glfixed (dJzInt+tid,0.,STAECKEL_CHI_MID,T);
    (dJzInt+tid)->function = &dJzdLzHighStaeckelIntegrand;
    *(djzdLz+ii)+= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    *(djzdLz+ii)*= - *(Lz+ii) * sqrt(2.) / M_PI / *(delta+ii*delta_stride);
    //Then calculate dJzdI3
    (dJzInt+tid)->function = &dJzdI3LowStaeckelIntegrand;
    //Integrate
    *(djzdI3+ii)= gsl_integration_glfixed (dJzInt+tid,0.,STAECKEL_CHI_MID,T);
    (dJzInt+tid)->function = &dJzdI3HighStaeckelIntegrand;
    *(djzdI3+ii)+= gsl_integration_glfixed (dJzInt+tid,0.,mid,T);
    *(djzdI3+ii)*= sqrt(2.) * *(delta+ii*delta_stride) / M_PI;
  }
  free(dJzInt);
  free(params);
  gsl_integration_glfixed_table_free ( T );
}
// The integral of the dJz/dI3 integrand along the orbit to the current v,
// from the same origin as I3r2 in calcAnglesStaeckel below (dJzdX: the
// complete integral)
static inline double anglevPartialStaeckel(gsl_function * f,
					   gsl_integration_glfixed_table * T,
					   double vx,double pvx,double vmin,
					   double delta,double dJzdX){
  double mid, out;
  double midpoint= vmin + 0.5 * ( 0.5 * M_PI - vmin );
  int above= vx > 0.5 * M_PI;
  if ( vx < midpoint || vx > (M_PI - midpoint) ) {
    mid= 2. * asin( sqrt( fmin ( fmax ( ( above ? ( M_PI - vx - vmin )
					  : ( vx - vmin ) )
					/ ( M_PI - 2. * vmin ), 0. ), 1. ) ) );
    f->function = &dJzdI3LowStaeckelIntegrand;
    out= gsl_integration_glfixed (f,0.,mid,T) * delta / sqrt(2.);
    if ( pvx > 0. ) return above ? M_PI * dJzdX - out : out;
    return above ? M_PI * dJzdX + out : 2. * M_PI * dJzdX - out;
  }
  mid= sqrt( fabs ( 0.5 * M_PI - vx ) );
  f->function = &dJzdI3HighStaeckelIntegrand;
  out= gsl_integration_glfixed (f,0.,mid,T) * delta / sqrt(2.);
  if ( pvx > 0. ) return 0.5 * M_PI * dJzdX + ( above ? out : -out );
  return 1.5 * M_PI * dJzdX + ( above ? -out : out );
}
// The same for the dJR/dI3 integrand and the current u (without the
// delta / sqrt(2) and the sign of I3r1 in calcAnglesStaeckel)
static inline double angleuPartialStaeckel(gsl_function * f,
					   gsl_integration_glfixed_table * T,
					   double ux,double pux,double umin,
					   double umax,double dJRdX){
  int high= ux > umin + 0.5 * ( umax - umin );
  double mid= 2. * asin( sqrt( fmin ( fmax ( ( high ? umax - ux : ux - umin )
					     / ( umax - umin ), 0. ), 1. ) ) );
  double out;
  f->function = high ? &dJRdI3HighStaeckelIntegrand
    : &dJRdI3LowStaeckelIntegrand;
  out= gsl_integration_glfixed (f,0.,mid,T) / sqrt(2.);
  if ( high ) return M_PI * dJRdX + ( pux > 0. ? -out : out );
  return pux > 0. ? out : 2. * M_PI * dJRdX - out;
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
			int order,
			int torus,
			double * ddelta){
  int ii, tid, nthreads;
  double Or1, Or2, I3r1, I3r2,phitmp;
  double mid, midpoint;
#ifdef _OPENMP
  nthreads = ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
#else
  nthreads = 1;
#endif
  gsl_function * AngleuInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  gsl_function * AnglevInt= (gsl_function *) malloc ( nthreads * sizeof(gsl_function) );
  struct dJRStaeckelArg * paramsu= (struct dJRStaeckelArg *) malloc ( nthreads * sizeof (struct dJRStaeckelArg) );
  struct dJzStaeckelArg * paramsv= (struct dJzStaeckelArg *) malloc ( nthreads * sizeof (struct dJzStaeckelArg) );
  for (tid=0; tid < nthreads; tid++){
    (paramsu+tid)->nargs= nargs;
    (paramsu+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
    (paramsu+tid)->wd= 0;
    (paramsv+tid)->nargs= nargs;
    (paramsv+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
    (paramsv+tid)->wu0= 0;
  }
  //Setup integrator
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk)				\
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
	mid= 2. * asin( sqrt( fmin ( fmax ( ( *(umax+ii) - *(ux+ii) )
			      / ( *(umax+ii) - *(umin+ii) ), 0. ), 1. ) ) );
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
	mid= 2. * asin( sqrt( fmin ( fmax ( ( *(ux+ii) - *(umin+ii) )
			      / ( *(umax+ii) - *(umin+ii) ), 0. ), 1. ) ) );
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
	mid= 2. * asin( sqrt( fmin ( fmax ( ( *(umax+ii) - *(ux+ii) )
			      / ( *(umax+ii) - *(umin+ii) ), 0. ), 1. ) ) );
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
	mid= 2. * asin( sqrt( fmin ( fmax ( ( *(ux+ii) - *(umin+ii) )
			      / ( *(umax+ii) - *(umin+ii) ), 0. ), 1. ) ) );
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
	// chi of the current v, measured from the vmin turning point along
	// the full loop [vmin, pi - vmin]; v beyond the midplane mirrors
	mid = 2. * asin( sqrt( fmin ( fmax ( ( ( *(vx+ii) > 0.5 * M_PI )
				 ? ( M_PI - *(vx+ii) - *(vmin+ii) )
				 : ( *(vx+ii) - *(vmin+ii) ) )
			       / ( M_PI - 2. * *(vmin+ii) ), 0. ), 1. ) ) );
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
	// chi of the current v, measured from the vmin turning point along
	// the full loop [vmin, pi - vmin]; v beyond the midplane mirrors
	mid = 2. * asin( sqrt( fmin ( fmax ( ( ( *(vx+ii) > 0.5 * M_PI )
				 ? ( M_PI - *(vx+ii) - *(vmin+ii) )
				 : ( *(vx+ii) - *(vmin+ii) ) )
			       / ( M_PI - 2. * *(vmin+ii) ), 0. ), 1. ) ) );
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
    if ( torus ) {
      // A reference curve that follows the tori, delta sinh(u0) = 2 / (1/R_-
      // + 1/R_+) at the turning points of the u libration (u0='torus' of
      // actionAngleStaeckelInverse): the family's generating function is
      // the model's at u0(J), so that its angles are the model's + K du0/dJ
      // with K = dS/du0 at fixed position and actions, and its frequencies
      // dE/dJ the model's + (dE/du0 at fixed actions) du0/dJ
      double tdelta= *(delta+ii*delta_stride);
      double dJzdu0, K, dEdu0, norm, sh2, chain, Rm, Rp;
      double r[4]= {0.,0.,0.,0.}, du0[3];
      double uturn[2]= {*(umin+ii),*(umax+ii)};
      int jj;
      (paramsv+tid)->wu0= 1;
      mid= sqrt( 0.5 * (M_PI/2. - *(vmin+ii) ) );
      (AnglevInt+tid)->function = &dJzdI3LowStaeckelIntegrand;
      dJzdu0= gsl_integration_glfixed (AnglevInt+tid,0.,STAECKEL_CHI_MID,T);
      (AnglevInt+tid)->function = &dJzdI3HighStaeckelIntegrand;
      dJzdu0+= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
      dJzdu0*= sqrt(2.) * tdelta / M_PI;
      K= anglevPartialStaeckel(AnglevInt+tid,T,*(vx+ii),*(pvx+ii),*(vmin+ii),
			       tdelta,dJzdu0)
	- ( *(Omegaz+ii) * ( Or1 + Or2 )
	    + *(dI3dJz+ii) * ( I3r1 + I3r2 ) ) * dJzdu0;
      (paramsv+tid)->wu0= 0;
      // the rule's partials in (E, I3, Lz), by the level-set rule on the
      // two turning points
      Rm= tdelta * sinh(uturn[0]);
      Rp= tdelta * sinh(uturn[1]);
      for (jj=0; jj < 2; jj++){
	sh2= sinh(uturn[jj]) * sinh(uturn[jj]);
	chain= 2. * ( jj == 0 ? Rp * Rp : Rm * Rm ) / ( Rm + Rp ) / ( Rm + Rp )
	  * cosh(uturn[jj]) / cosh(*(u0+ii))
	  / dSuduStaeckel(uturn[jj],*(E+ii),(paramsu+tid)->Lz22delta,tdelta,
			  *(v0+ii),*(sin2v0+ii),nargs,
			  actionAngleArgs + tid * nargs);
	r[0]-= chain * sh2;
	r[1]+= chain;
	r[2]+= chain * *(Lz+ii) / tdelta / tdelta / sh2;
	r[3]-= chain * ( cosh(uturn[jj]) * cosh(uturn[jj]) * sinh(uturn[jj])
			 * calcRforce(tdelta * sinh(uturn[jj]),0.,0.,0.,nargs,
				      actionAngleArgs + tid * nargs)
			 + *(Lz+ii) * *(Lz+ii) / tdelta / tdelta / tdelta / sh2 );
      }
      dEdu0= - *(Omegaz+ii) * dJzdu0;
      norm= 1. - r[0] * dEdu0 + r[1] * *(dI3dJz+ii) * dJzdu0;
      du0[0]= r[0] * *(Omegar+ii) + r[1] * *(dI3dJR+ii);
      du0[1]= r[0] * *(Omegaphi+ii) + r[1] * *(dI3dLz+ii) + r[2];
      du0[2]= r[0] * *(Omegaz+ii) + r[1] * *(dI3dJz+ii);
      if ( ddelta ) {
	// A focal length that also follows the tori, with derivatives ddelta
	// in (J_R, L_z, J_z): K_delta = dS/ddelta at fixed position and
	// actions is the dilation of the point transformation, -(R p_R + z
	// p_z)/delta, plus the integrals of dp_u/ddelta and dp_v/ddelta along
	// the orbit less the angles times their complete versions
	double dJRdD, dJzdD, KD, dEdD, dI3dD, shu, snv, feed;
	double dD[3]= {*(ddelta+ii),*(ddelta+ndata+ii),*(ddelta+2*ndata+ii)};
	(paramsu+tid)->wd= 1;
	(AngleuInt+tid)->function = &dJRdI3LowStaeckelIntegrand;
	dJRdD= gsl_integration_glfixed (AngleuInt+tid,0.,0.5 * M_PI,T);
	(AngleuInt+tid)->function = &dJRdI3HighStaeckelIntegrand;
	dJRdD+= gsl_integration_glfixed (AngleuInt+tid,0.,0.5 * M_PI,T);
	dJRdD/= M_PI * sqrt(2.);
	KD= angleuPartialStaeckel(AngleuInt+tid,T,*(ux+ii),*(pux+ii),
				  *(umin+ii),*(umax+ii),dJRdD);
	(paramsu+tid)->wd= 0;
	(paramsv+tid)->wu0= 2;
	(AnglevInt+tid)->function = &dJzdI3LowStaeckelIntegrand;
	dJzdD= gsl_integration_glfixed (AnglevInt+tid,0.,STAECKEL_CHI_MID,T);
	(AnglevInt+tid)->function = &dJzdI3HighStaeckelIntegrand;
	dJzdD+= gsl_integration_glfixed (AnglevInt+tid,0.,mid,T);
	dJzdD*= sqrt(2.) / M_PI;
	KD+= anglevPartialStaeckel(AnglevInt+tid,T,*(vx+ii),*(pvx+ii),
				   *(vmin+ii),1.,dJzdD);
	(paramsv+tid)->wu0= 0;
	shu= sinh(*(ux+ii));
	snv= sin(*(vx+ii));
	KD-= ( shu * cosh(*(ux+ii)) * *(pux+ii)
	       - snv * cos(*(vx+ii)) * *(pvx+ii) )
	  / ( shu * shu + snv * snv ) / tdelta
	  + ( *(Omegar+ii) * ( Or1 + Or2 )
	      + *(dI3dJR+ii) * ( I3r1 + I3r2 ) ) * dJRdD
	  + ( *(Omegaz+ii) * ( Or1 + Or2 )
	      + *(dI3dJz+ii) * ( I3r1 + I3r2 ) ) * dJzdD;
	dEdD= - ( *(Omegar+ii) * dJRdD + *(Omegaz+ii) * dJzdD );
	dI3dD= - ( *(dI3dJR+ii) * dJRdD + *(dI3dJz+ii) * dJzdD );
	feed= r[0] * dEdD + r[1] * dI3dD + r[3];
	for (jj=0; jj < 3; jj++) du0[jj]+= feed * dD[jj];
	*(Angler+ii)+= KD * dD[0];
	*(Anglephi+ii)+= KD * dD[1];
	*(Anglez+ii)+= KD * dD[2];
	*(Omegar+ii)+= dEdD * dD[0];
	*(Omegaphi+ii)+= dEdD * dD[1];
	*(Omegaz+ii)+= dEdD * dD[2];
      }
      for (jj=0; jj < 3; jj++) du0[jj]/= norm;
      *(Angler+ii)= fmod(*(Angler+ii) + K * du0[0] + 4. * M_PI,2. * M_PI);
      *(Anglephi+ii)+= K * du0[1];
      *(Anglez+ii)= fmod(*(Anglez+ii) + K * du0[2] + 4. * M_PI,2. * M_PI);
      *(Omegar+ii)+= dEdu0 * du0[0];
      *(Omegaphi+ii)+= dEdu0 * du0[1];
      *(Omegaz+ii)+= dEdu0 * du0[2];
    }
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
  nthreads = ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
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
    (params+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
    (s+tid)->s= gsl_root_fsolver_alloc (T);
  }
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
  gsl_set_error_handler_off();
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk)				\
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
	//The point lies within 1e-6 of umax. Adopting the point itself as
	//the turning point is fine for the actions (an O(eps) endpoint
	//error only enters them at O(eps^1.5)), but the frequency and
	//angle integrands diverge as 1/sqrt(W) at the endpoint, so there
	//it enters at O(sqrt(eps)); solve for the true root instead (the
	//+-1e-6 bracket has the sign change by the branch condition),
	//keeping the point itself only as the degenerate fallback
	status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid,
				       *(ux+ii) - 0.000001,
				       *(ux+ii) + 0.000001);
	if (status == GSL_EINVAL)
// LCOV_EXCL_START
	  *(umax+ii)= *(ux+ii);//can't happen: the branch condition peps*meps<0 is the bracket
// LCOV_EXCL_STOP
	else {
	  iter= 0;
	  do
	    {
	      iter++;
	      status = gsl_root_fsolver_iterate ((s+tid)->s);
	      u_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	      u_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	      status = gsl_root_test_interval (u_lo, u_hi,
					       4.4408920985006262e-16,
					       4.4408920985006262e-16);
	    }
	  while (status == GSL_CONTINUE && iter < max_iter);
	  *(umax+ii)= gsl_root_fsolver_root ((s+tid)->s);
	}
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
					       4.4408920985006262e-16,
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
	//Same as the umax case above: solve for the true root, with the
	//point itself only as the degenerate fallback
	status = gsl_root_fsolver_set ((s+tid)->s, JRRoot+tid,
				       *(ux+ii) - 0.000001,
				       *(ux+ii) + 0.000001);
	if (status == GSL_EINVAL)
// LCOV_EXCL_START
	  *(umin+ii)= *(ux+ii);//can't happen: the branch condition peps*meps<0 is the bracket
// LCOV_EXCL_STOP
	else {
	  iter= 0;
	  do
	    {
	      iter++;
	      status = gsl_root_fsolver_iterate ((s+tid)->s);
	      u_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	      u_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	      status = gsl_root_test_interval (u_lo, u_hi,
					       4.4408920985006262e-16,
					       4.4408920985006262e-16);
	    }
	  while (status == GSL_CONTINUE && iter < max_iter);
	  *(umin+ii)= gsl_root_fsolver_root ((s+tid)->s);
	}
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
					     4.4408920985006262e-16,
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
					     4.4408920985006262e-16,
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
					   4.4408920985006262e-16,
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
  nthreads = ndata < omp_get_max_threads() ? ndata : omp_get_max_threads();
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
    (params+tid)->actionAngleArgs= actionAngleArgs + tid * nargs;
    (s+tid)->s= gsl_root_fsolver_alloc (T);
  }
  int delta_stride= ndelta == 1 ? 0 : 1;
  UNUSED int chunk= CHUNKSIZE;
  gsl_set_error_handler_off();
#pragma omp parallel for num_threads(nthreads) schedule(static,chunk)				\
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
    if ( *(vx+ii) == 0.5 * M_PI
	 && fabs(*(pvx+ii)) <= 1e-12 * *(delta+ii*delta_stride) ) {
      //In the midplane (z == 0 makes v exactly pi/2) with no vertical
      //momentum: W is symmetric about pi/2 with W(pi/2) = pv^2/2delta^2, so
      //the true vmin is within ~|pv|/delta (< 1e-12) of pi/2. The root solve
      //below instead lands on a ~1e-8 root of W's rounding noise (W(pi/2)
      //is a difference of O(1) terms), which put zmax ~ 1e-8 on circular
      //orbits depending on the last bits of the potential.
      *(vmin+ii)= 0.5 * M_PI;
    }
    else if ( fabs(GSL_FN_EVAL(JzRoot+tid,*(vx+ii))) < 0.0000001) {//we are at vmin
      //The point lies within 1e-6 of vmin. Adopting the point itself as
      //the turning point is fine for the actions (an O(eps) endpoint
      //error only enters them at O(eps^1.5)), but the frequency and
      //angle integrands diverge as 1/sqrt(W) at the endpoint, so there
      //it enters at O(sqrt(eps)); solve for the true root instead: W is
      //negative below vmin and positive above, so the +-1e-6 bracket
      //around the (mirrored) point holds unless the point is degenerate
      //at roundoff level, where the point IS the turning point and the
      //old assignment remains as the fallback
      v_lo= ( ( *(vx+ii) > 0.5 * M_PI ) ? M_PI - *(vx+ii): *(vx+ii) )
	- 0.000001;
      if ( v_lo < 0.000000001 ) v_lo= 0.000000001;
      v_hi= ( ( *(vx+ii) > 0.5 * M_PI ) ? M_PI - *(vx+ii): *(vx+ii) )
	+ 0.000001;
      if ( v_hi > 0.5 * M_PI ) v_hi= 0.5 * M_PI;
      status = gsl_root_fsolver_set ((s+tid)->s, JzRoot+tid, v_lo, v_hi);
      if (status == GSL_EINVAL)
	*(vmin+ii)= ( *(vx+ii) > 0.5 * M_PI ) ? M_PI - *(vx+ii): *(vx+ii);
      else {
	iter= 0;
	do
	  {
	    iter++;
	    status = gsl_root_fsolver_iterate ((s+tid)->s);
	    v_lo = gsl_root_fsolver_x_lower ((s+tid)->s);
	    v_hi = gsl_root_fsolver_x_upper ((s+tid)->s);
	    status = gsl_root_test_interval (v_lo, v_hi,
					     4.4408920985006262e-16,
					     4.4408920985006262e-16);
	  }
	while (status == GSL_CONTINUE && iter < max_iter);
	*(vmin+ii)= gsl_root_fsolver_root ((s+tid)->s);
      }
    }
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
					   4.4408920985006262e-16,
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

/*
  The chi anomaly, q = qmin + D sin^2(chi/2), as used by the pure-Python
  path.  With y = sin^2(chi/2) and Q = S/[y(1-y)],

      int sqrt(S) dq = (D/4) int sqrt(Q) sin^2(chi) dchi
      int f/sqrt(S) dq = D int f/sqrt(Q) dchi

  and Q is regular at the turning point, where S itself is a difference of
  O(1) terms cancelling to zero: its relative error blows up while its
  absolute error stays at machine epsilon, and the 1/sqrt(S) of the
  frequency and angle integrands amplifies it.  The t^2 substitution this
  replaces put Gauss-Legendre nodes within ~(range/order^2)^2 of the turning
  point, so raising the order moved them closer and made the frequencies
  WORSE, not better: dOmega_z/Omega_z was 1.8e-9 at order 200 and 3.3e-8 at
  order 800, against ~1e-12 for the pure-Python path.

  Near the endpoints Q is taken from the analytic derivative instead, by the
  trapezoid between the turning point and the node,
  S ~ (q - q0) [S'(q0) + S'(q)] / 2, whose O((q-q0)^2) model error is far
  below the cancellation it replaces.
*/
static inline double SuTermsStaeckel(double u,double E,double I3U,
				     double Lz22delta,double delta,double v0,
				     double sin2v0,double sinh2u0,
				     double potu0v0,int nargs,
				     struct potentialArg * args){
  double sinh2u= sinh(u) * sinh(u);
  double dU= (sinh2u+sin2v0)
    * evaluatePotentialsUV(u,v0,delta,nargs,args)
    - (sinh2u0+sin2v0) * potu0v0;
  return E * sinh2u - I3U - dU
    - ( Lz22delta > 0. ? Lz22delta / sinh2u : 0. );
}
static inline double dSuduStaeckel(double u,double E,double Lz22delta,
				   double delta,double v0,double sin2v0,
				   int nargs,struct potentialArg * args){
  double R,z;
  double shu= sinh(u), chu= cosh(u), s2u= 2. * shu * chu;
  uv_to_Rz(u,v0,&R,&z,delta);
  double dPhidu= -delta
    * ( calcRforce(R,z,0.,0.,nargs,args) * chu * sin(v0)
	+ calczforce(R,z,0.,0.,nargs,args) * shu * cos(v0) );
  return E * s2u - s2u * evaluatePotentialsUV(u,v0,delta,nargs,args)
    - ( shu * shu + sin2v0 ) * dPhidu
    + ( Lz22delta > 0. ? 2. * Lz22delta * chu / ( shu * shu * shu ) : 0. );
}
// Q and the coordinate at anomaly chi; high selects the umax end
static inline double chiQuStaeckel(double chi,int high,double umin,double umax,
				   double E,double I3U,double Lz22delta,
				   double delta,double v0,double sin2v0,
				   double sinh2u0,double potu0v0,int nargs,
				   struct potentialArg * args,double * uu){
  double sc= sin(0.5*chi);
  double y= sc * sc;
  double D= umax - umin;
  double u= high ? umax - D * y : umin + D * y;
  double y1my= y * ( 1. - y );
  double Q;
  *uu= u;
  if ( y1my > STAECKEL_CHI_EDGE )
    Q= SuTermsStaeckel(u,E,I3U,Lz22delta,delta,v0,sin2v0,sinh2u0,potu0v0,
		       nargs,args) / y1my;
  else {
    double dS0= dSuduStaeckel(high ? umax : umin,E,Lz22delta,delta,v0,sin2v0,
			      nargs,args);
    double dSq= dSuduStaeckel(u,E,Lz22delta,delta,v0,sin2v0,nargs,args);
    Q= ( high ? -1. : 1. ) * D * ( dS0 + dSq ) / 2. / ( 1. - y );
  }
  return Q;
}

static inline double dmax(double x,double y){ return x > y ? x : y; }

// Q at anomaly chi WITHOUT the endpoint model, for the regular sqrt(S) action
// integrand near the axis (see chiQuStaeckel for the model the singular
// frequency/angle integrands still need).
static inline double chiQuStaeckelDirect(double chi,int high,double umin,
					 double umax,double E,double I3U,
					 double Lz22delta,double delta,double v0,
					 double sin2v0,double sinh2u0,
					 double potu0v0,int nargs,
					 struct potentialArg * args,double * uu){
  double sc= sin(0.5*chi), y= sc * sc, D= umax - umin;
  double u= high ? umax - D * y : umin + D * y;
  *uu= u;
  return SuTermsStaeckel(u,E,I3U,Lz22delta,delta,v0,sin2v0,sinh2u0,potu0v0,
			 nargs,args) / ( y * ( 1. - y ) );
}

// Composite GL over [a,b] with panels graded geometrically toward a (the cusp).
static inline double glfixed_graded(gsl_function * f,double a,double b,
				    gsl_integration_glfixed_table * T){
  double out= 0., lo, hi, w= b - a;
  int k, n= STAECKEL_CHI_NPANELS;
  for (k=0; k < n; k++){
    lo= k == 0 ? a : a + w * pow( 2., dmax( (double) ( k - n ), -52. ) );
    hi= a + w * pow( 2., dmax( (double) ( k + 1 - n ), -52. ) );
    out+= gsl_integration_glfixed(f,lo,hi,T);
  }
  return out;
}
static inline double SvTermsStaeckel(double v,double E,double I3V,
				     double Lz22delta,double delta,double u0,
				     double cosh2u0,double sinh2u0,
				     double potupi2,int nargs,
				     struct potentialArg * args){
  double sin2v= sin(v) * sin(v);
  double dV= cosh2u0 * potupi2
    - (sinh2u0+sin2v) * evaluatePotentialsUV(u0,v,delta,nargs,args);
  return E * sin2v + I3V + dV
    - ( Lz22delta > 0. ? Lz22delta / sin2v : 0. );
}
static inline double dSvdvStaeckel(double v,double E,double Lz22delta,
				   double delta,double u0,double sinh2u0,
				   int nargs,struct potentialArg * args){
  double R,z;
  double sv= sin(v), cv= cos(v), s2v= 2. * sv * cv;
  uv_to_Rz(u0,v,&R,&z,delta);
  double dPhidv= -delta
    * ( calcRforce(R,z,0.,0.,nargs,args) * sinh(u0) * cv
	- calczforce(R,z,0.,0.,nargs,args) * cosh(u0) * sv );
  return E * s2v - s2v * evaluatePotentialsUV(u0,v,delta,nargs,args)
    - ( sinh2u0 + sv * sv ) * dPhidv
    + ( Lz22delta > 0. ? 2. * Lz22delta * cv / ( sv * sv * sv ) : 0. );
}
// the v loop spans [vmin, pi - vmin]; only the vmin end is a turning point
// reached from here, the midplane being an interior symmetry point of S_z
static inline double chiQvStaeckel(double chi,double vmin,double E,double I3V,
				   double Lz22delta,double delta,double u0,
				   double cosh2u0,double sinh2u0,
				   double potupi2,int nargs,
				   struct potentialArg * args,double * vv){
  double sc= sin(0.5*chi);
  double y= sc * sc;
  double D= M_PI - 2. * vmin;
  double v= vmin + D * y;
  double y1my= y * ( 1. - y );
  double Q;
  *vv= v;
  if ( y1my > STAECKEL_CHI_EDGE )
    Q= SvTermsStaeckel(v,E,I3V,Lz22delta,delta,u0,cosh2u0,sinh2u0,potupi2,
		       nargs,args) / y1my;
  else
    Q= D * ( dSvdvStaeckel(vmin,E,Lz22delta,delta,u0,sinh2u0,nargs,args)
	     + dSvdvStaeckel(v,E,Lz22delta,delta,u0,sinh2u0,nargs,args) )
      / 2. / ( 1. - y );
  return Q;
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
double JRLowStaeckelIntegrand(double chi,
			      void * p){
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) p;
  double u, sc= sin(chi);
  double Q= chiQuStaeckel(chi,0,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  if ( Q <= 0. ) return 0.;
  return D / 4. * sqrt(Q) * sc * sc;
}

double JRLowStaeckelIntegrandDirect(double chi,
			      void * p){
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) p;
  double u, sc= sin(chi);
  double Q= chiQuStaeckelDirect(chi,0,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  if ( Q <= 0. ) return 0.;
  return D / 4. * sqrt(Q) * sc * sc;
}
double JRHighStaeckelIntegrand(double chi,
			      void * p){
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) p;
  double u, sc= sin(chi);
  double Q= chiQuStaeckel(chi,1,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  if ( Q <= 0. ) return 0.;
  return D / 4. * sqrt(Q) * sc * sc;
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
double JzLowStaeckelIntegrand(double chi,
			      void * p){
  struct JzStaeckelArg * params= (struct JzStaeckelArg *) p;
  double v, sc= sin(chi);
  double Q= chiQvStaeckel(chi,params->vmin,params->E,params->I3V,params->Lz22delta,
			  params->delta,params->u0,params->cosh2u0,params->sinh2u0,
			  params->potupi2,params->nargs,params->actionAngleArgs,&v);
  double D= M_PI - 2. * params->vmin;
  if ( Q <= 0. ) return 0.;
  return D / 4. * sqrt(Q) * sc * sc;
}
double JzHighStaeckelIntegrand(double t,
			       void * p){
  double v= M_PI/2. - t * t;
  return 2. * t * JzStaeckelIntegrand(v,p);
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
double dJRdELowStaeckelIntegrand(double chi,
			      void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u;
  double Q= chiQuStaeckel(chi,0,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  if ( Q <= 0. ) return 0.;
  return D * sinh(u) * sinh(u) / sqrt(Q);
}
double dJRdEHighStaeckelIntegrand(double chi,
			      void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u;
  double Q= chiQuStaeckel(chi,1,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  if ( Q <= 0. ) return 0.;
  return D * sinh(u) * sinh(u) / sqrt(Q);
}
double dJRdLzLowStaeckelIntegrand(double chi,
			      void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u;
  double Q= chiQuStaeckel(chi,0,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  if ( Q <= 0. ) return 0.;
  return D / sinh(u) / sinh(u) / sqrt(Q);
}
double dJRdLzHighStaeckelIntegrand(double chi,
			      void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u;
  double Q= chiQuStaeckel(chi,1,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  if ( Q <= 0. ) return 0.;
  return D / sinh(u) / sinh(u) / sqrt(Q);
}
// sqrt(2) dp_u/ddelta sqrt(S) at fixed (E, I3), S = p_u^2 / (2 delta^2):
// 2 S + Lz^2 / (delta^2 sinh^2 u) - delta dU/ddelta, with U(u) = cosh^2 u
// Phi(u,pi/2)
static inline double wdUStaeckel(double u,double S,
				 struct dJRStaeckelArg * params){
  double sh= sinh(u), ch= cosh(u);
  return 2. * S + 2. * params->Lz22delta / sh / sh
    + params->delta * ch * ch * sh
    * calcRforce(params->delta * sh,0.,0.,0.,
		 params->nargs,params->actionAngleArgs);
}
// The same for p_v: 2 S + Lz^2 / (delta^2 sin^2 v) + delta dV/ddelta
static inline double wdVStaeckel(double v,double S,
				 struct dJzStaeckelArg * params){
  double R, z;
  double delta= params->delta;
  double sh= sinh(params->u0), ch= cosh(params->u0);
  uv_to_Rz(params->u0,v,&R,&z,delta);
  return 2. * S + 2. * params->Lz22delta / sin(v) / sin(v)
    + delta * ( - ch * ch * sh
		* calcRforce(delta * sh,0.,0.,0.,
			     params->nargs,params->actionAngleArgs)
		+ ( sh * sh + sin(v) * sin(v) )
		* ( calcRforce(R,z,0.,0.,params->nargs,params->actionAngleArgs)
		    * sh * sin(v)
		    + calczforce(R,z,0.,0.,params->nargs,params->actionAngleArgs)
		    * ch * cos(v) ) );
}
double dJRdI3LowStaeckelIntegrand(double chi,
			      void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u;
  double Q= chiQuStaeckel(chi,0,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  double y= sin(0.5*chi) * sin(0.5*chi);
  if ( Q <= 0. ) return 0.;
  return ( params->wd ? wdUStaeckel(u,Q * y * ( 1. - y ),params) : 1. )
    * D / sqrt(Q);
}
double dJRdI3HighStaeckelIntegrand(double chi,
			      void * p){
  struct dJRStaeckelArg * params= (struct dJRStaeckelArg *) p;
  double u;
  double Q= chiQuStaeckel(chi,1,params->umin,params->umax,params->E,params->I3U,params->Lz22delta,
			  params->delta,params->v0,params->sin2v0,params->sinh2u0,
			  params->potu0v0,params->nargs,params->actionAngleArgs,&u);
  double D= params->umax - params->umin;
  double y= sin(0.5*chi) * sin(0.5*chi);
  if ( Q <= 0. ) return 0.;
  return ( params->wd ? wdUStaeckel(u,Q * y * ( 1. - y ),params) : 1. )
    * D / sqrt(Q);
}

double dJzdELowStaeckelIntegrand(double chi,
			      void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double v;
  double Q= chiQvStaeckel(chi,params->vmin,params->E,params->I3V,params->Lz22delta,
			  params->delta,params->u0,params->cosh2u0,params->sinh2u0,
			  params->potupi2,params->nargs,params->actionAngleArgs,&v);
  double D= M_PI - 2. * params->vmin;
  if ( Q <= 0. ) return 0.;
  return D * sin(v) * sin(v) / sqrt(Q);
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
double dJzdLzLowStaeckelIntegrand(double chi,
			      void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double v;
  double Q= chiQvStaeckel(chi,params->vmin,params->E,params->I3V,params->Lz22delta,
			  params->delta,params->u0,params->cosh2u0,params->sinh2u0,
			  params->potupi2,params->nargs,params->actionAngleArgs,&v);
  double D= M_PI - 2. * params->vmin;
  if ( Q <= 0. ) return 0.;
  return D / sin(v) / sin(v) / sqrt(Q);
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
// The derivative of V(v) = cosh^2 u0 Phi(u0,pi/2) - (sinh^2 u0 + sin^2 v)
// Phi(u0,v) with respect to the reference u0 along which it is built
static inline double dVdu0Staeckel(double v,struct dJzStaeckelArg * params){
  double R, z;
  double delta= params->delta;
  double sh= sinh(params->u0), ch= cosh(params->u0);
  double R0= delta * sh;
  uv_to_Rz(params->u0,v,&R,&z,delta);
  return 2. * sh * ch
    * ( evaluatePotentials(R0,0.,params->nargs,params->actionAngleArgs)
	- evaluatePotentials(R,z,params->nargs,params->actionAngleArgs) )
    - ch * ch * ch * delta
    * calcRforce(R0,0.,0.,0.,params->nargs,params->actionAngleArgs)
    + ( sh * sh + sin(v) * sin(v) ) * delta
    * ( calcRforce(R,z,0.,0.,params->nargs,params->actionAngleArgs)
	* ch * sin(v)
	+ calczforce(R,z,0.,0.,params->nargs,params->actionAngleArgs)
	* sh * cos(v) );
}
double dJzdI3LowStaeckelIntegrand(double chi,
			      void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double v;
  double Q= chiQvStaeckel(chi,params->vmin,params->E,params->I3V,params->Lz22delta,
			  params->delta,params->u0,params->cosh2u0,params->sinh2u0,
			  params->potupi2,params->nargs,params->actionAngleArgs,&v);
  double D= M_PI - 2. * params->vmin;
  double y= sin(0.5*chi) * sin(0.5*chi);
  if ( Q <= 0. ) return 0.;
  return ( params->wu0 == 1 ? dVdu0Staeckel(v,params)
	   : ( params->wu0 == 2 ? wdVStaeckel(v,Q * y * ( 1. - y ),params)
	       : 1. ) ) * D / sqrt(Q);
}
double dJzdI3HighStaeckelIntegrand(double t,
				   void * p){
  double v= M_PI/2. - t * t;
  return 2. * t * dJzdI3StaeckelIntegrand(v,p);
}
double dJzdI3StaeckelIntegrand(double v,
			       void * p){
  struct dJzStaeckelArg * params= (struct dJzStaeckelArg *) p;
  double out= JzStaeckelIntegrandSquared4dJz(v,p);
  if ( out <= 0. ) return 0.;
  else return ( params->wu0 == 1 ? dVdu0Staeckel(v,params)
		: ( params->wu0 == 2 ? wdVStaeckel(v,out,params) : 1. ) )
	 / sqrt(out);
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
