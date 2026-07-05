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
				       double *,int,int *,double *,tfuncs_type_arr,double,
				       double *,double *,double *,int *);
EXPORT void actionAngleAdiabatic_actions(int,double *,double *,double *,double *,
				 double *,int,int *,double *,tfuncs_type_arr,double,
				 double *,double *,int *);
// C-native differentiable actions: (jr,jz) + the fused (ndata,2,5) Jacobian
// d(jr,jz)/d(R,vR,vT,z,vz), assembled analytically (#131 Adiabatic PR-2a).
EXPORT void actionAngleAdiabatic_actionsJac(int,double *,double *,double *,double *,
				 double *,int,int *,double *,tfuncs_type_arr,double,int,
				 double *,double *,double *,int *);
void calcdJRAdiabatic(int,double *,double *,double *,double *,double *,double *,
		      int,struct potentialArg *,int);
void calcdJzAdiabatic(int,double *,double *,double *,double *,double *,
		      int,struct potentialArg *,int);
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
               tfuncs_type_arr pot_tfuncs,
				       double gamma,
				       double *rperi,
				       double *rap,
				       double *zmax,
				       int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
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
          tfuncs_type_arr pot_tfuncs,
				  double gamma,
				  double *jr,
				  double *jz,
				  int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
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
// C-native differentiable actions: forward (jr,jz) via the existing self-contained
// quadratures, plus the fused (ndata,2,5) Jacobian d(jr,jz)/d(R,vR,vT,z,vz).
// Adiabatic is separable EXCEPT the vertical action is injected into the radial
// Lz (Lz = |R vT| + gamma*Jz; gamma=1 by default), so the vertical block
// (Ez,zmax,Jz + its derivatives) is Lz-independent and computed first, then fed
// into the radial Lz/E_radial. Boundary terms vanish (p=0 at every turning point)
// so each dJ/dparam is a pure interior integral (Leibniz). First-order only.
void actionAngleAdiabatic_actionsJac(int ndata,
				     double *R,
				     double *vR,
				     double *vT,
				     double *z,
				     double *vz,
				     int npot,
				     int * pot_type,
				     double * pot_args,
				     tfuncs_type_arr pot_tfuncs,
				     double gamma,
				     int order,
				     double *jr,
				     double *jz,
				     double *jac,
				     int * err){
  int ii;
  struct potentialArg * actionAngleArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,actionAngleArgs,&pot_type,&pot_args,&pot_tfuncs);
  double *ER= (double *) malloc ( ndata * sizeof(double) );
  double *Ez= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  double *rperi= (double *) malloc ( ndata * sizeof(double) );
  double *rap= (double *) malloc ( ndata * sizeof(double) );
  double *zmax= (double *) malloc ( ndata * sizeof(double) );
  // vertical block first (Lz-independent)
  calcEREzL(ndata,R,vR,vT,z,vz,ER,Ez,Lz,npot,actionAngleArgs);
  calcZmax(ndata,zmax,z,R,Ez,npot,actionAngleArgs);
  calcJzAdiabatic(ndata,jz,zmax,R,Ez,npot,actionAngleArgs,order);
  double *djzdEz= (double *) malloc ( ndata * sizeof(double) );
  double *djzdR=  (double *) malloc ( ndata * sizeof(double) );
  calcdJzAdiabatic(ndata,djzdEz,djzdR,zmax,R,Ez,npot,actionAngleArgs,order);
  // gamma injection: Lz = |R vT| + gamma*Jz, then radial effective energy
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii)
  for (ii=0; ii < ndata; ii++){
    *(Lz+ii)= fabs( *(Lz+ii) ) + gamma * *(jz+ii);
    *(ER+ii)+= 0.5 * *(Lz+ii) * *(Lz+ii) / *(R+ii) / *(R+ii)
      - 0.5 * *(vT+ii) * *(vT+ii);
  }
  // radial block (uses the gamma-adjusted ER,Lz)
  calcRapRperi(ndata,rperi,rap,R,ER,Lz,npot,actionAngleArgs);
  calcJRAdiabatic(ndata,jr,rperi,rap,ER,Lz,npot,actionAngleArgs,order);
  double *djrdER= (double *) malloc ( ndata * sizeof(double) );
  double *djrdLz= (double *) malloc ( ndata * sizeof(double) );
  calcdJRAdiabatic(ndata,djrdER,djrdLz,rperi,rap,ER,Lz,npot,actionAngleArgs,order);
  // assemble the (2,5) Jacobian per orbit from the elementary chains
#pragma omp parallel for schedule(static,chunk) private(ii)
  for (ii=0; ii < ndata; ii++){
    int kk;
    double tR= *(R+ii), tvR= *(vR+ii), tvT= *(vT+ii), tz= *(z+ii), tvz= *(vz+ii);
    if ( *(rperi+ii) == -9999.99 || *(rap+ii) == -9999.99
	 || *(zmax+ii) == -9999.99 ){
      for (kk=0;kk<10;kk++) *(jac+ii*10+kk)= 0.;
      continue;
    }
    double tLz= *(Lz+ii), tER= *(ER+ii);
    double s= ( tR * tvT >= 0. ) ? 1. : -1.;  // sign(R vT); Lz used fabs(R vT)
    // forces for the elementary Ez-chain (at the INITIAL z)
    double FR_R0= calcRforce(tR,0.,0.,0.,npot,actionAngleArgs);
    double FR_Rz= calcRforce(tR,tz,0.,0.,npot,actionAngleArgs);
    double Fz_Rz= calczforce(tR,tz,0.,0.,npot,actionAngleArgs);
    double dEz[5]= { FR_R0 - FR_Rz, 0., 0., -Fz_Rz, tvz };
    // dJz/dcoord = dJz/dEz * dEz/dcoord + dJz/dR|_Ez * e_R
    double bE= *(djzdEz+ii), bR= *(djzdR+ii);
    double dJz[5];
    for (kk=0;kk<5;kk++) dJz[kk]= bE*dEz[kk];
    dJz[0]+= bR;
    // dLz/dcoord = [s vT + gamma dJz/dR, 0, s R, gamma dJz/dz, gamma dJz/dvz]
    double dLz[5]= { s*tvT + gamma*dJz[0], 0., s*tR, gamma*dJz[3], gamma*dJz[4] };
    // dE_radial/dcoord (E_R = Phi(R,0) + vR^2/2 + Lz^2/(2R^2))
    double LzR2= tLz/(tR*tR);
    double dER[5];
    dER[0]= -FR_R0 - tLz*tLz/(tR*tR*tR) + LzR2*dLz[0];
    dER[1]= tvR;
    dER[2]= LzR2*dLz[2];
    dER[3]= LzR2*dLz[3];
    dER[4]= LzR2*dLz[4];
    // dJr/dcoord = dJr/dER * dER/dcoord + dJr/dLz * dLz/dcoord
    double aE= *(djrdER+ii), aL= *(djrdLz+ii);
    for (kk=0;kk<5;kk++){
      *(jac+ii*10+0*5+kk)= aE*dER[kk] + aL*dLz[kk];  // jr row
      *(jac+ii*10+1*5+kk)= dJz[kk];                  // jz row
    }
    (void) tER;
  }
  free_potentialArgs(npot,actionAngleArgs);
  free(actionAngleArgs);
  free(ER); free(Ez); free(Lz);
  free(rperi); free(rap); free(zmax);
  free(djzdEz); free(djzdR); free(djrdER); free(djrdLz);
}
// dJr/dE_radial = (1/(sqrt2 pi)) int_rperi^Rap dr/sqrt(F_R),
// dJr/dLz       = -(Lz/(sqrt2 pi)) int_rperi^Rap dr/(r^2 sqrt(F_R)),
// F_R(r)= E_radial - Phi(r,0) - Lz^2/(2 r^2). Both 1/sqrt-singular at BOTH turning
// points -> the theta-substitution r = cc - rr*cos(theta), theta in [0,pi]
// (dr = rr sin(theta) dtheta) regularizes both ends in one GL pass. Degenerate
// (unbound sentinel / circular) -> 0.
void calcdJRAdiabatic(int ndata,
		      double * djrdER,
		      double * djrdLz,
		      double * rperi,
		      double * rap,
		      double * ER,
		      double * Lz,
		      int nargs,
		      struct potentialArg * actionAngleArgs,
		      int order){
  int ii, gi;
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,gi) \
  shared(djrdER,djrdLz,rperi,rap,ER,Lz,T)
  for (ii=0; ii < ndata; ii++){
    if ( *(rperi+ii) == -9999.99 || *(rap+ii) == -9999.99 ){
      *(djrdER+ii)= 0.; *(djrdLz+ii)= 0.; continue;
    }
    if ( ( *(rap+ii) - *(rperi+ii) ) / *(rap+ii) < 0.000001 ){ //circular
      *(djrdER+ii)= 0.; *(djrdLz+ii)= 0.; continue;
    }
    double tER= *(ER+ii), tLz= *(Lz+ii);
    double Lz22= 0.5*tLz*tLz;
    double cc= 0.5*( *(rap+ii) + *(rperi+ii) );
    double rr= 0.5*( *(rap+ii) - *(rperi+ii) );
    double accE= 0., accL= 0., xi, wi;
    for (gi=0; gi < order; gi++){
      gsl_integration_glfixed_point(0.,M_PI,gi,&xi,&wi,T);
      double sinth= sin(xi), costh= cos(xi);
      double r= cc - rr*costh;
      double FR= tER - evaluatePotentials(r,0.,nargs,actionAngleArgs)
	- Lz22/(r*r);
      if ( FR <= 0. ) continue;
      double w= wi*rr*sinth/sqrt(FR);
      accE+= w;
      accL+= w/(r*r);
    }
    *(djrdER+ii)= accE / ( sqrt(2.)*M_PI );
    *(djrdLz+ii)= -tLz*accL / ( sqrt(2.)*M_PI );
  }
  gsl_integration_glfixed_table_free ( T );
}
// dJz/dEz    = (sqrt2/pi) int_0^zmax dz/sqrt(F_z),
// dJz/dR|_Ez = (sqrt2/pi) int_0^zmax [Rforce(R,z)-Rforce(R,0)]/sqrt(F_z) dz,
// F_z(z)= Ez - [Phi(R,z)-Phi(R,0)]. 1/sqrt-singular only at the upper end zmax
// (F_z(0)=Ez>0) -> z = zmax*sin(phi), phi in [0,pi/2] (dz = zmax cos(phi) dphi)
// regularizes zmax. Degenerate (unbound sentinel / planar) -> 0.
void calcdJzAdiabatic(int ndata,
		      double * djzdEz,
		      double * djzdR,
		      double * zmax,
		      double * R,
		      double * Ez,
		      int nargs,
		      struct potentialArg * actionAngleArgs,
		      int order){
  int ii, gi;
  gsl_integration_glfixed_table * T= gsl_integration_glfixed_table_alloc (order);
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,gi) \
  shared(djzdEz,djzdR,zmax,R,Ez,T)
  for (ii=0; ii < ndata; ii++){
    if ( *(zmax+ii) == -9999.99 ){ *(djzdEz+ii)= 0.; *(djzdR+ii)= 0.; continue; }
    if ( *(zmax+ii) < 0.000001 ){ //planar (J_z=0)
      *(djzdEz+ii)= 0.; *(djzdR+ii)= 0.; continue;
    }
    double tR= *(R+ii), tEz= *(Ez+ii), tzmax= *(zmax+ii);
    double FR_R0= calcRforce(tR,0.,0.,0.,nargs,actionAngleArgs);
    double accEz= 0., accR= 0., xi, wi;
    for (gi=0; gi < order; gi++){
      gsl_integration_glfixed_point(0.,0.5*M_PI,gi,&xi,&wi,T);
      double sph= sin(xi), cph= cos(xi);
      double zz= tzmax*sph;
      double Fz= tEz - evaluateVerticalPotentials(tR,zz,nargs,actionAngleArgs);
      if ( Fz <= 0. ) continue;
      double w= wi*tzmax*cph/sqrt(Fz);  // dz/dphi = zmax cos(phi)
      accEz+= w;
      double FR_Rz= calcRforce(tR,zz,0.,0.,nargs,actionAngleArgs);
      accR+= w*(FR_Rz - FR_R0);
    }
    *(djzdEz+ii)= sqrt(2.)/M_PI * accEz;
    *(djzdR+ii)=  sqrt(2.)/M_PI * accR;
  }
  gsl_integration_glfixed_table_free ( T );
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
    peps= GSL_FN_EVAL(JRRoot+tid,*(R+ii)+0.0000001);
    meps= GSL_FN_EVAL(JRRoot+tid,*(R+ii)-0.0000001);
    if ( fabs(GSL_FN_EVAL(JRRoot+tid,*(R+ii))) < 0.0000001 && peps*meps < 0 ){ //we are at rap or rperi
      if ( peps < 0. && meps > 0. ) {//rap
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
      else {// JB: Should catch all: if ( peps > 0. && meps < 0. ){//rperi
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
    else if ( fabs(peps) < 0.00000001 && fabs(meps) < 0.00000001 && peps <= 0 && meps <= 0 ) {//circular
      *(rperi+ii) = *(R+ii);
      *(rap+ii) = *(R+ii);
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
