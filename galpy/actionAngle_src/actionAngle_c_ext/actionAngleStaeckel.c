/*
  C code for Binney (2012)'s Staeckel approximation code
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
//Potentials
#include <galpy_potentials.h>
#include <actionAngle.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
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
  struct actionAngleArg * actionAngleArgs;
};
/*
  Function Declarations
*/
void actionAngleStaeckel_actions(int,double *,double *,double *,double *,
				 double *,int,int *,double *,double,
				 double *,double *,int *);
void calcUminUmax(int,double *,double *,double *,double *,double *,double *,
		  double,double *,double *,double *,double *,double *,int,
		  struct actionAngleArg *);
double JRStaeckelIntegrandSquared(double,void *);
double JRStaeckelIntegrand(double,void *);
double evaluatePotentials(double,double,int, struct actionAngleArg *);
double evaluatePotentialsUV(double,double,double,int,struct actionAngleArg *);
/*
  Actual functions, inlines first
*/
inline void parse_actionAngleArgs(int npot,
				  struct actionAngleArg * actionAngleArgs,
				  int * pot_type,
				  double * pot_args){
  int ii,jj;
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 2 arguments
      actionAngleArgs->potentialEval= &LogarithmicHaloPotentialEval;
      actionAngleArgs->nargs= 3;
      break;
    /*
    case 5: //MiyamotoNagaiPotential, 3 arguments
      actionAngleArgs->Rforce= &MiyamotoNagaiPotentialEval;
      actionAngleArgs->nargs= 3;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      actionAngleArgs->Rforce= &PowerSphericalPotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      actionAngleArgs->Rforce= &HernquistPotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      actionAngleArgs->Rforce= &NFWPotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      actionAngleArgs->Rforce= &JaffePotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    */
    }
    actionAngleArgs->args= (double *) malloc( actionAngleArgs->nargs * sizeof(double));
    for (jj=0; jj < actionAngleArgs->nargs; jj++){
      *(actionAngleArgs->args)= *pot_args++;
      actionAngleArgs->args++;
    }
    actionAngleArgs->args-= actionAngleArgs->nargs;
    actionAngleArgs++;
  }
  actionAngleArgs-= npot;
}
inline void uv_to_Rz(double u, double v, double * R, double *z,double delta){
  *R= delta * sinh(u) * sin(v);
  *z= delta * cosh(u) * cos(v);
}
inline void Rz_to_uv_vec(int ndata,
			 double *R,
			 double *z,
			 double *u,
			 double *v,
			 double delta){
  int ii;
  double d12, d22, coshu, cosv;
  for (ii=0; ii < ndata; ii++) {
    d12= (*(z+ii)+delta)*(*(z+ii)+delta)+(*(R+ii))*(*(R+ii));
    d22= (*(z+ii)-delta)*(*(z+ii)-delta)+(*(R+ii))*(*(R+ii));
    coshu= 0.5/delta*(sqrt(d12)+sqrt(d22));
    cosv=  0.5/delta*(sqrt(d12)-sqrt(d22));
    *u++= acosh(coshu);
    *v++= acos(cosv);
  }
  u-= ndata;
  v-= ndata;
}
inline void calcEL(int ndata,
		   double *R,
		   double *vR,
		   double *vT,
		   double *z,
		   double *vz,
		   double *E,
		   double *Lz,
		   int nargs,
		   struct actionAngleArg * actionAngleArgs){
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
  MAIN FUNCTION
 */
void actionAngleStaeckel_actions(int ndata,
				 double *R,
				 double *vR,
				 double *vT,
				 double *z,
				 double *vz,
				 int npot,
				 int * pot_type,
				 double * pot_args,
				 double delta,
				 double *jr,
				 double *jz,
				 int * err){
  int ii;
  //Set up the potentials
  struct actionAngleArg * actionAngleArgs= (struct actionAngleArg *) malloc ( npot * sizeof (struct actionAngleArg) );
  parse_actionAngleArgs(npot,actionAngleArgs,pot_type,pot_args);
  //E,Lz
  double *E= (double *) malloc ( ndata * sizeof(double) );
  double *Lz= (double *) malloc ( ndata * sizeof(double) );
  calcEL(ndata,R,vR,vT,z,vz,E,Lz,npot,actionAngleArgs);
  //Calculate all necessary parameters
  double *ux= (double *) malloc ( ndata * sizeof(double) );
  double *vx= (double *) malloc ( ndata * sizeof(double) );
  Rz_to_uv_vec(ndata,R,z,ux,vx,delta);
  double *coshux= (double *) malloc ( ndata * sizeof(double) );
  double *sinhux= (double *) malloc ( ndata * sizeof(double) );
  double *sinvx= (double *) malloc ( ndata * sizeof(double) );
  double *cosvx= (double *) malloc ( ndata * sizeof(double) );
  double *pux= (double *) malloc ( ndata * sizeof(double) );
  double *pvx= (double *) malloc ( ndata * sizeof(double) );
  double *u0= (double *) malloc ( ndata * sizeof(double) );
  double *sinh2u0= (double *) malloc ( ndata * sizeof(double) );
  double *v0= (double *) malloc ( ndata * sizeof(double) );
  double *sin2v0= (double *) malloc ( ndata * sizeof(double) );
  double *potu0v0= (double *) malloc ( ndata * sizeof(double) );
  double *potupi2= (double *) malloc ( ndata * sizeof(double) );
  double *I3U= (double *) malloc ( ndata * sizeof(double) );
  double *I3V= (double *) malloc ( ndata * sizeof(double) );
  for (ii=0; ii < ndata; ii++){
    *(coshux+ii)= cosh(*(ux+ii));
    *(sinhux+ii)= sinh(*(ux+ii));
    *(cosvx+ii)= cos(*(vx+ii));
    *(sinvx+ii)= sin(*(vx+ii));
    *(pux+ii)= delta * (*(vR+ii) * *(coshux+ii) * *(sinvx+ii) 
			+ *(vz+ii) * *(sinhux+ii) * *(cosvx+ii));
    *(pvx+ii)= delta * (*(vR+ii) * *(sinhux+ii) * *(cosvx+ii) 
			- *(vz+ii) * *(coshux+ii) * *(sinvx+ii));
    *(u0+ii)= *(ux+ii);
    *(sinh2u0+ii)= sinh(*(u0+ii)) * sinh(*(u0+ii));
    *(v0+ii)= *(vx+ii);
    *(sin2v0+ii)= sin(*(v0+ii)) * sin(*(v0+ii));
    *(potu0v0+ii)= evaluatePotentialsUV(*(u0+ii),*(vx+ii),delta,
					npot,actionAngleArgs);
    *(I3U+ii)= *(E+ii) * *(sinhux+ii) * *(sinhux+ii)
      - 0.5 * *(pux+ii) * *(pux+ii) / delta / delta
      - 0.5 * *(Lz+ii) * *(Lz+ii) / delta / delta / *(sinhux+ii) / *(sinhux+ii);
    *(potupi2+ii)= evaluatePotentialsUV(*(ux+ii),0.5 * M_PI,delta,
					npot,actionAngleArgs);
    *(I3V+ii)= - *(E+ii) * *(sinvx+ii) * *(sinvx+ii)
      + 0.5 * *(pvx+ii) * *(pvx+ii) / delta / delta
      + 0.5 * *(Lz+ii) * *(Lz+ii) / delta / delta / *(sinvx+ii) / *(sinvx+ii)
      - *(coshux+ii) * *(coshux+ii) * *(potupi2+ii)
      + ( *(sinhux+ii) * *(sinhux+ii) + *(sinvx+ii) * *(sinvx+ii))
      * evaluatePotentialsUV(*(ux+ii),*(vx+ii),delta,
			     npot,actionAngleArgs);
  }
  double *umin= (double *) malloc ( ndata * sizeof(double) );
  double *umax= (double *) malloc ( ndata * sizeof(double) );
  calcUminUmax(ndata,umin,umax,ux,E,Lz,I3U,delta,u0,sinh2u0,v0,sin2v0,potu0v0,
	       npot,actionAngleArgs);
}
void calcUminUmax(int ndata,
		  double * umin,
		  double * umax,
		  double * ux,
		  double * E,
		  double * Lz,
		  double * I3U,
		  double delta,
		  double * u0,
		  double * sinh2u0,
		  double * v0,
		  double * sin2v0,
		  double * potu0v0,
		  int nargs,
		  struct actionAngleArg * actionAngleArgs){
  int ii;
  gsl_function JRRoot;
  struct JRStaeckelArg * params= (struct JRStaeckelArg *) malloc ( sizeof (struct JRStaeckelArg) );
  params->delta= delta;
  params->nargs= nargs;
  params->actionAngleArgs= actionAngleArgs;
  //Setup solver
  int status;
  int iter, max_iter = 100;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *s;
  double u_lo, u_hi;
  T = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc (T);
  JRRoot.function = &JRStaeckelIntegrandSquared;
  for (ii=0; ii < ndata; ii++){
    //Setup function
    params->E= *(E+ii);
    params->Lz22delta= 0.5 * *(Lz+ii) * *(Lz+ii) / delta / delta;
    params->I3U= *(I3U+ii);
    params->u0= *(u0+ii);
    params->sinh2u0= *(sinh2u0+ii);
    params->v0= *(v0+ii);
    params->sin2v0= *(sin2v0+ii);
    params->potu0v0= *(potu0v0+ii);
    JRRoot.params = params;
    //Find starting points for minimum
    u_lo= 0.01;
    u_hi= *(ux+ii);
    //Find root
    gsl_root_fsolver_set (s, &JRRoot, u_lo, u_hi);
    //printf("Here %i\n",ii);
    //fflush(stdout);
    iter= 0;
    do
      {
	iter++;
	status = gsl_root_fsolver_iterate (s);
	u_lo = gsl_root_fsolver_x_lower (s);
	u_hi = gsl_root_fsolver_x_upper (s);
	status = gsl_root_test_interval (u_lo, u_hi,
					 9.9999999999999998e-13,
					 4.4408920985006262e-16);
      }
    while (status == GSL_CONTINUE && iter < max_iter);
    *(umin+ii) = gsl_root_fsolver_root (s);
    //Find starting points for maximum
    u_lo= *(ux+ii);
    u_hi=10.;
    //Find root
    gsl_root_fsolver_set (s, &JRRoot, u_lo, u_hi);
    iter= 0;
    do
      {
	iter++;
	status = gsl_root_fsolver_iterate (s);
	u_lo = gsl_root_fsolver_x_lower (s);
	u_hi = gsl_root_fsolver_x_upper (s);
	status = gsl_root_test_interval (u_lo, u_hi,
					 9.9999999999999998e-13,
					 4.4408920985006262e-16);
      }
    while (status == GSL_CONTINUE && iter < max_iter);
    *(umax+ii) = gsl_root_fsolver_root (s);
  }
 gsl_root_fsolver_free (s);    
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
  
double evaluatePotentials(double R, double Z, 
			  int nargs, struct actionAngleArg * actionAngleArgs){
  int ii;
  double pot= 0.;
  for (ii=0; ii < nargs; ii++){
    pot+= actionAngleArgs->potentialEval(R,Z,0.,0.,
					 actionAngleArgs->nargs,
					 actionAngleArgs->args);
    actionAngleArgs++;
  }
  actionAngleArgs-= nargs;
  return pot;
}
double evaluatePotentialsUV(double u, double v, double delta,
			    int nargs, 
			    struct actionAngleArg * actionAngleArgs){
  double R,z;
  uv_to_Rz(u,v,&R,&z,delta);
  return evaluatePotentials(R,z,nargs,actionAngleArgs);
}
