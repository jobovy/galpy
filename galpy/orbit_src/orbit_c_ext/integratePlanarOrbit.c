/*
  Wrappers around the C integration code for planar Orbits
*/
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <bovy_symplecticode.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
  Function Declarations
*/
void evalPlanarRectForce(double, double *, double *,
			 int, struct leapFuncArg *);
double calcPlanarRforce(double, double, double, 
			int, struct leapFuncArg *);
double calcPlanarphiforce(double, double, double, 
			int, struct leapFuncArg *);
/*
  Actual functions
*/
void integratePlanarOrbit(double *yo,
			  int nt, 
			  double *t,
			  char logp, //LogarithmicHaloPotential?
			  int nlpargs,
			  double * lpargs,
			  double rtol,
			  double atol,
			  double *result){
  //Set up the forces, first count
  int ii;
  int npot= 0;
  bool lp= (bool) lp;
  if ( lp ) npot++;
  struct leapFuncArg * leapFuncArgs= (struct leapFuncArg *) malloc ( npot * sizeof (struct leapFuncArg) );
  //LogarithmicHaloPotential
  if ( lp ){
    leapFuncArgs->planarRforce= &LogarithmicHaloPotentialPlanarRforce;
    leapFuncArgs->planarphiforce= &ZeroPlanarForce;
    leapFuncArgs->nargs= 2;
    for (ii=0; ii < leapFuncArgs->nargs; ii++)
      *(leapFuncArgs->args)++= *lpargs++;
    leapFuncArgs->args-= leapFuncArgs->nargs;
    lpargs-= leapFuncArgs->nargs;
  }
  //Integrate
  leapfrog(&evalPlanarRectForce,2,yo,nt,t,npot,leapFuncArgs,rtol,atol,result);
  //Done!
}

void evalPlanarRectForce(double t, double *q, double *a,
			 int nargs, struct leapFuncArg * leapFuncArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phiforce;
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= phi+2.*M_PI;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,leapFuncArgs);
  phiforce= calcPlanarphiforce(R,phi,t,nargs,leapFuncArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phiforce;
  *a--= sinphi*Rforce+1./R*cosphi*phiforce;
}

double calcPlanarRforce(double R, double phi, double t, 
			int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs; ii++){
    Rforce+= leapFuncArgs->planarRforce(R,phi,
					leapFuncArgs->nargs,
					leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return Rforce;
}
double calcPlanarphiforce(double R, double phi, double t, 
			  int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double phiforce= 0.;
  for (ii=0; ii < nargs; ii++){
    phiforce+= leapFuncArgs->planarphiforce(R,phi,
					    leapFuncArgs->nargs,
					    leapFuncArgs->args);
    leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
  return phiforce;
}
