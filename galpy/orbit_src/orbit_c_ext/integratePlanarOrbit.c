/*
  Wrappers around the C integration code for planar Orbits
*/
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
void evalPlanarRectForce(int, double, double *, double *,
			 int, struct leapFuncArg *);
double calcPlanarRforce(int,double, double, double, 
			int, struct leapFuncArg *);
double calcPlanarphiforce(int,double, double, double, 
			int, struct leapFuncArg *);
/*
  Actual functions
*/
void integratePlanarOrbit(int dim,
			  double *yo,
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
  struct leapFuncArg * leapFuncArgs= (struct leapFuncArg *) malloc ( npot * sizeof (struct  leapFuncArg) );
  //LogarithmicHaloPotential
  if ( lp ){
    leapFuncArgs->Rforce= &LogarithmicHaloPotentialPlanarRforce;
    //phiforce needs to be set to zero somehow
    leapFuncArgs->nargs= 2;
    for (ii=0; ii < leapFuncArgs->nargs; ii++)
      *(leapFuncArgs->args)++= *lpargs++;
    leapFuncArgs->args-= leapFuncArgs->nargs;
    lpargs-= leapFuncArgs->nargs;
  }
  //Integrate
  leapfrog(evalPlanarRectForce,dim,yo,nt,t,npot,leapFuncArgs,rtol,atol,result);
  //Done!
}

void evalPlanarRectForce(int dim, double t, double *q, double *a,
			 int nargs, struct leapFuncArg * leapFuncArgs,
			 double * result){
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
  Rforce= calcPlanarRforce(dim,R,phi,t,nargs,leapFuncArgs);
  phiforce= calcPlanarphiforce(dim,R,phi,t,nargs,leapFuncArgs);
  *result++= cosphi*Rforce-1./R*sinphi*phiforce;
  *result--= sinphi*Rforce+1./R*cosphi*phiforce;
}

double calcPlanarRforce(int dim,double R, double phi, double t, 
			int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double Rforce= 0.;
  for (ii=0; ii < nargs, ii++){
    Rforce+= leapFuncArgs->Rforce(R,0.,phi,
				  leapFuncArgs->nargs,
				  leapFuncArgs->args,
				  dim)
      leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
}
double calcPlanarphiforce(int dim,double R, double phi, double t, 
			  int nargs, struct leapFuncArg * leapFuncArgs){
  int ii;
  double phiforce= 0.;
  for (ii=0; ii < nargs, ii++){
    phiforce+= leapFuncArgs->phiforce(R,0.,phi,
				      leapFuncArgs->nargs,
				      leapFuncArgs->args,
				      dim)
      leapFuncArgs++;
  }
  leapFuncArgs-= nargs;
}
