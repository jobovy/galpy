/*
  Wrappers around the C integration code for planar Orbits
*/
#include <stdbool.h>
#include <bovy_symplecticode.h>
//Potentials
#include <galpy_potentials.h>
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
  if ( lp ){
    leapFuncArgs->Rforce= &LogarithmicHaloPotentialRforce;
    leapFuncArgs->zforce= &LogarithmicHaloPotentialzforce;
    //phiforce needs to be set to zero somehow
    leapFuncArgs->nargs= 2;
    for (ii=0; ii < leapFuncArgs->nargs; ii++)
      *(leapFuncArgs->args)++= *lpargs++;
    leapFuncArgs->args-= leapFuncArgs->nargs;
    lpargs-= leapFuncArgs->nargs;
  }
}
