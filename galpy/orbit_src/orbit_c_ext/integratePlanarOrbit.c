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
  int npot= 0;
  bool lp= (bool) lp;
  if ( lp ) npot++;
  struct leapFuncArg * leapFuncArgs= (struct leapFuncArg *) malloc ( npot * sizeof (struct  leapFuncArg) );
  if ( lp ){
    leapFuncArgs->
  }
}
