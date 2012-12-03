/*
  C code for Binney (2012)'s Staeckel approximation code
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/*
  Function Declarations
*/
void actionAngleStaeckel_actions();
/*
  Actual functions
*/
inline void parse_actionAngleArgs(int npot,
				  struct actionAngleArg * actionAngleArgs,
				  int * pot_type,
				  double * pot_args){
  int ii,jj;
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 2 arguments
      leapFuncArgs->Rforce= &LogarithmicHaloPotentialRforce;
      leapFuncArgs->zforce= &LogarithmicHaloPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      leapFuncArgs->Rforce= &MiyamotoNagaiPotentialRforce;
      leapFuncArgs->zforce= &MiyamotoNagaiPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      leapFuncArgs->nargs= 3;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      leapFuncArgs->Rforce= &PowerSphericalPotentialRforce;
      leapFuncArgs->zforce= &PowerSphericalPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      leapFuncArgs->Rforce= &HernquistPotentialRforce;
      leapFuncArgs->zforce= &HernquistPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      leapFuncArgs->Rforce= &NFWPotentialRforce;
      leapFuncArgs->zforce= &NFWPotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      leapFuncArgs->Rforce= &JaffePotentialRforce;
      leapFuncArgs->zforce= &JaffePotentialzforce;
      leapFuncArgs->phiforce= &ZeroForce;
      leapFuncArgs->nargs= 2;
      break;
    }
    leapFuncArgs->args= (double *) malloc( leapFuncArgs->nargs * sizeof(double));
    for (jj=0; jj < leapFuncArgs->nargs; jj++){
      *(leapFuncArgs->args)= *pot_args++;
      leapFuncArgs->args++;
    }
    leapFuncArgs->args-= leapFuncArgs->nargs;
    leapFuncArgs++;
  }
  leapFuncArgs-= npot;
}
