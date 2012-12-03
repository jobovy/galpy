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
      actionAngleArgs->potentialEval= &LogarithmicHaloPotentialEval;
      actionAngleArgs->nargs= 2;
      break;
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

