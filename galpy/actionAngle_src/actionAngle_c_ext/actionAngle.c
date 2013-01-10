#include <stdlib.h>
#include <galpy_potentials.h>
#include <actionAngle.h>
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
void parse_actionAngleArgs(int npot,
			   struct actionAngleArg * actionAngleArgs,
			   int * pot_type,
			   double * pot_args){
  int ii,jj;
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 3 arguments
      actionAngleArgs->potentialEval= &LogarithmicHaloPotentialEval;
      actionAngleArgs->nargs= 3;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      actionAngleArgs->potentialEval= &MiyamotoNagaiPotentialEval;
      actionAngleArgs->nargs= 3;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      actionAngleArgs->potentialEval= &PowerSphericalPotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      actionAngleArgs->potentialEval= &HernquistPotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      actionAngleArgs->potentialEval= &NFWPotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      actionAngleArgs->potentialEval= &JaffePotentialEval;
      actionAngleArgs->nargs= 2;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      actionAngleArgs->potentialEval= &DoubleExponentialDiskPotentialEval;
      //Look at pot_args to figure out the number of arguments
      actionAngleArgs->nargs= 8 + 2 * *(pot_args+5) + 4 * ( *(pot_args+4) + 1);
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      actionAngleArgs->potentialEval= &FlattenedPowerPotentialEval;
      actionAngleArgs->nargs= 4;
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
