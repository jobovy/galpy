#include <stdlib.h>
#include <galpy_potentials.h>
#include <actionAngle.h>
#include <cubic_bspline_2d_coeffs.h>
double evaluatePotentials(double R, double Z, 
			  int nargs, struct potentialArg * potentialArgs){
  int ii;
  double pot= 0.;
  for (ii=0; ii < nargs; ii++){
    pot+= potentialArgs->potentialEval(R,Z,0.,0.,
				       potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= nargs;
  return pot;
}
void parse_actionAngleArgs(int npot,
			   struct potentialArg * potentialArgs,
			   int * pot_type,
			   double * pot_args){
  int ii,jj,kk,ll;
  int nR, nz;
  double * Rgrid, * zgrid, * potGrid_splinecoeffs, * row;
  for (ii=0; ii < npot; ii++){
    switch ( *pot_type++ ) {
    case 0: //LogarithmicHaloPotential, 3 arguments
      potentialArgs->potentialEval= &LogarithmicHaloPotentialEval;
      potentialArgs->nargs= 3;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->potentialEval= &MiyamotoNagaiPotentialEval;
      potentialArgs->nargs= 3;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialEval;
      potentialArgs->nargs= 2;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;
    case 8: //HernquistPotential, 2 arguments
      potentialArgs->potentialEval= &HernquistPotentialEval;
      potentialArgs->nargs= 2;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->potentialEval= &NFWPotentialEval;
      potentialArgs->nargs= 2;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;
    case 10: //JaffePotential, 2 arguments
      potentialArgs->potentialEval= &JaffePotentialEval;
      potentialArgs->nargs= 2;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      potentialArgs->potentialEval= &DoubleExponentialDiskPotentialEval;
      //Look at pot_args to figure out the number of arguments
      potentialArgs->nargs= (int) (8 + 2 * *(pot_args+5) + 4 * ( *(pot_args+4) + 1));
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      potentialArgs->potentialEval= &FlattenedPowerPotentialEval;
      potentialArgs->nargs= 4;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;     
    case 13: //interpRZPotential, XX arguments
      //Grab the grids and the coefficients
      nR= (int) *pot_args++;
      nz= (int) *pot_args++;
      Rgrid= (double *) malloc ( nR * sizeof ( double ) );
      zgrid= (double *) malloc ( nz * sizeof ( double ) );
      row= (double *) malloc ( nz * sizeof ( double ) );
      potGrid_splinecoeffs= (double *) malloc ( nR * nz * sizeof ( double ) );
      for (kk=0; kk < nR; kk++)
	*(Rgrid+kk)= *pot_args++;
      for (kk=0; kk < nz; kk++)
	*(zgrid+kk)= *pot_args++;
      for (kk=0; kk < nR; kk++){
	for (ll=0; ll < nz; ll++)
	  *(row+ll)= *pot_args++;
	put_row(potGrid_splinecoeffs,kk,row,nz); 
      }
      potentialArgs->i2d= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2d,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->acc= gsl_interp_accel_alloc ();
      potentialArgs->potentialEval= &interpRZPotentialEval;
      potentialArgs->nargs= 2;
      //clean up
      free(Rgrid);
      free(zgrid);
      free(row);
      free(potGrid_splinecoeffs);
      break;
    case 14: //IsochronePotential, 2 arguments
      potentialArgs->potentialEval= &IsochronePotentialEval;
      potentialArgs->nargs= 2;
      potentialArgs->i2d= NULL;
      potentialArgs->acc= NULL;
      break;     
    }
    potentialArgs->args= (double *) malloc( potentialArgs->nargs * sizeof(double));
    for (jj=0; jj < potentialArgs->nargs; jj++){
      *(potentialArgs->args)= *pot_args++;
      potentialArgs->args++;
    }
    potentialArgs->args-= potentialArgs->nargs;
    potentialArgs++;
  }
  potentialArgs-= npot;
}
