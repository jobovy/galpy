#include <stdlib.h>
#include <stdbool.h>
#include <galpy_potentials.h>
#include <actionAngle.h>
#include <cubic_bspline_2d_coeffs.h>
void parse_actionAngleArgs(int npot,
			   struct potentialArg * potentialArgs,
			   int ** pot_type,
			   double ** pot_args,
			   bool forTorus){
  int ii,jj,kk;
  int nR, nz;
  double * Rgrid, * zgrid, * potGrid_splinecoeffs;
  init_potentialArgs(npot,potentialArgs);
  for (ii=0; ii < npot; ii++){
    switch ( *(*pot_type)++ ) {
    case 0: //LogarithmicHaloPotential, 4 arguments
      potentialArgs->potentialEval= &LogarithmicHaloPotentialEval;
      potentialArgs->Rforce= &LogarithmicHaloPotentialRforce;
      potentialArgs->zforce= &LogarithmicHaloPotentialzforce;
      potentialArgs->nargs= 4;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->potentialEval= &MiyamotoNagaiPotentialEval;
      potentialArgs->Rforce= &MiyamotoNagaiPotentialRforce;
      potentialArgs->zforce= &MiyamotoNagaiPotentialzforce;
      potentialArgs->nargs= 3;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialEval;
      potentialArgs->Rforce= &PowerSphericalPotentialRforce;
      potentialArgs->zforce= &PowerSphericalPotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 8: //HernquistPotential, 2 arguments
      potentialArgs->potentialEval= &HernquistPotentialEval;
      potentialArgs->Rforce= &HernquistPotentialRforce;
      potentialArgs->zforce= &HernquistPotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->potentialEval= &NFWPotentialEval;
      potentialArgs->Rforce= &NFWPotentialRforce;
      potentialArgs->zforce= &NFWPotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 10: //JaffePotential, 2 arguments
      potentialArgs->potentialEval= &JaffePotentialEval;
      potentialArgs->Rforce= &JaffePotentialRforce;
      potentialArgs->zforce= &JaffePotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      potentialArgs->potentialEval= &DoubleExponentialDiskPotentialEval;
      potentialArgs->Rforce= &DoubleExponentialDiskPotentialRforce;
      potentialArgs->zforce= &DoubleExponentialDiskPotentialzforce;
      //Look at pot_args to figure out the number of arguments
      potentialArgs->nargs= (int) (8 + 2 * *(*pot_args+5) + 4 * ( *(*pot_args+4) + 1));
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      potentialArgs->potentialEval= &FlattenedPowerPotentialEval;
      potentialArgs->Rforce= &FlattenedPowerPotentialRforce;
      potentialArgs->zforce= &FlattenedPowerPotentialzforce;
      potentialArgs->nargs= 4;
      break;     
    case 13: //interpRZPotential, XX arguments
      //Grab the grids and the coefficients
      nR= (int) *(*pot_args)++;
      nz= (int) *(*pot_args)++;
      Rgrid= (double *) malloc ( nR * sizeof ( double ) );
      zgrid= (double *) malloc ( nz * sizeof ( double ) );
      potGrid_splinecoeffs= (double *) malloc ( nR * nz * sizeof ( double ) );
      for (kk=0; kk < nR; kk++)
	*(Rgrid+kk)= *(*pot_args)++;
      for (kk=0; kk < nz; kk++)
	*(zgrid+kk)= *(*pot_args)++;
      for (kk=0; kk < nR; kk++)
	put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz);
      *pot_args+= nR*nz;
      potentialArgs->i2d= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2d,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accx= gsl_interp_accel_alloc ();
      potentialArgs->accy= gsl_interp_accel_alloc ();
      potentialArgs->potentialEval= &interpRZPotentialEval;
      if ( forTorus == true ) {
	for (kk=0; kk < nR; kk++)
	  put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz); 
	*pot_args+= nR*nz;
	potentialArgs->i2drforce= interp_2d_alloc(nR,nz);
	interp_2d_init(potentialArgs->i2drforce,Rgrid,zgrid,potGrid_splinecoeffs,
		       INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
	potentialArgs->accxrforce= gsl_interp_accel_alloc ();
	potentialArgs->accyrforce= gsl_interp_accel_alloc ();
	for (kk=0; kk < nR; kk++)
	  put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz); 
	*pot_args+= nR*nz;    
	potentialArgs->i2dzforce= interp_2d_alloc(nR,nz);
	interp_2d_init(potentialArgs->i2dzforce,Rgrid,zgrid,potGrid_splinecoeffs,
		       INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
	potentialArgs->accxzforce= gsl_interp_accel_alloc ();
	potentialArgs->accyzforce= gsl_interp_accel_alloc ();
	potentialArgs->Rforce= &interpRZPotentialRforce;
	potentialArgs->zforce= &interpRZPotentialzforce;
      }
      potentialArgs->nargs= 2;
      //clean up
      free(Rgrid);
      free(zgrid);
      free(potGrid_splinecoeffs);
      break;
    case 14: //IsochronePotential, 2 arguments
      potentialArgs->potentialEval= &IsochronePotentialEval;
      potentialArgs->Rforce= &IsochronePotentialRforce;
      potentialArgs->zforce= &IsochronePotentialzforce;
      potentialArgs->nargs= 2;
      break;     
    case 15: //PowerSphericalPotentialwCutoff, 3 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialwCutoffEval;
      potentialArgs->Rforce= &PowerSphericalPotentialwCutoffRforce;
      potentialArgs->zforce= &PowerSphericalPotentialwCutoffzforce;
      potentialArgs->nargs= 3;
      break;
    case 16: //KuzminKutuzovStaeckelPotential, 3 arguments
      potentialArgs->potentialEval= &KuzminKutuzovStaeckelPotentialEval;
      potentialArgs->Rforce= &KuzminKutuzovStaeckelPotentialRforce;
      potentialArgs->zforce= &KuzminKutuzovStaeckelPotentialzforce;
      potentialArgs->nargs= 3;
      break;
    case 17: //PlummerPotential, 2 arguments
      potentialArgs->potentialEval= &PlummerPotentialEval;
      potentialArgs->Rforce= &PlummerPotentialRforce;
      potentialArgs->zforce= &PlummerPotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 18: //PseudoIsothermalPotential, 2 arguments
      potentialArgs->potentialEval= &PseudoIsothermalPotentialEval;
      potentialArgs->Rforce= &PseudoIsothermalPotentialRforce;
      potentialArgs->zforce= &PseudoIsothermalPotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 19: //KuzminDiskPotential, 2 arguments
      potentialArgs->potentialEval= &KuzminDiskPotentialEval;
      potentialArgs->Rforce= &KuzminDiskPotentialRforce;
      potentialArgs->zforce= &KuzminDiskPotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 20: //BurkertPotential, 2 arguments
      potentialArgs->potentialEval= &BurkertPotentialEval;
      potentialArgs->Rforce= &BurkertPotentialRforce;
      potentialArgs->zforce= &BurkertPotentialzforce;
      potentialArgs->nargs= 2;
      break;
    case 21: //TriaxialHernquistPotential, lots of arguments
      potentialArgs->potentialEval= &TriaxialHernquistPotentialEval;
      potentialArgs->Rforce= &TriaxialHernquistPotentialRforce;
      potentialArgs->zforce= &TriaxialHernquistPotentialzforce;
      potentialArgs->nargs= (int) (21 + 2 * *(*pot_args+14));
      break;
    case 22: //TriaxialNFWPotential, lots of arguments
      potentialArgs->potentialEval= &TriaxialNFWPotentialEval;
      potentialArgs->Rforce= &TriaxialNFWPotentialRforce;
      potentialArgs->zforce= &TriaxialNFWPotentialzforce;
      potentialArgs->nargs= (int) (21 + 2 * *(*pot_args+14));
      break;
    case 23: //TriaxialJaffePotential, lots of arguments
      potentialArgs->potentialEval= &TriaxialJaffePotentialEval;
      potentialArgs->Rforce= &TriaxialJaffePotentialRforce;
      potentialArgs->zforce= &TriaxialJaffePotentialzforce;
      potentialArgs->nargs= (int) (21 + 2 * *(*pot_args+14));
      break;
    case 24: //SCFPotential, many arguments
      potentialArgs->potentialEval= &SCFPotentialEval;
      potentialArgs->Rforce= &SCFPotentialRforce;
      potentialArgs->zforce= &SCFPotentialzforce;
      potentialArgs->nargs= (int) (5 + (1 + *(*pot_args + 1)) * *(*pot_args+2) * *(*pot_args+3)* *(*pot_args+4) + 7);
      break;
    case 25: //SoftenedNeedleBarPotential, 13 arguments
      potentialArgs->potentialEval= &SoftenedNeedleBarPotentialEval;
      potentialArgs->Rforce= &SoftenedNeedleBarPotentialRforce;
      potentialArgs->zforce= &SoftenedNeedleBarPotentialzforce;
      potentialArgs->nargs= (int) 13;
      break;
    case 26: //DiskSCFPotential, nsigma+3 arguments
      potentialArgs->potentialEval= &DiskSCFPotentialEval;
      potentialArgs->Rforce= &DiskSCFPotentialRforce;
      potentialArgs->zforce= &DiskSCFPotentialzforce;
      potentialArgs->nargs= (int) **pot_args + 3;
      break;      
//////////////////////////////// WRAPPERS /////////////////////////////////////
    case -1: //DehnenSmoothWrapperPotential
      potentialArgs->potentialEval= &DehnenSmoothWrapperPotentialEval;
      potentialArgs->Rforce= &DehnenSmoothWrapperPotentialRforce;
      potentialArgs->zforce= &DehnenSmoothWrapperPotentialzforce;
      potentialArgs->nargs= (int) 3;
      break;
    }
    if ( *(*pot_type-1) < 0 ) { // Parse wrapped potential for wrappers
      potentialArgs->nwrapped= (int) *(*pot_args)++;
      potentialArgs->wrappedPotentialArg= \
	(struct potentialArg *) malloc ( potentialArgs->nwrapped	\
					 * sizeof (struct potentialArg) );
      parse_actionAngleArgs(potentialArgs->nwrapped,
			    potentialArgs->wrappedPotentialArg,
			    pot_type,pot_args,forTorus);
    }
    potentialArgs->args= (double *) malloc( potentialArgs->nargs * sizeof(double));
    for (jj=0; jj < potentialArgs->nargs; jj++){
      *(potentialArgs->args)= *(*pot_args)++;
      potentialArgs->args++;
    }
    potentialArgs->args-= potentialArgs->nargs;
    potentialArgs++;
  }
  potentialArgs-= npot;
}
