/*
  C code for calculating a potential and its forces on a grid
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
//Potentials
#include <galpy_potentials.h>
#include <actionAngle.h>
#include <integrateFullOrbit.h>
#include <interp_2d.h>
#include <cubic_bspline_2d_coeffs.h>
/*
  MAIN FUNCTIONS
*/
void calc_potential(int nR,
		    double *R,
		    int nz,
		    double *z,
		    int npot,
		    int * pot_type,
		    double * pot_args,
		    double *out,
		    int * err){
  int ii, jj;
  double * row= (double *) malloc ( nz * ( sizeof ( double ) ) );
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_actionAngleArgs(npot,potentialArgs,pot_type,pot_args);
  //Run through the grid and calculate
  for (ii=0; ii < nR; ii++){
    for (jj=0; jj < nz; jj++){
      *(row+jj)= evaluatePotentials(*(R+ii),*(z+jj),npot,potentialArgs);
    }
    put_row(out,ii ,row,nz); 
  }
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->i2d )
      interp_2d_free((potentialArgs+ii)->i2d) ;
    if ((potentialArgs+ii)->acc )
      gsl_interp_accel_free ((potentialArgs+ii)->acc);
    free((potentialArgs+ii)->args);
  }
  free(potentialArgs);
  free(row);
}
void calc_rforce(int nR,
		 double *R,
		 int nz,
		 double *z,
		 int npot,
		 int * pot_type,
		 double * pot_args,
		 double *out,
		 int * err){
  int ii, jj;
  double * row= (double *) malloc ( nz * ( sizeof ( double ) ) );
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,pot_type,pot_args);
  //Run through the grid and calculate
  for (ii=0; ii < nR; ii++){
    for (jj=0; jj < nz; jj++){
      *(row+jj)= calcRforce(*(R+ii),*(z+jj),0.,0.,npot,potentialArgs);
    }
    put_row(out,ii ,row,nz); 
  }
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->i2drforce )
      interp_2d_free((potentialArgs+ii)->i2drforce ) ;
    if ((potentialArgs+ii)->accrforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accrforce );
    if ( (potentialArgs+ii)->i2dzforce )
      interp_2d_free((potentialArgs+ii)->i2dzforce ) ;
    if ((potentialArgs+ii)->acczforce )
      gsl_interp_accel_free ((potentialArgs+ii)->acczforce );
    free((potentialArgs+ii)->args);
  }
  free(potentialArgs);
  free(row);
}
void calc_zforce(int nR,
		 double *R,
		 int nz,
		 double *z,
		 int npot,
		 int * pot_type,
		 double * pot_args,
		 double *out,
		 int * err){
  int ii, jj;
  double * row= (double *) malloc ( nz * ( sizeof ( double ) ) );
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,pot_type,pot_args);
  //Run through the grid and calculate
  for (ii=0; ii < nR; ii++){
    for (jj=0; jj < nz; jj++){
      *(row+jj)= calczforce(*(R+ii),*(z+jj),0.,0.,npot,potentialArgs);
    }
    put_row(out,ii ,row,nz); 
  }
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->i2drforce )
      interp_2d_free((potentialArgs+ii)->i2drforce ) ;
    if ((potentialArgs+ii)->accrforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accrforce );
    if ( (potentialArgs+ii)->i2dzforce )
      interp_2d_free((potentialArgs+ii)->i2dzforce ) ;
    if ((potentialArgs+ii)->acczforce )
      gsl_interp_accel_free ((potentialArgs+ii)->acczforce );
    free((potentialArgs+ii)->args);
  }
  free(potentialArgs);
  free(row);
}
void eval_potential(int nR,
		    double *R,
		    double *z,
		    int npot,
		    int * pot_type,
		    double * pot_args,
		    double *out,
		    int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_actionAngleArgs(npot,potentialArgs,pot_type,pot_args);
  //Run through and evaluate
  for (ii=0; ii < nR; ii++){
    *(out+ii)= evaluatePotentials(*(R+ii),*(z+ii),npot,potentialArgs);
  }
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->i2d )
      interp_2d_free((potentialArgs+ii)->i2d) ;
    if ((potentialArgs+ii)->acc )
      gsl_interp_accel_free ((potentialArgs+ii)->acc);
    free((potentialArgs+ii)->args);
  }
  free(potentialArgs);
}
void eval_rforce(int nR,
		 double *R,
		 double *z,
		 int npot,
		 int * pot_type,
		 double * pot_args,
		 double *out,
		 int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,pot_type,pot_args);
  //Run through and evaluate
  for (ii=0; ii < nR; ii++){
    *(out+ii)= calcRforce(*(R+ii),*(z+ii),0.,0.,npot,potentialArgs);
  }
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->i2drforce )
      interp_2d_free((potentialArgs+ii)->i2drforce) ;
    if ((potentialArgs+ii)->accrforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accrforce);
    if ( (potentialArgs+ii)->i2dzforce )
      interp_2d_free((potentialArgs+ii)->i2dzforce) ;
    if ((potentialArgs+ii)->acczforce )
      gsl_interp_accel_free ((potentialArgs+ii)->acczforce);
    free((potentialArgs+ii)->args);
  }
  free(potentialArgs);
}
void eval_zforce(int nR,
		 double *R,
		 double *z,
		 int npot,
		 int * pot_type,
		 double * pot_args,
		 double *out,
		 int * err){
  int ii;
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,pot_type,pot_args);
  //Run through and evaluate
  for (ii=0; ii < nR; ii++){
    *(out+ii)= calczforce(*(R+ii),*(z+ii),0.,0.,npot,potentialArgs);
  }
  for (ii=0; ii < npot; ii++) {
    if ( (potentialArgs+ii)->i2drforce )
      interp_2d_free((potentialArgs+ii)->i2drforce) ;
    if ((potentialArgs+ii)->accrforce )
      gsl_interp_accel_free ((potentialArgs+ii)->accrforce);
    if ( (potentialArgs+ii)->i2dzforce )
      interp_2d_free((potentialArgs+ii)->i2dzforce) ;
    if ((potentialArgs+ii)->acczforce )
      gsl_interp_accel_free ((potentialArgs+ii)->acczforce);
    free((potentialArgs+ii)->args);
  }
  free(potentialArgs);
}
