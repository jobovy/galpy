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
  struct actionAngleArg * actionAngleArgs= (struct actionAngleArg *) malloc ( npot * sizeof (struct actionAngleArg) );
  parse_actionAngleArgs(npot,actionAngleArgs,pot_type,pot_args);
  //Run through the grid and calculate
  for (ii=0; ii < nR; ii++){
    for (jj=0; jj < nz; jj++){
      *(row+jj)= evaluatePotentials(*(R+ii),*(z+jj),npot,actionAngleArgs);
    }
    put_row(out,ii ,row,nz); 
  }
  free(actionAngleArgs);
  free(row);
}
