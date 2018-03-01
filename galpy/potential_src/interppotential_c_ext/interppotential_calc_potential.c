/*
  C code for calculating a potential and its forces on a grid
*/
#ifdef _WIN32
#include <Python.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define CHUNKSIZE 1
//Potentials
#include <galpy_potentials.h>
#include <actionAngle.h>
#include <integrateFullOrbit.h>
#include <interp_2d.h>
#include <cubic_bspline_2d_coeffs.h>
#ifdef _WIN32
// On Windows, *need* to define this function to allow the package to be imported
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_galpy_interppotential_c(void) { // Python 3
  return NULL;
}
#else
PyMODINIT_FUNC initgalpy_interppotential_c(void) {} // Python 2
#endif
#endif
//Macros to export functions in DLL on different OS
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
// Just do nothing?
#define EXPORT
#endif
/*
  MAIN FUNCTIONS
*/
EXPORT void calc_potential(int nR,
			   double *R,
			   int nz,
			   double *z,
			   int npot,
			   int * pot_type,
			   double * pot_args,
			   double *out,
			   int * err){
  int ii, jj, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  double * row= (double *) malloc ( nthreads * nz * ( sizeof ( double ) ) );
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Run through the grid and calculate
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tid,jj)	\
  shared(row,npot,potentialArgs,R,z,nR,nz)
  for (ii=0; ii < nR; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    for (jj=0; jj < nz; jj++){
      *(row+jj+tid*nz)= evaluatePotentials(*(R+ii),*(z+jj),npot,potentialArgs);
    }
    put_row(out,ii,row+tid*nz,nz); 
  }
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  free(row);
}
EXPORT void calc_rforce(int nR,
			double *R,
			int nz,
			double *z,
			int npot,
			int * pot_type,
			double * pot_args,
			double *out,
			int * err){
  int ii, jj, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  double * row= (double *) malloc ( nthreads * nz * ( sizeof ( double ) ) );
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Run through the grid and calculate
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tid,jj)	\
  shared(row,npot,potentialArgs,R,z,nR,nz)
  for (ii=0; ii < nR; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    for (jj=0; jj < nz; jj++){
      *(row+jj+tid*nz)= calcRforce(*(R+ii),*(z+jj),0.,0.,npot,potentialArgs);
    }
    put_row(out,ii,row+tid*nz,nz); 
  }
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  free(row);
}
EXPORT void calc_zforce(int nR,
			double *R,
			int nz,
			double *z,
			int npot,
			int * pot_type,
			double * pot_args,
			double *out,
			int * err){
  int ii, jj, tid, nthreads;
#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#else
  nthreads = 1;
#endif
  double * row= (double *) malloc ( nthreads * nz * ( sizeof ( double ) ) );
  //Set up the potentials
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Run through the grid and calculate
  UNUSED int chunk= CHUNKSIZE;
#pragma omp parallel for schedule(static,chunk) private(ii,tid,jj)	\
  shared(row,npot,potentialArgs,R,z,nR,nz)
  for (ii=0; ii < nR; ii++){
#ifdef _OPENMP
    tid= omp_get_thread_num();
#else
    tid = 0;
#endif
    for (jj=0; jj < nz; jj++){
      *(row+jj+tid*nz)= calczforce(*(R+ii),*(z+jj),0.,0.,npot,potentialArgs);
    }
    put_row(out,ii,row+tid*nz,nz); 
  }
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  free(row);
}
EXPORT void eval_potential(int nR,
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
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Run through and evaluate
  for (ii=0; ii < nR; ii++){
    *(out+ii)= evaluatePotentials(*(R+ii),*(z+ii),npot,potentialArgs);
  }
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
}
EXPORT void eval_rforce(int nR,
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
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Run through and evaluate
  for (ii=0; ii < nR; ii++){
    *(out+ii)= calcRforce(*(R+ii),*(z+ii),0.,0.,npot,potentialArgs);
  }
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
}
EXPORT void eval_zforce(int nR,
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
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args);
  //Run through and evaluate
  for (ii=0; ii < nR; ii++){
    *(out+ii)= calczforce(*(R+ii),*(z+ii),0.,0.,npot,potentialArgs);
  }
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
}
