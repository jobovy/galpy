/*
  C code for actionAngle calculations
*/
#ifndef __GALPY_ACTIONANGLE_H__
#define __GALPY_ACTIONANGLE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_spline.h>
#include "interp_2d.h"
/*
  Macro for dealing with potentially unused variables due to OpenMP
 */
/* If we're not using GNU C, elide __attribute__ if it doesn't exist*/
#ifndef __has_attribute      // Compatibility with non-clang compilers. 
#define __has_attribute(x) 0  
#endif
#if defined(__GNUC__) || __has_attribute(unused)
#  define UNUSED __attribute__((unused))
#else
#  define UNUSED /*NOTHING*/
#endif
/*
  Structure declarations
*/
struct actionAngleArg{ //I think this isn't used JB 06/24/14
  double (*potentialEval)(double R, double Z, double phi, double t,
			  int nargs, double * args);
  int nargs;
  double * args;
  interp_2d * i2d;
  gsl_interp_accel * acc;
};
struct pragmasolver{
  gsl_root_fsolver *s;
};
/*
  Function declarations
*/
  void parse_actionAngleArgs(int,struct potentialArg *,int **,double **,bool);
#ifdef __cplusplus
}
#endif
#endif /* actionAngle.h */
