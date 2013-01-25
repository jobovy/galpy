/*
  C code for actionAngle calculations
*/
#ifndef __GALPY_ACTIONANGLE_H__
#define __GALPY_ACTIONANGLE_H_
#include <gsl/gsl_roots.h>
#include <gsl/gsl_spline.h>
#include "interp_2d.h"
/*
  Structure declarations
*/
struct actionAngleArg{
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
double evaluatePotentials(double,double,int, struct potentialArg *);
void parse_actionAngleArgs(int,struct potentialArg *,int *,double *);
#endif /* actionAngle.h */
