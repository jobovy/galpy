/*
  C code for actionAngle calculations
*/
#ifndef __GALPY_ACTIONANGLE_H__
#define __GALPY_ACTIONANGLE_H_
#include <gsl/gsl_roots.h>
/*
  Structure declarations
*/
struct actionAngleArg{
  double (*potentialEval)(double R, double Z, double phi, double t,
			  int nargs, double * args);
  int nargs;
  double * args;
};
struct pragmasolver{
  gsl_root_fsolver *s;
};
/*
  Function declarations
*/
double evaluatePotentials(double,double,int, struct actionAngleArg *);
void parse_actionAngleArgs(int,struct actionAngleArg *,int *,double *);
#endif /* actionAngle.h */
