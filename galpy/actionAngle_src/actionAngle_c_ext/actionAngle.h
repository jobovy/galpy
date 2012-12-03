/*
  C code for actionAngle calculations
*/
#ifndef __GALPY_ACTIONANGLE_H__
#define __GALPY_ACTIONANGLE_H_
/*
  Structure declarations
*/
struct actionAngleArg{
  double (*potentialEval)(double R, double Z, double phi, double t,
			  int nargs, double * args);
  int nargs;
  double * args;
};
