#ifndef __INTEGRATEFULLORBIT_H__
#define __INTEGRATEFULLORBIT_H__
#include <galpy_potentials.h>
void parse_leapFuncArgs_Full(int, struct potentialArg *,int *,double *);
double calcRforce(double,double,double,double,int,struct potentialArg *);
double calczforce(double,double,double,double,int,struct potentialArg *);
#endif /* integrateFullOrbit.h */
