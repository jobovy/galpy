/******************************************************************************
 *                      Code generated with SymPy 1.13.3                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'example'                       *
 ******************************************************************************/
#include "myfunc.h"
#include <math.h>

double myfunc(double x, double y) {

   double myfunc_result;
   myfunc_result = sqrt(pow(x, 2) + pow(y, 2)) + exp(y)*sin(x);
   return myfunc_result;

}
