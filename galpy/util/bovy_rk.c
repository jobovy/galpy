/*
  C implementations of Runge-Kutta integrators
*/
/*
Copyright (c) 2011, Jo Bovy
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

   Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright notice, 
      this list of conditions and the following disclaimer in the 
      documentation and/or other materials provided with the distribution.
   The name of the author may not be used to endorse or promote products 
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <bovy_rk.h>
/*
Runge-Kutta 4 integrator
Usage:
   Provide the acceleration function func with calling sequence
       func (t,q,a,nargs,args)
   where
       double t: time
       double * q: current value (dimension: dim)
       double * a: will be set to the derivative by func
       int nargs: number of arguments the function takes
       double *args: arguments
  Other arguments are:
       int dim: dimension
       double *yo: initial value, dimension: dim
       int nt: number of times at which the output is wanted
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       double *args: see above
       double rtol, double atol: relative and absolute tolerance levels desired
  Output:
       double *result: result (nt blocks of size 2dim)
*/
void bovy_rk4(void (*func)(double t, double *q, double *a,
			   int nargs, struct leapFuncArg * leapFuncArgs),
	      int dim,
	      double * yo,
	      int nt, double *t,
	      int nargs, struct leapFuncArg * leapFuncArgs,
	      double rtol, double atol,
	      double *result){
  //Initialize
  double *yn= (double *) malloc ( dim * sizeof(double) );
  double *yn1= (double *) malloc ( dim * sizeof(double) );
  double *ynk= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  int ii, jj, kk;
  save_rk(dim,yo,result);
  result+= dim;
  for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
  for (ii=0; ii < dim; ii++) *(yn1+ii)= *(yo+ii);
  //Estimate necessary stepsize
  double dt= (*(t+1))-(*t);
  double init_dt= dt;
  dt= rk_estimate_step(*func,dim,yo,dt,t,nargs,leapFuncArgs,
		       rtol,atol,4.);
  long ndt= (long) (init_dt/dt);
  //Integrate the system
  double to= *t;
  for (ii=0; ii < (nt-1); ii++){
    for (jj=0; jj < (ndt-1); jj++) {
      bovy_rk4_onestep(func,dim,yn,yn1,to,dt,nargs,leapFuncArgs,ynk,a);
      to+= dt;
      //reset yn
      for (kk=0; kk < dim; kk++) *(yn+kk)= *(yn1+kk);
    }
    bovy_rk4_onestep(func,dim,yn,yn1,to,dt,nargs,leapFuncArgs,ynk,a);
    to+= dt;
    //save
    save_rk(dim,yn1,result);
    result+= dim;
    //reset yn
    for (kk=0; kk < dim; kk++) *(yn+kk)= *(yn1+kk);
  }
  //Free allocated memory
  free(yn);
  free(yn1);
  free(ynk);
  free(a);
  //We're done
}

inline void bovy_rk4_onestep(void (*func)(double t, double *q, double *a,
					  int nargs, struct leapFuncArg * leapFuncArgs),
			     int dim,
			     double * yn,double * yn1,
			     double tn, double dt,
			     int nargs, struct leapFuncArg * leapFuncArgs,
			     double * ynk, double * a){
  int ii;
  //calculate k1
  func(tn,yn,a,nargs,leapFuncArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 6.;
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii) / 2.;
  //calculate k2
  func(tn+dt/2.,ynk,a,nargs,leapFuncArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 3.;
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii) / 2.;
  //calculate k3
  func(tn+dt/2.,ynk,a,nargs,leapFuncArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 3.;
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii);
  //calculate k4
  func(tn+dt,ynk,a,nargs,leapFuncArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 6.;
  //yn1 is new value
}

inline void save_rk(int dim, double *yo, double *result){
  int ii;
  for (ii=0; ii < dim; ii++) *result++= *yo++;
}
double rk_estimate_step(void (*func)(double t, double *y, double *a,int nargs, struct leapFuncArg *),
			int dim, double *yo,
			double dt, double *t,
			int nargs,struct leapFuncArg * leapFuncArgs,
			double rtol,double atol, double order){
  //return dt;
  //scalars
  double err= 2.;
  double max_val;
  double to= *t;
  double *yn= (double *) malloc ( dim * sizeof(double) );
  double *y1= (double *) malloc ( dim * sizeof(double) );
  double *y21= (double *) malloc ( dim * sizeof(double) );
  double *y2= (double *) malloc ( dim * sizeof(double) );
  double *ynk= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *scale= (double *) malloc ( dim * sizeof(double) );
  int ii;
  //find maximum values
  max_val= fabs(*yo);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(yo+ii)) > max_val )
      max_val= fabs(*(yo+ii));
  //set up scale
  double c= fmax(atol, rtol * max_val);
  double s= log(exp(atol-c)+exp(rtol*max_val-c))+c;
  for (ii=0; ii < dim; ii++) *(scale+ii)= s;
  //find good dt
  dt*= 2.;
  while ( err > 1. ){
    dt/= 2.;
    //copy initial codition
    for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
    for (ii=0; ii < dim; ii++) *(y1+ii)= *(yo+ii);
    for (ii=0; ii < dim; ii++) *(y21+ii)= *(yo+ii);
    //do one step with step dt, and one with step dt/2.
    //dt
    bovy_rk4_onestep(func,dim,yn,y1,to,dt,nargs,leapFuncArgs,ynk,a);
    //dt/2
    bovy_rk4_onestep(func,dim,yn,y21,to,dt/2.,nargs,leapFuncArgs,ynk,a);
    for (ii=0; ii < dim; ii++) *(y2+ii)= *(y21+ii);
    bovy_rk4_onestep(func,dim,y21,y2,to+dt/2.,dt/2.,nargs,leapFuncArgs,ynk,a);
    //Norm
    err= 0.;
    for (ii=0; ii < dim; ii++) {
      err+= exp(2.*log(fabs(*(y1+ii)-*(y2+ii)))-2.* *(scale+ii));
    }
    err= sqrt(err/dim);
  }
  //free what we allocated
  free(yn);
  free(y1);
  free(y2);
  free(y21);
  free(ynk);
  free(a);
  free(scale);
  //return
  return dt;
} 
