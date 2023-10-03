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
#include <bovy_symplecticode.h>
#include <bovy_rk.h>
#include "signal.h"
#define _MAX_STEPCHANGE_POWERTWO 3.
#define _MIN_STEPCHANGE_POWERTWO -3.
#define _MAX_STEPREDUCE 10000.
#define _MAX_DT_REDUCE 10000.
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
       double dt: (optional) stepsize to use, must be an integer divisor of time difference between output steps (NOT CHECKED EXPLICITLY)
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       double *args: see above
       double rtol, double atol: relative and absolute tolerance levels desired
  Output:
       double *result: result (nt blocks of size 2dim)
       int *err: error: -10 if interrupted by CTRL-C (SIGINT)
*/
void bovy_rk4(void (*func)(double t, double *q, double *a,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      double *result, int * err){
  //Declare and initialize
  double *yn= (double *) malloc ( dim * sizeof(double) );
  double *yn1= (double *) malloc ( dim * sizeof(double) );
  double *ynk= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  int ii, jj, kk;
  save_rk(dim,yo,result);
  result+= dim;
  *err= 0;
  for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
  for (ii=0; ii < dim; ii++) *(yn1+ii)= *(yo+ii);
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= rk4_estimate_step(*func,dim,yo,init_dt,t,nargs,potentialArgs,
			  rtol,atol);
  }
  long ndt= (long) (init_dt/dt);
  //Integrate the system
  double to= *t;
  // Handle KeyboardInterrupt gracefully
#ifndef _WIN32
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler= handle_sigint;
  sigaction(SIGINT,&action,NULL);
#else
    if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {}
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0; // need to reset, bc library and vars stay in memory
#ifdef USING_COVERAGE
      __gcov_dump();
// LCOV_EXCL_START
      __gcov_reset();
#endif
      break;
// LCOV_EXCL_STOP
    }
    for (jj=0; jj < (ndt-1); jj++) {
      bovy_rk4_onestep(func,dim,yn,yn1,to,dt,nargs,potentialArgs,ynk,a);
      to+= dt;
      //reset yn
      for (kk=0; kk < dim; kk++) *(yn+kk)= *(yn1+kk);
    }
    bovy_rk4_onestep(func,dim,yn,yn1,to,dt,nargs,potentialArgs,ynk,a);
    to+= dt;
    //save
    save_rk(dim,yn1,result);
    result+= dim;
    //reset yn
    for (kk=0; kk < dim; kk++) *(yn+kk)= *(yn1+kk);
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //Free allocated memory
  free(yn);
  free(yn1);
  free(ynk);
  free(a);
  //We're done
}

void bovy_rk4_onestep(void (*func)(double t, double *q, double *a,
				   int nargs, struct potentialArg * potentialArgs),
		      int dim,
		      double * yn,double * yn1,
		      double tn, double dt,
		      int nargs, struct potentialArg * potentialArgs,
		      double * ynk, double * a){
  int ii;
  //calculate k1
  func(tn,yn,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 6.;
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii) / 2.;
  //calculate k2
  func(tn+dt/2.,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 3.;
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii) / 2.;
  //calculate k3
  func(tn+dt/2.,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 3.;
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + dt * *(a+ii);
  //calculate k4
  func(tn+dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += dt * *(a+ii) / 6.;
  //yn1 is new value
}

/*
  RK6 integrator, same calling sequence as RK4
*/
void bovy_rk6(void (*func)(double t, double *q, double *a,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      double *result, int * err){
  //Declare and initialize
  double *yn= (double *) malloc ( dim * sizeof(double) );
  double *yn1= (double *) malloc ( dim * sizeof(double) );
  double *ynk= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *k1= (double *) malloc ( dim * sizeof(double) );
  double *k2= (double *) malloc ( dim * sizeof(double) );
  double *k3= (double *) malloc ( dim * sizeof(double) );
  double *k4= (double *) malloc ( dim * sizeof(double) );
  double *k5= (double *) malloc ( dim * sizeof(double) );
  int ii, jj, kk;
  save_rk(dim,yo,result);
  result+= dim;
  *err= 0;
  for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
  for (ii=0; ii < dim; ii++) *(yn1+ii)= *(yo+ii);
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= rk6_estimate_step(*func,dim,yo,init_dt,t,nargs,potentialArgs,
			  rtol,atol);
  }
  long ndt= (long) (init_dt/dt);
  //Integrate the system
  double to= *t;
  // Handle KeyboardInterrupt gracefully
#ifndef _WIN32
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler= handle_sigint;
  sigaction(SIGINT,&action,NULL);
#else
    if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {}
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0; // need to reset, bc library and vars stay in memory
#ifdef USING_COVERAGE
      __gcov_dump();
// LCOV_EXCL_START
      __gcov_reset();
#endif
      break;
// LCOV_EXCL_STOP
    }
    for (jj=0; jj < (ndt-1); jj++) {
      bovy_rk6_onestep(func,dim,yn,yn1,to,dt,nargs,potentialArgs,ynk,a,
		       k1,k2,k3,k4,k5);
      to+= dt;
      //reset yn
      for (kk=0; kk < dim; kk++) *(yn+kk)= *(yn1+kk);
    }
    bovy_rk6_onestep(func,dim,yn,yn1,to,dt,nargs,potentialArgs,ynk,a,
		     k1,k2,k3,k4,k5);
    to+= dt;
    //save
    save_rk(dim,yn1,result);
    result+= dim;
    //reset yn
    for (kk=0; kk < dim; kk++) *(yn+kk)= *(yn1+kk);
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //Free allocated memory
  free(yn);
  free(yn1);
  free(ynk);
  free(a);
  free(k1);
  free(k2);
  free(k3);
  free(k4);
  free(k5);
  //We're done
}
/* RK6 SOLVER: needs 7 function evaluations per step

  x[i+1] = x[i] + (11*k_1 + 81*k_3 + 81*k_4 - 32*k_5 - 32*k_6 + 11*k_7)/120
  k_1 = step * dxdt(start, x[i])
  k_2 = step * dxdt(start + step/3, x[i] + k_1/3)
  k_3 = step * dxdt(start + 2*step/3, x[i] + 2 * k_2/3)
  k_4 = step * dxdt(start + step/3, x[i] + (k_1 + 4*k_2 - k_3)/12)
  k_5 = step * dxdt(start + step/2, x[i] + (-k_1 + 18*k_2 - 3*k_3 - 6*k_4)/16)
  k_6 = step * dxdt(start + step/2, x[i] + (9 * k_2 - 3*k_3 - 6*k_4 + 4*k_5)/8)
  k_7 = step * dxdt(start + step, x[i] + (9*k_1 - 36*k_2 + 63*k_3 +
72*k_4 -64*k_6)/44)
*/
void bovy_rk6_onestep(void (*func)(double t, double *q, double *a,
				   int nargs, struct potentialArg * potentialArgs),
		      int dim,
		      double * yn,double * yn1,
		      double tn, double dt,
		      int nargs, struct potentialArg * potentialArgs,
		      double * ynk, double * a,
		      double * k1, double * k2,
		      double * k3, double * k4,
		      double * k5){
  int ii;
  //calculate k1
  func(tn,yn,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += 11.* dt * *(a+ii) / 120.;
  for (ii=0; ii < dim; ii++) *(k1+ii)= dt * *(a+ii);
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + *(k1+ii)/3.;
  //calculate k2
  func(tn+dt/3.,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(k2+ii)= dt * *(a+ii);
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + 2. * *(k2+ii)/3.;
  //calculate k3
  func(tn+2.*dt/3.,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += 81. * dt * *(a+ii) / 120.;
  for (ii=0; ii < dim; ii++) *(k3+ii)= dt * *(a+ii);
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + ( *(k1+ii)
						     + 4. * *(k2+ii)
						     - *(k3+ii))/12.;
  //calculate k4
  func(tn+dt/3.,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += 81.* dt * *(a+ii) / 120.;
  for (ii=0; ii < dim; ii++) *(k4+ii)= dt * *(a+ii);
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + ( -*(k1+ii)
						     + 18. * *(k2+ii)
						     - 3. * *(k3+ii)
						     -6.* *(k4+ii))/16.;
  //calculate k5
  func(tn+dt/2.,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) -= 32.* dt * *(a+ii) / 120.;
  for (ii=0; ii < dim; ii++) *(k5+ii)= dt * *(a+ii);
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + ( 9. * *(k2+ii)
						     - 3. * *(k3+ii)
						     -6.* *(k4+ii)
						     + 4. * *(k5+ii))/8.;
  //calculate k6
  func(tn+dt/2.,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) -= 32.* dt * *(a+ii) / 120.;
  for (ii=0; ii < dim; ii++) *(k5+ii)= dt * *(a+ii); //reuse k5 for k6
  for (ii=0; ii < dim; ii++) *(ynk+ii)= *(yn+ii) + ( 9. * *(k1+ii)
						     - 36. * *(k2+ii)
						     +63.* *(k3+ii)
						     + 72. * *(k4+ii)
						     -64. * *(k5+ii))/44.;
  //calculate k7
  func(tn+dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yn1+ii) += 11.* dt * *(a+ii) / 120.;
  //yn1 is new value
}

double rk4_estimate_step(void (*func)(double t, double *y, double *a,int nargs, struct potentialArg *),
			 int dim, double *yo,
			 double dt, double *t,
			 int nargs,struct potentialArg * potentialArgs,
			 double rtol,double atol){
  //return dt;
  //scalars
  double err= 2.;
  double max_val;
  double to= *t;
  double init_dt= dt;
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
  //dt*= 2.;
  while ( err > 1. ){
    //dt/= 2.;
    //copy initial condition
    for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
    for (ii=0; ii < dim; ii++) *(y1+ii)= *(yo+ii);
    for (ii=0; ii < dim; ii++) *(y21+ii)= *(yo+ii);
    //do one step with step dt, and one with step dt/2.
    //dt
    bovy_rk4_onestep(func,dim,yn,y1,to,dt,nargs,potentialArgs,ynk,a);
    //dt/2
    bovy_rk4_onestep(func,dim,yn,y21,to,dt/2.,nargs,potentialArgs,ynk,a);
    for (ii=0; ii < dim; ii++) *(y2+ii)= *(y21+ii);
    bovy_rk4_onestep(func,dim,y21,y2,to+dt/2.,dt/2.,nargs,potentialArgs,ynk,a);
    //Norm
    err= 0.;
    for (ii=0; ii < dim; ii++) {
      err+= exp(2.*log(fabs(*(y1+ii)-*(y2+ii)))-2.* *(scale+ii));
    }
    err= sqrt(err/dim);
    if ( ceil(pow(err,1./5.)) > 1.
	 && init_dt / dt * ceil(pow(err,1./5.)) < _MAX_DT_REDUCE)
      dt/= ceil(pow(err,1./5.));
    else
      break;
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
  //printf("%f\n",dt);
  //fflush(stdout);
  return dt;
}
double rk6_estimate_step(void (*func)(double t, double *y, double *a,int nargs, struct potentialArg *),
			 int dim, double *yo,
			 double dt, double *t,
			 int nargs,struct potentialArg * potentialArgs,
			 double rtol,double atol){
  //return dt;
  //scalars
  double err= 2.;
  double max_val;
  double to= *t;
  double init_dt= dt;
  double *yn= (double *) malloc ( dim * sizeof(double) );
  double *y1= (double *) malloc ( dim * sizeof(double) );
  double *y21= (double *) malloc ( dim * sizeof(double) );
  double *y2= (double *) malloc ( dim * sizeof(double) );
  double *ynk= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *k1= (double *) malloc ( dim * sizeof(double) );
  double *k2= (double *) malloc ( dim * sizeof(double) );
  double *k3= (double *) malloc ( dim * sizeof(double) );
  double *k4= (double *) malloc ( dim * sizeof(double) );
  double *k5= (double *) malloc ( dim * sizeof(double) );
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
  //dt*= 2.;
  while ( err > 1. ){
    //dt/= 2.;
    //copy initial condition
    for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
    for (ii=0; ii < dim; ii++) *(y1+ii)= *(yo+ii);
    for (ii=0; ii < dim; ii++) *(y21+ii)= *(yo+ii);
    //do one step with step dt, and one with step dt/2.
    //dt
    bovy_rk6_onestep(func,dim,yn,y1,to,dt,nargs,potentialArgs,ynk,a,
		     k1,k2,k3,k4,k5);
    //dt/2
    bovy_rk6_onestep(func,dim,yn,y21,to,dt/2.,nargs,potentialArgs,ynk,a,
		     k1,k2,k3,k4,k5);
    for (ii=0; ii < dim; ii++) *(y2+ii)= *(y21+ii);
    bovy_rk6_onestep(func,dim,y21,y2,to+dt/2.,dt/2.,nargs,potentialArgs,ynk,a,
		     k1,k2,k3,k4,k5);
    //Norm
    err= 0.;
    for (ii=0; ii < dim; ii++) {
      err+= exp(2.*log(fabs(*(y1+ii)-*(y2+ii)))-2.* *(scale+ii));
    }
    err= sqrt(err/dim);
    if ( ceil(pow(err,1./7.)) > 1.
	 && init_dt / dt * ceil(pow(err,1./7.)) < _MAX_DT_REDUCE)
      dt/= ceil(pow(err,1./7.));
    else
      break;
  }
  //free what we allocated
  free(yn);
  free(y1);
  free(y2);
  free(y21);
  free(ynk);
  free(a);
  free(scale);
  free(k1);
  free(k2);
  free(k3);
  free(k4);
  free(k5);
  //return
  //printf("%f\n",dt);
  //fflush(stdout);
  return dt;
}
/*
Runge-Kutta Dormand-Prince 5/4 integrator
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
       double dt_one: (optional) stepsize to use, must be an integer divisor of time difference between output steps (NOT CHECKED EXPLICITLY)
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       double *args: see above
       double rtol, double atol: relative and absolute tolerance levels desired
  Output:
       double *result: result (nt blocks of size 2dim)
       int * err: if non-zero, something bad happened (1: maximum step reduction happened; -10: interrupted by CTRL-C (SIGINT)
*/
void bovy_dopr54(void (*func)(double t, double *q, double *a,
			      int nargs, struct potentialArg * potentialArgs),
		 int dim,
		 double * yo,
		 int nt, double dt_one, double *t,
		 int nargs, struct potentialArg * potentialArgs,
		 double rtol, double atol,
		 double *result, int * err){
  //Declare and initialize
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *a1= (double *) malloc ( dim * sizeof(double) );
  double *k1= (double *) malloc ( dim * sizeof(double) );
  double *k2= (double *) malloc ( dim * sizeof(double) );
  double *k3= (double *) malloc ( dim * sizeof(double) );
  double *k4= (double *) malloc ( dim * sizeof(double) );
  double *k5= (double *) malloc ( dim * sizeof(double) );
  double *k6= (double *) malloc ( dim * sizeof(double) );
  double *yn= (double *) malloc ( dim * sizeof(double) );
  double *yn1= (double *) malloc ( dim * sizeof(double) );
  double *yerr= (double *) malloc ( dim * sizeof(double) );
  double *ynk= (double *) malloc ( dim * sizeof(double) );
  int ii;
  save_rk(dim,yo,result);
  result+= dim;
  *err= 0;
  for (ii=0; ii < dim; ii++) *(yn+ii)= *(yo+ii);
  double dt= (*(t+1))-(*t);
  if ( dt_one == -9999.99 ) {
    dt_one= rk4_estimate_step(*func,dim,yo,dt,t,nargs,potentialArgs,
			      rtol,atol);
  }
  //Integrate the system
  double to= *t;
  //set up a1
  func(to,yn,a1,nargs,potentialArgs);
  // Handle KeyboardInterrupt gracefully
#ifndef _WIN32
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler= handle_sigint;
  sigaction(SIGINT,&action,NULL);
#else
    if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {}
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0; // need to reset, bc library and vars stay in memory
#ifdef USING_COVERAGE
      __gcov_dump();
// LCOV_EXCL_START
      __gcov_reset();
#endif
      break;
// LCOV_EXCL_STOP
    }
    bovy_dopr54_onestep(func,dim,yn,dt,&to,&dt_one,
			nargs,potentialArgs,rtol,atol,
			a1,a,k1,k2,k3,k4,k5,k6,yn1,yerr,ynk,err);
    //save
    save_rk(dim,yn,result);
    result+= dim;
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  // Free allocated memory
  free(a);
  free(a1);
  free(k1);
  free(k2);
  free(k3);
  free(k4);
  free(k5);
  free(k6);
  free(yn);
  free(yn1);
  free(yerr);
  free(ynk);
}
//one output step, consists of multiple steps potentially
void bovy_dopr54_onestep(void (*func)(double t, double *y, double *a,int nargs, struct potentialArg *),
			 int dim, double *yo,
			 double dt, double *to,double * dt_one,
			 int nargs,struct potentialArg * potentialArgs,
			 double rtol,double atol,
			 double * a1, double * a,
			 double * k1, double * k2,
			 double * k3, double * k4,
			 double * k5, double * k6,
			 double * yn1, double * yerr,double * ynk, int * err){
  double init_dt_one= *dt_one;
  double init_to= *to;
  unsigned char accept;
  //printf("%f,%f\n",*to,init_to+dt);
  while ( ( dt >= 0. && *to < (init_to+dt))
	  || ( dt < 0. && *to > (init_to+dt)) ) {
    accept= 0;
    if ( init_dt_one/ *dt_one > _MAX_STEPREDUCE
	 || *dt_one != *dt_one) { // check for NaN
      *dt_one= init_dt_one/_MAX_STEPREDUCE;
      accept= 1;
      if ( *err % 2 ==  0) *err+= 1;
    }
    if ( dt >= 0. && *dt_one > (init_to+dt - *to) )
      *dt_one= (init_to + dt - *to);
    if ( dt < 0. && *dt_one < (init_to+dt - *to) )
      *dt_one = (init_to + dt - *to);
    *dt_one= bovy_dopr54_actualstep(func,dim,yo,*dt_one,to,nargs,potentialArgs,
				    rtol,atol,
				    a1,a,k1,k2,k3,k4,k5,k6,yn1,yerr,ynk,
				    accept);
  }
}
double bovy_dopr54_actualstep(void (*func)(double t, double *y, double *a,int nargs, struct potentialArg *),
			      int dim, double *yo,
			      double dt, double *to,
			      int nargs,struct potentialArg * potentialArgs,
			      double rtol,double atol,
			      double * a1, double * a,
			      double * k1, double * k2,
			      double * k3, double * k4,
			      double * k5, double * k6,
			      double * yn1, double * yerr,double * ynk,
			      unsigned char accept){
  //constant
  static const double c2= 0.2;
  static const double c3= 0.3;
  static const double c4= 0.8;
  static const double c5= 8./9.;
  static const double a21= 0.2;
  static const double a31= 3./40.;
  static const double a41= 44./45.;
  static const double a51= 19372./6561;
  static const double a61= 9017./3168.;
  static const double a71= 35./384.;
  static const double a32= 9./40.;
  static const double a42= -56./15.;
  static const double a52= -25360./2187.;
  static const double a62= -355./33.;
  static const double a43= 32./9.;
  static const double a53= 64448./6561.;
  static const double a63= 46732./5247.;
  static const double a73= 500./1113.;
  static const double a54= -212./729.;
  static const double a64= 49./176.;
  static const double a74= 125./192.;
  static const double a65= -5103./18656.;
  static const double a75= -2187./6784.;
  static const double a76= 11./84.;
  static const double b1= 35./384.;
  static const double b3= 500./1113.;
  static const double b4= 125./192.;
  static const double b5= -2187./6784.;
  static const double b6= 11./84.;
  //coefficients of the error estimate
  const double be1= b1-5179./57600.;
  const double be3= b3-7571./16695.;
  const double be4= b4-393./640.;
  const double be5= b5+92097./339200.;
  const double be6= b6-187./2100.;
  static const double be7= -1./40.;
  int ii;
  //setup yn1
  for (ii=0; ii < dim; ii++) *(yn1+ii) = *(yo+ii);
  //calculate k1
  for (ii=0; ii < dim; ii++) *(a+ii)= *(a1+ii);
  for (ii=0; ii < dim; ii++){
    *(k1+ii)= dt * *(a+ii);
    *(yn1+ii) += b1* *(k1+ii);
    *(yerr+ii) = be1* *(k1+ii);
    *(ynk+ii)= *(yo+ii) + a21 * *(k1+ii);
  }
  //calculate k2
  func(*to+c2*dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++){
    *(k2+ii)= dt * *(a+ii);
    *(ynk+ii)= *(yo+ii) + a31 * *(k1+ii)
      + a32 * *(k2+ii);
  }
  //calculate k3
  func(*to+c3*dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++){
    *(k3+ii)= dt * *(a+ii);
    *(yn1+ii) += b3* *(k3+ii);
    *(yerr+ii) += be3* *(k3+ii);
    *(ynk+ii)= *(yo+ii) + a41 * *(k1+ii)
      + a42 * *(k2+ii) + a43 * *(k3+ii);
  }
  //calculate k4
  func(*to+c4*dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++){
    *(k4+ii)= dt * *(a+ii);
    *(yn1+ii) += b4* *(k4+ii);
    *(yerr+ii) += be4* *(k4+ii);
    *(ynk+ii)= *(yo+ii) + a51 * *(k1+ii)
      + a52 * *(k2+ii) + a53 * *(k3+ii)
      + a54 * *(k4+ii);
  }
  //calculate k5
  func(*to+c5*dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++){
    *(k5+ii)= dt * *(a+ii);
    *(yn1+ii) += b5* *(k5+ii);
    *(yerr+ii) += be5* *(k5+ii);
    *(ynk+ii)= *(yo+ii) + a61 * *(k1+ii)
      + a62 * *(k2+ii) + a63 * *(k3+ii)
      + a64 * *(k4+ii) + a65 * *(k5+ii);
  }
  //calculate k6
  func(*to+dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++){
    *(k6+ii)= dt * *(a+ii);
    *(yn1+ii) += b6* *(k6+ii);
    *(yerr+ii) += be6* *(k6+ii);
    *(ynk+ii)= *(yo+ii) + a71 * *(k1+ii)
      + a73 * *(k3+ii) //a72=0
      + a74 * *(k4+ii) + a75 * *(k5+ii)
      + a76 * *(k6+ii);
  }
  //calculate k7
  func(*to+dt,ynk,a,nargs,potentialArgs);
  for (ii=0; ii < dim; ii++) *(yerr+ii) += be7 * dt * *(a+ii);
  //yn1 is proposed new value
  //find maximum values
  double max_val= fabs(*yo);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(yo+ii)) > max_val )
      max_val= fabs(*(yo+ii));
  //set up scale
  double c= fmax(atol, rtol * max_val);
  double s= log(exp(atol-c)+exp(rtol*max_val-c))+c;
  //Norm
  double err= 0.;
  for (ii=0; ii < dim; ii++)
    err+= exp(2.*log(fabs(*(yerr+ii)))-2.* s);
  err= sqrt(err/dim);
  double corr= 0.85*pow(err,-.2);
  //Round to the nearest power of two
  double powertwo= round(log(corr)/log(2.));
  if ( powertwo > _MAX_STEPCHANGE_POWERTWO )
    powertwo= _MAX_STEPCHANGE_POWERTWO;
  else if ( powertwo < _MIN_STEPCHANGE_POWERTWO )
    powertwo= _MIN_STEPCHANGE_POWERTWO;
  //printf("%f,%f\n",powertwo,err);
  //fflush(stdout);
  //accept or reject
  double dt_one;
  if ( ( powertwo >= 0. ) || accept ) {//accept, if the step is the smallest possible, always accept
    for (ii= 0; ii < dim; ii++) {
      *(a1+ii)= *(a+ii);
      *(yo+ii)= *(yn1+ii);
    }
    *to+= dt;
    //printf("%f,%f\n",*to,dt);
  }
  dt_one= dt*pow(2.,powertwo);
  return dt_one;
}
