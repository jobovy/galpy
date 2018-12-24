/*
C implementations of symplectic integrators
 */
/*
Copyright (c) 2011, 2018 Jo Bovy
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
#include <stdbool.h>
#include <math.h>
#include <bovy_symplecticode.h>
#define _MAX_DT_REDUCE 10000.
#include "signal.h"
volatile sig_atomic_t interrupted= 0;

// handle CTRL-C differently on UNIX systems and Windows
#ifndef _WIN32
void handle_sigint(int signum)
{
  interrupted= 1;
}
#else
#include <windows.h>
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
    // Handle the CTRL-C signal.
    case CTRL_C_EVENT:
        interrupted= 1;
        // needed to avoid other control handlers like python from being called before us
        return TRUE;
    default:
        return FALSE;
    }
}

#endif
static inline void save_result(int dim, double *yo, double *result,
			       bool construct_Lz){
  int ii;
  for (ii=0; ii < dim; ii++) *(result+ii)= *(yo+ii);
  if ( construct_Lz ) 
    *(result+2) /= *result;
}
/*
Leapfrog integrator
Usage:
   Provide the drift function 'drift' with calling sequence
       drift(dt,t)
   and the kick function 'kick' with calling sequence
       kick(dt,t,*y,nargs,* potentialArgs)
   that move the system by a set dt where
       double dt: time step
       double t: time
       double * y: current phase-space position
       int nargs: number of arguments the function takes
       struct potentialArg * potentialArg structure pointer, see header file
  Other arguments are:
       int dim: dimension
       double *yo: initial value, dimension: dim
       int nt: number of times at which the output is wanted
       double dt: (optional) stepsize to use, must be an integer divisor of time difference between output steps (NOT CHECKED EXPLICITLY)
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       struct potentialArg * potentialArgs: see above
       double rtol, double atol: relative and absolute tolerance levels desired
       void (*tol_scaling)(double *y,double * result): function that computes the scaling to use in relative/absolute tolerance combination: scale= atol+rtol*scaling
       void (*metric)(int dim,double *x, double *y,double * result): function that computes the distance between two phase-space positions x and y and stores this in result
       bool construct_Lz: if true, construct Lz = yo[0] * yo[2] for integration in cylindrical/polar coordinates (output is still vT, not Lz)
  Output:
       double *result: result (nt blocks of size dim)
       int *err: error: -10 if interrupted by CTRL-C (SIGINT)
*/
void leapfrog(void (*drift)(double dt, double *y),
	      void  (*kick)(double dt, double t, double *y,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      void (*tol_scaling)(double *yo,double * result),
	      void (*metric)(int dim,double *x, double *y,double * result),
	      bool construct_Lz,
	      double *result,int * err){
  //Initialize
  int ii, jj;
  save_result(dim,yo,result,false);
  result+= dim;
  *err= 0;
  if ( construct_Lz ) *(yo+2) *= *yo;
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= leapfrog_estimate_step(*drift,*kick,dim,yo,init_dt,t,
			       nargs,potentialArgs,rtol,atol,
			       tol_scaling,metric);
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
    if (SetConsoleCtrlHandler(CtrlHandler, TRUE)){}
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0; // need to reset, bc library and vars stay in memory
      break;
    }
    //drift half
    drift(dt/2.,yo);
    //now drift full for a while
    for (jj=0; jj < (ndt-1); jj++){
      kick(dt,to+dt/2.,yo,nargs,potentialArgs);
      drift(dt,yo);
      to= to+dt;
    }
    //end with one last kick and half drift
    kick(dt,to+dt/2.,yo,nargs,potentialArgs);
    drift(dt/2.,yo);
    to= to+dt;
    //save
    save_result(dim,yo,result,construct_Lz);
    result+= dim;
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //We're done
}

/*
Fourth order symplectic integrator from Kinoshika et al.
Usage:
   Provide the drift function 'drift' with calling sequence
       drift(dt,t)
   and the kick function 'kick' with calling sequence
       kick(dt,t,*y,nargs,* potentialArgs)
   that move the system by a set dt where
       double dt: time step
       double t: time
       double * y: current phase-space position
       int nargs: number of arguments the function takes
       struct potentialArg * potentialArg structure pointer, see header file
  Other arguments are:
       int dim: dimension
       double *yo: initial value, dimension: dim
       int nt: number of times at which the output is wanted
       double dt: (optional) stepsize to use, must be an integer divisor of time difference between output steps (NOT CHECKED EXPLICITLY)
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       struct potentialArg * potentialArgs: see above
       double rtol, double atol: relative and absolute tolerance levels desired
       void (*tol_scaling)(double *y,double * result): function that computes the scaling to use in relative/absolute tolerance combination: scale= atol+rtol*scaling
       void (*metric)(int dim,double *x, double *y,double * result): function that computes the distance between two phase-space positions x and y and stores this in result
       bool construct_Lz: if true, construct Lz = yo[0] * yo[2] for integration in cylindrical/polar coordinates (output is still vT, not Lz)
  Output:
       double *result: result (nt blocks of size dim)
       int *err: error: -10 if interrupted by CTRL-C (SIGINT)
*/
void symplec4(void (*drift)(double dt, double *y),
	      void  (*kick)(double dt, double t, double *y,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      void (*tol_scaling)(double *yo,double * result),
	      void (*metric)(int dim,double *x, double *y,double * result),
	      bool construct_Lz,
	      double *result,int * err){
  //coefficients
  double c1= 0.6756035959798289;
  double c4= c1;
  double c41= c4+c1;
  double c2= -0.1756035959798288;
  double c3= c2;
  double d1= 1.3512071919596578;
  double d3= d1;
  double d2= -1.7024143839193153; //d4=0
  //Initialize
  int ii, jj;
  save_result(dim,yo,result,false);
  result+= dim;
  *err= 0;
  if ( construct_Lz ) *(yo+2) *= *yo;
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= symplec4_estimate_step(*drift,*kick,dim,yo,init_dt,t,
			       nargs,potentialArgs,rtol,atol,
			       tol_scaling,metric);
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
      break;
    }
    //drift for c1*dt
    drift(c1*dt,yo);
    to+= c1*dt;
    //steps ignoring q4/p4 when output is not wanted
    for (jj=0; jj < (ndt-1); jj++){
      kick(d1*dt,to,yo,nargs,potentialArgs);
      drift(c2*dt,yo);
      to+= c2*dt;
      kick(d2*dt,to,yo,nargs,potentialArgs);
      drift(c3*dt,yo);
      to+= c3*dt;
      kick(d3*dt,to,yo,nargs,potentialArgs);
      //drift for (c4+c1)*dt
      drift(c41*dt,yo);
      to+= c41*dt;
    }
    //steps not ignoring q4/p4 when output is wanted
    kick(d1*dt,to,yo,nargs,potentialArgs);
    drift(c2*dt,yo);
    to+= c2*dt;
    kick(d2*dt,to,yo,nargs,potentialArgs);
    drift(c3*dt,yo);
    to+= c3*dt;
    kick(d3*dt,to,yo,nargs,potentialArgs);
    drift(c4*dt,yo);
    to+= c4*dt;
    //save
    save_result(dim,yo,result,construct_Lz);
    result+= dim;
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //We're done
}

/*
Sixth order symplectic integrator from Kinoshika et al., Yoshida (1990)
Usage:
   Provide the drift function 'drift' with calling sequence
       drift(dt,t)
   and the kick function 'kick' with calling sequence
       kick(dt,t,*y,nargs,* potentialArgs)
   that move the system by a set dt where
       double dt: time step
       double t: time
       double * y: current phase-space position
       int nargs: number of arguments the function takes
       struct potentialArg * potentialArg structure pointer, see header file
  Other arguments are:
       int dim: dimension
       double *yo: initial value, dimension: dim
       int nt: number of times at which the output is wanted
       double dt: (optional) stepsize to use, must be an integer divisor of time difference between output steps (NOT CHECKED EXPLICITLY)
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       struct potentialArg * potentialArgs: see above
       double rtol, double atol: relative and absolute tolerance levels desired
       void (*tol_scaling)(double *y,double * result): function that computes the scaling to use in relative/absolute tolerance combination: scale= atol+rtol*scaling
       void (*metric)(int dim,double *x, double *y,double * result): function that computes the distance between two phase-space positions x and y and stores this in result
       bool construct_Lz: if true, construct Lz = yo[0] * yo[2] for integration in cylindrical/polar coordinates (output is still vT, not Lz)
  Output:
       double *result: result (nt blocks of size dim)
       int *err: error: -10 if interrupted by CTRL-C (SIGINT)
*/
void symplec6(void (*drift)(double dt, double *y),
	      void  (*kick)(double dt, double t, double *y,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      void (*tol_scaling)(double *yo,double * result),
	      void (*metric)(int dim,double *x, double *y,double * result),
	      bool construct_Lz,
	      double *result,int * err){
  //coefficients
  double c1= 0.392256805238780;
  double c8= c1;
  double c81= c8+c1;
  double c2= 0.510043411918458;
  double c7= c2;
  double c3= -0.471053385409758;
  double c6= c3;
  double c4= 0.687531682525198e-1;
  double c5= c4;
  double d1= 0.784513610477560;
  double d7= d1;
  double d2= 0.235573213359357;
  double d6= d2;
  double d3= -0.117767998417887e1;
  double d5= d3;
  double d4= 0.131518632068391e1; //d8=0
  //Initialize
  int ii, jj;
  save_result(dim,yo,result,false);
  result+= dim;
  *err= 0;
  if ( construct_Lz ) *(yo+2) *= *yo;
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= symplec6_estimate_step(*drift,*kick,dim,yo,init_dt,t,
			       nargs,potentialArgs,rtol,atol,
			       tol_scaling,metric);
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
      break;
    }
    drift(c1*dt,yo);
    to+= c1*dt;
    //steps ignoring q8/p8 when output is not wanted
    for (jj=0; jj < (ndt-1); jj++){
      kick(d1*dt,to,yo,nargs,potentialArgs);
      drift(c2*dt,yo);
      to+= c2*dt;
      kick(d2*dt,to,yo,nargs,potentialArgs);
      drift(c3*dt,yo);
      to+= c3*dt;
      kick(d3*dt,to,yo,nargs,potentialArgs);
      drift(c4*dt,yo);
      to+= c4*dt;
      kick(d4*dt,to,yo,nargs,potentialArgs);
      drift(c5*dt,yo);
      to+= c5*dt;
      kick(d5*dt,to,yo,nargs,potentialArgs);
      drift(c6*dt,yo);
      to+= c6*dt;
      kick(d6*dt,to,yo,nargs,potentialArgs);
      drift(c7*dt,yo);
      to+= c7*dt;
      kick(d7*dt,to,yo,nargs,potentialArgs);
      drift(c81*dt,yo);
      to+= c81*dt;
    }
    //steps not ignoring q8/p8 when output is wanted
    kick(d1*dt,to,yo,nargs,potentialArgs);
    drift(c2*dt,yo);
    to+= c2*dt;
    kick(d2*dt,to,yo,nargs,potentialArgs);
    drift(c3*dt,yo);
    to+= c3*dt;
    kick(d3*dt,to,yo,nargs,potentialArgs);
    drift(c4*dt,yo);
    to+= c4*dt;
    kick(d4*dt,to,yo,nargs,potentialArgs);
    drift(c5*dt,yo);
    to+= c5*dt;
    kick(d5*dt,to,yo,nargs,potentialArgs);
    drift(c6*dt,yo);
    to+= c6*dt;
    kick(d6*dt,to,yo,nargs,potentialArgs);
    drift(c7*dt,yo);
    to+= c7*dt;
    kick(d7*dt,to,yo,nargs,potentialArgs);
    drift(c8*dt,yo);
    to+= c8*dt;
    //save
    save_result(dim,yo,result,construct_Lz);
    result+= dim;
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //We're done
}

double leapfrog_estimate_step(void (*drift)(double dt, double *y),
			      void  (*kick)(double dt, double t, double *y,
					  int nargs,
					  struct potentialArg * potentialArgs),
			      int dim, double *yo,
			      double dt, double *t,
			      int nargs,struct potentialArg * potentialArgs,
			      double rtol,double atol,
			      void (*tol_scaling)(double * yo,double * result),
			      void (*metric)(int dim,double *x, 
					     double *y,double * result)){
  int ii;
  //scalars
  double err= 2.;
  double to= *t;
  double init_dt= dt;
  //allocate and initialize
  double *y11= (double *) malloc ( dim * sizeof(double) );
  double *y12= (double *) malloc ( dim * sizeof(double) );
  double *delta= (double *) malloc ( dim * sizeof(double) );
  //set up scale
  double *scaling= (double *) malloc ( dim * sizeof(double) );
  double *scale2= (double *) malloc ( dim * sizeof(double) );
  tol_scaling(yo,scaling);
  for (ii=0; ii < dim; ii++)
    *(scale2+ii) = pow ( exp(atol) + exp(rtol) * *(scaling+ii), 2);
  //find good dt
  dt*= 2.;
  while ( err > 1.  && init_dt / dt < _MAX_DT_REDUCE){
    dt/= 2.;
    // Reset to initial condition
    for (ii=0; ii < dim; ii++) {
      *(y11+ii)= *(yo+ii);
      *(y12+ii)= *(yo+ii);
    }
    //do one leapfrog step with step dt, and one with step dt/2.
    //dt
    drift(dt/2.,y11);
    kick(dt,to+dt/2.,y11,nargs,potentialArgs);
    drift(dt/2.,y11);
    //dt/2.
    drift(dt/4.,y12);
    kick(dt/2.,to+dt/4.,y12,nargs,potentialArgs);
    drift(dt/2.,y12);//Take full step combining two half
    kick(dt/2.,to+3.*dt/4.,y12,nargs,potentialArgs);
    drift(dt/4.,y12);
    //Norm
    metric(dim,y11,y12,delta);
    err= 0.;
    for (ii=0; ii < dim; ii++)
      err+= *(delta+ii) * *(delta+ii) / *(scale2+ii);
    err= sqrt(err/dim);
  }
  //free what we allocated
  free(y11);
  free(y12);
  free(delta);
  free(scaling);
  free(scale2);
  return dt;
}

double symplec4_estimate_step(void (*drift)(double dt, double *y),
			      void  (*kick)(double dt, double t, double *y,
					  int nargs,
					  struct potentialArg * potentialArgs),
			      int dim, double *yo,
			      double dt, double *t,
			      int nargs,struct potentialArg * potentialArgs,
			      double rtol,double atol,
			      void (*tol_scaling)(double * yo,double * result),
			      void (*metric)(int dim,double *x, 
					     double *y,double * result)){
  //coefficients
  double c1= 0.6756035959798289;
  double c4= c1;
  double c41= c4+c1;
  double c2= -0.1756035959798288;
  double c3= c2;
  double d1= 1.3512071919596578;
  double d3= d1;
  double d2= -1.7024143839193153; //d4=0
  //scalars
  int ii;
  double err= 2.;
  double to;
  double init_dt= dt;
  double dt2;
  //allocate and initialize
  double *y11= (double *) malloc ( dim * sizeof(double) );
  double *y12= (double *) malloc ( dim * sizeof(double) );
  double *delta= (double *) malloc ( dim * sizeof(double) );
  //set up scale
  double *scaling= (double *) malloc ( dim * sizeof(double) );
  double *scale2= (double *) malloc ( dim * sizeof(double) );
  tol_scaling(yo,scaling);
  for (ii=0; ii < dim; ii++)
    *(scale2+ii) = pow ( exp(atol) + exp(rtol) * *(scaling+ii), 2);
  //find good dt
  dt*= 2.;
  while ( err > 1. && init_dt / dt < _MAX_DT_REDUCE ){
    dt/= 2.;
    // Reset to initial condition
    to= *t;
    for (ii=0; ii < dim; ii++) {
      *(y11+ii)= *(yo+ii);
      *(y12+ii)= *(yo+ii);
    }
    //do one step with step dt, and one with step dt/2.
    /*
      dt
    */
    drift(c1*dt,y11);
    to+= c1*dt;
    kick(d1*dt,to,y11,nargs,potentialArgs);
    drift(c2*dt,y11);
    to+= c2*dt;
    kick(d2*dt,to,y11,nargs,potentialArgs);
    drift(c3*dt,y11);
    to+= c3*dt;
    kick(d3*dt,to,y11,nargs,potentialArgs);
    drift(c4*dt,y11);
    //reset
    to-= dt;   
    /*
      dt/2
    */
    dt2= dt/2.;
    drift(c1*dt2,y12);
    to+= c1*dt2;
    kick(d1*dt2,to,y12,nargs,potentialArgs);
    drift(c2*dt2,y12);
    to+= c2*dt2;
    kick(d2*dt2,to,y12,nargs,potentialArgs);
    drift(c3*dt2,y12);
    to+= c3*dt2;
    kick(d3*dt2,to,y12,nargs,potentialArgs);
    drift(c41*dt2,y12);
    to+= c41*dt2;
    kick(d1*dt2,to,y12,nargs,potentialArgs);
    drift(c2*dt2,y12);
    to+= c2*dt2;
    kick(d2*dt2,to,y12,nargs,potentialArgs);
    drift(c3*dt2,y12);
    to+= c3*dt2;
    kick(d3*dt2,to,y12,nargs,potentialArgs);
    drift(c4*dt2,y12);
    //Norm
    metric(dim,y11,y12,delta);
    for (ii=0; ii < dim; ii++)
      err+= *(delta+ii) * *(delta+ii) / *(scale2+ii);
    err= sqrt(err/dim);
  }
  //free what we allocated
  free(y11);
  free(y12);
  free(delta);
  free(scaling);
  free(scale2);
  return dt;
}

double symplec6_estimate_step(void (*drift)(double dt, double *y),
			      void  (*kick)(double dt, double t, double *y,
					  int nargs,
					  struct potentialArg * potentialArgs),
			      int dim, double *yo,
			      double dt, double *t,
			      int nargs,struct potentialArg * potentialArgs,
			      double rtol,double atol,
			      void (*tol_scaling)(double * yo,double * result),
			      void (*metric)(int dim,double *x, 
					     double *y,double * result)){
  //coefficients
  double c1= 0.392256805238780;
  double c8= c1;
  double c81= c8+c1;
  double c2= 0.510043411918458;
  double c7= c2;
  double c3= -0.471053385409758;
  double c6= c3;
  double c4= 0.687531682525198e-1;
  double c5= c4;
  double d1= 0.784513610477560;
  double d7= d1;
  double d2= 0.235573213359357;
  double d6= d2;
  double d3= -0.117767998417887e1;
  double d5= d3;
  double d4= 0.131518632068391e1; //d8=0
  //scalars
  int ii;
  double err= 2.;
  double to;
  double init_dt= dt;
  double dt2;
  //allocate and initialize
  double *y11= (double *) malloc ( dim * sizeof(double) );
  double *y12= (double *) malloc ( dim * sizeof(double) );
  double *delta= (double *) malloc ( dim * sizeof(double) );
  //set up scale
  double *scaling= (double *) malloc ( dim * sizeof(double) );
  double *scale2= (double *) malloc ( dim * sizeof(double) );
  tol_scaling(yo,scaling);
  for (ii=0; ii < dim; ii++)
    *(scale2+ii) = pow ( exp(atol) + exp(rtol) * *(scaling+ii), 2);
  //find good dt
  dt*= 2.;
  while ( err > 1. && init_dt / dt < _MAX_DT_REDUCE ){
    dt/= 2.;
    // Reset to initial condition
    to= *t;
    for (ii=0; ii < dim; ii++) {
      *(y11+ii)= *(yo+ii);
      *(y12+ii)= *(yo+ii);
    }
    //do one step with step dt, and one with step dt/2.
    /*
      dt
    */
    drift(c1*dt,y11);
    to+= c1*dt;
    kick(d1*dt,to,y11,nargs,potentialArgs);
    drift(c2*dt,y11);
    to+= c2*dt;
    kick(d2*dt,to,y11,nargs,potentialArgs);
    drift(c3*dt,y11);
    to+= c3*dt;
    kick(d3*dt,to,y11,nargs,potentialArgs);
    drift(c4*dt,y11);
    to+= c4*dt;
    kick(d4*dt,to,y11,nargs,potentialArgs);
    drift(c5*dt,y11);
    to+= c5*dt;
    kick(d5*dt,to,y11,nargs,potentialArgs);
    drift(c6*dt,y11);
    to+= c6*dt;
    kick(d6*dt,to,y11,nargs,potentialArgs);
    drift(c7*dt,y11);
    to+= c7*dt;
    kick(d7*dt,to,y11,nargs,potentialArgs);
    drift(c8*dt,y11);
    to+= c8*dt;
    //reset
    to-= dt;   
    /*
      dt/2
    */
    dt2= dt/2.;
    drift(c1*dt2,y12);
    to+= c1*dt2;
    kick(d1*dt2,to,y12,nargs,potentialArgs);
    drift(c2*dt2,y12);
    to+= c2*dt2;
    kick(d2*dt2,to,y12,nargs,potentialArgs);
    drift(c3*dt2,y12);
    to+= c3*dt2;
    kick(d3*dt2,to,y12,nargs,potentialArgs);
    drift(c4*dt2,y12);
    to+= c4*dt2;
    kick(d4*dt2,to,y12,nargs,potentialArgs);
    drift(c5*dt2,y12);
    to+= c5*dt2;
    kick(d5*dt2,to,y12,nargs,potentialArgs);
    drift(c6*dt2,y12);
    to+= c6*dt2;
    kick(d6*dt2,to,y12,nargs,potentialArgs);
    drift(c7*dt2,y12);
    to+= c7*dt2;
    kick(d7*dt2,to,y12,nargs,potentialArgs);
    drift(c81*dt2,y12);
    to+= c81*dt2;
    kick(d1*dt2,to,y12,nargs,potentialArgs);
    drift(c2*dt2,y12);
    to+= c2*dt2;
    kick(d2*dt2,to,y12,nargs,potentialArgs);
    drift(c3*dt2,y12);
    to+= c3*dt2;
    kick(d3*dt2,to,y12,nargs,potentialArgs);
    drift(c4*dt2,y12);
    to+= c4*dt2;
    kick(d4*dt2,to,y12,nargs,potentialArgs);
    drift(c5*dt2,y12);
    to+= c5*dt2;
    kick(d5*dt2,to,y12,nargs,potentialArgs);
    drift(c6*dt2,y12);
    to+= c6*dt2;
    kick(d6*dt2,to,y12,nargs,potentialArgs);
    drift(c7*dt2,y12);
    to+= c7*dt2;
    kick(d7*dt2,to,y12,nargs,potentialArgs);
    drift(c8*dt2,y12);
    to+= c8*dt2;
    //Norm 
    metric(dim,y11,y12,delta);
    for (ii=0; ii < dim; ii++)
      err+= *(delta+ii) * *(delta+ii) / *(scale2+ii);
    err= sqrt(err/dim);
  }
  //free what we allocated
  free(y11);
  free(y12);
  free(delta);
  free(scaling);
  free(scale2);
  return dt;
}
