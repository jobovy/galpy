/*
C implementations of symplectic integrators
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
#define _MAX_DT_REDUCE 10000.
//CTRL-C only works on UNIX systems with signal library
#ifndef _WIN32
#include "signal.h"
volatile sig_atomic_t interrupted= 0;
void handle_sigint(int signum)
{
  interrupted= 1;
}
#else
int interrupted= 0;
#endif
static inline void leapfrog_leapq(int dim, double *q,double *p,double dt,
				  double *qn){
  int ii;
  for (ii=0; ii < dim; ii++) (*qn++)= (*q++) +dt * (*p++);
}
static inline void leapfrog_leapp(int dim, double *p,double dt,double *a,
				  double *pn){
  int ii;
  for (ii=0; ii< dim; ii++) (*pn++)= (*p++) + dt * (*a++);
}

static inline void save_qp(int dim, double *qo, double *po, double *result){
  int ii;
  for (ii=0; ii < dim; ii++) *result++= *qo++;
  for (ii=0; ii < dim; ii++) *result++= *po++;
}
/*
Leapfrog integrator
Usage:
   Provide the acceleration function func with calling sequence
       func (t,q,a,nargs,args)
   where
       double t: time
       double * q: current position (dimension: dim)
       double * a: will be set to the derivative
       int nargs: number of arguments the function takes
       struct potentialArg * potentialArg structure pointer, see header file
  Other arguments are:
       int dim: dimension
       double *yo: initial value [qo,po], dimension: 2*dim
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
void leapfrog(void (*func)(double t, double *q, double *a,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      double *result,int * err){
  //Initialize
  double *qo= (double *) malloc ( dim * sizeof(double) );
  double *po= (double *) malloc ( dim * sizeof(double) );
  double *q12= (double *) malloc ( dim * sizeof(double) );
  double *p12= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  int ii, jj, kk;
  for (ii=0; ii < dim; ii++) {
    *qo++= *(yo+ii);
    *po++= *(yo+dim+ii);
  }
  qo-= dim;
  po-= dim;
  save_qp(dim,qo,po,result);
  result+= 2 * dim;
  *err= 0;
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= leapfrog_estimate_step(*func,dim,qo,po,init_dt,t,nargs,potentialArgs,
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
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0; // need to reset, bc library and vars stay in memory
      break;
    }
    //drift half
    leapfrog_leapq(dim,qo,po,dt/2.,q12);
    //now drift full for a while
    for (jj=0; jj < (ndt-1); jj++){
      //kick
      func(to+dt/2.,q12,a,nargs,potentialArgs);
      leapfrog_leapp(dim,po,dt,a,p12);
      //drift
      leapfrog_leapq(dim,q12,p12,dt,qo);
      //reset
      to= to+dt;
      for (kk=0; kk < dim; kk++) {
	*(q12+kk)= *(qo+kk);
	*(po+kk)= *(p12+kk);
      }
    }
    //end with one last kick and drift
    //kick
    func(to+dt/2.,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,dt,a,po);
    //drift
    leapfrog_leapq(dim,q12,po,dt/2.,qo);
    to= to+dt;
    //save
    save_qp(dim,qo,po,result);
    result+= 2 * dim;
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //Free allocated memory
  free(qo);
  free(po);
  free(q12);
  free(a);
  //We're done
}

/*
Fourth order symplectic integrator from Kinoshika et al.
Usage:
   Provide the acceleration function func with calling sequence
       func (t,q,a,nargs,args)
   where
       double t: time
       double * q: current position (dimension: dim)
       double * a: will be set to the derivative
       int nargs: number of arguments the function takes
       struct potentialArg * potentialArg structure pointer, see header file
  Other arguments are:
       int dim: dimension
       double *yo: initial value [qo,po], dimension: 2*dim
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
void symplec4(void (*func)(double t, double *q, double *a,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      double *result,int * err){
  //coefficients
  double c1= 0.6756035959798289;
  double c4= c1;
  double c2= -0.1756035959798288;
  double c3= c2;
  double d1= 1.3512071919596578;
  double d3= d1;
  double d2= -1.7024143839193153; //d4=0
  //Initialize
  double *qo= (double *) malloc ( dim * sizeof(double) );
  double *po= (double *) malloc ( dim * sizeof(double) );
  double *q12= (double *) malloc ( dim * sizeof(double) );
  double *p12= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  int ii, jj, kk;
  for (ii=0; ii < dim; ii++) {
    *qo++= *(yo+ii);
    *po++= *(yo+dim+ii);
  }
  qo-= dim;
  po-= dim;
  save_qp(dim,qo,po,result);
  result+= 2 * dim;
  *err= 0;
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= symplec4_estimate_step(*func,dim,qo,po,init_dt,t,nargs,potentialArgs,
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
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0; // need to reset, bc library and vars stay in memory
      break;
    }
    //drift for c1*dt
    leapfrog_leapq(dim,qo,po,c1*dt,q12);
    to+= c1*dt;
    //steps ignoring q4/p4 when output is not wanted
    for (jj=0; jj < (ndt-1); jj++){
      //kick for d1*dt
      func(to,q12,a,nargs,potentialArgs);
      leapfrog_leapp(dim,po,d1*dt,a,p12);
      //drift for c2*dt
      leapfrog_leapq(dim,q12,p12,c2*dt,qo);
      //kick for d2*dt
      to+= c2*dt;
      func(to,qo,a,nargs,potentialArgs);
      leapfrog_leapp(dim,p12,d2*dt,a,po);
      //drift for c3*dt
      leapfrog_leapq(dim,qo,po,c3*dt,q12);
      to+= c3*dt;
      //kick for d3*dt
      func(to,q12,a,nargs,potentialArgs);
      leapfrog_leapp(dim,po,d3*dt,a,p12);
      //drift for (c4+c1)*dt
      leapfrog_leapq(dim,q12,p12,(c4+c1)*dt,qo);
      to+= (c4+c1)*dt;
      //reset
      for (kk=0; kk < dim; kk++) {
	*(q12+kk)= *(qo+kk);
	*(po+kk)= *(p12+kk);
      }
    }
    //steps not ignoring q4/p4 when output is wanted
    //kick for d1*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d1*dt,a,p12);
    //drift for c2*dt
    leapfrog_leapq(dim,q12,p12,c2*dt,qo);
    //kick for d2*dt
    to+= c2*dt;
    func(to,qo,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d2*dt,a,po);
    //drift for c3*dt
    leapfrog_leapq(dim,qo,po,c3*dt,q12);
    to+= c3*dt;
    //kick for d3*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d3*dt,a,p12);
    //drift for c4*dt
    leapfrog_leapq(dim,q12,p12,c4*dt,qo);
    to+= c4*dt;
    //p4=p3
    for (kk=0; kk < dim; kk++) *(po+kk)= *(p12+kk);
    //save
    save_qp(dim,qo,po,result);
    result+= 2 * dim;
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //Free allocated memory
  free(qo);
  free(po);
  free(q12);
  free(a);
  //We're done
}

/*
Sixth order symplectic integrator from Kinoshika et al., Yoshida (1990)
Usage:
   Provide the acceleration function func with calling sequence
       func (t,q,a,nargs,args)
   where
       double t: time
       double * q: current position (dimension: dim)
       double * a: will be set to the derivative
       int nargs: number of arguments the function takes
       struct potentialArg * potentialArg structure pointer, see header file
  Other arguments are:
       int dim: dimension
       double *yo: initial value [qo,po], dimension: 2*dim
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
void symplec6(void (*func)(double t, double *q, double *a,
			   int nargs, struct potentialArg * potentialArgs),
	      int dim,
	      double * yo,
	      int nt, double dt, double *t,
	      int nargs, struct potentialArg * potentialArgs,
	      double rtol, double atol,
	      double *result,int * err){
  //coefficients
  double c1= 0.392256805238780;
  double c8= c1;
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
  double *qo= (double *) malloc ( dim * sizeof(double) );
  double *po= (double *) malloc ( dim * sizeof(double) );
  double *q12= (double *) malloc ( dim * sizeof(double) );
  double *p12= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  int ii, jj, kk;
  for (ii=0; ii < dim; ii++) {
    *qo++= *(yo+ii);
    *po++= *(yo+dim+ii);
  }
  qo-= dim;
  po-= dim;
  save_qp(dim,qo,po,result);
  result+= 2 * dim;
  *err= 0;
  //Estimate necessary stepsize
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= symplec6_estimate_step(*func,dim,qo,po,init_dt,t,nargs,potentialArgs,
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
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0; // need to reset, bc library and vars stay in memory
      break;
    }
    //drift for c1*dt
    leapfrog_leapq(dim,qo,po,c1*dt,q12);
    to+= c1*dt;
    //steps ignoring q8/p8 when output is not wanted
    for (jj=0; jj < (ndt-1); jj++){
      //kick for d1*dt
      func(to,q12,a,nargs,potentialArgs);
      leapfrog_leapp(dim,po,d1*dt,a,p12);
      //drift for c2*dt
      leapfrog_leapq(dim,q12,p12,c2*dt,qo);
      to+= c2*dt;
      //kick for d2*dt
      func(to,qo,a,nargs,potentialArgs);
      leapfrog_leapp(dim,p12,d2*dt,a,po);
      //drift for c3*dt
      leapfrog_leapq(dim,qo,po,c3*dt,q12);
      to+= c3*dt;
      //kick for d3*dt
      func(to,q12,a,nargs,potentialArgs);
      leapfrog_leapp(dim,po,d3*dt,a,p12);
      //drift for c4*dt
      leapfrog_leapq(dim,q12,p12,c4*dt,qo);
      //kick for d4*dt
      to+= c4*dt;
      func(to,qo,a,nargs,potentialArgs);
      leapfrog_leapp(dim,p12,d4*dt,a,po);
      //drift for c5*dt
      leapfrog_leapq(dim,qo,po,c5*dt,q12);
      to+= c5*dt;
      //kick for d5*dt
      func(to,q12,a,nargs,potentialArgs);
      leapfrog_leapp(dim,po,d5*dt,a,p12);
      //drift for c6*dt
      leapfrog_leapq(dim,q12,p12,c6*dt,qo);
      //kick for d6*dt
      to+= c6*dt;
      func(to,qo,a,nargs,potentialArgs);
      leapfrog_leapp(dim,p12,d6*dt,a,po);
      //drift for c7*dt
      leapfrog_leapq(dim,qo,po,c7*dt,q12);
      to+= c7*dt;
      //kick for d7*dt
      func(to,q12,a,nargs,potentialArgs);
      leapfrog_leapp(dim,po,d7*dt,a,p12);
      //drift for (c8+c1)*dt
      leapfrog_leapq(dim,q12,p12,(c8+c1)*dt,qo);
      to+= (c8+c1)*dt;
      //reset
      for (kk=0; kk < dim; kk++) {
	*(q12+kk)= *(qo+kk);
	*(po+kk)= *(p12+kk);
      }
    }
    //steps not ignoring q8/p8 when output is wanted
    //kick for d1*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d1*dt,a,p12);
    //drift for c2*dt
    leapfrog_leapq(dim,q12,p12,c2*dt,qo);
    to+= c2*dt;
    //kick for d2*dt
    func(to,qo,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d2*dt,a,po);
    //drift for c3*dt
    leapfrog_leapq(dim,qo,po,c3*dt,q12);
    to+= c3*dt;
    //kick for d3*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d3*dt,a,p12);
    //drift for c4*dt
    leapfrog_leapq(dim,q12,p12,c4*dt,qo);
    to+= c4*dt;
    //kick for d4*dt
    func(to,qo,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d4*dt,a,po);
    //drift for c5*dt
    leapfrog_leapq(dim,qo,po,c5*dt,q12);
    to+= c5*dt;
    //kick for d5*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d5*dt,a,p12);
    //drift for c6*dt
    leapfrog_leapq(dim,q12,p12,c6*dt,qo);
    //kick for d6*dt
    to+= c6*dt;
    func(to,qo,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d6*dt,a,po);
    //drift for c7*dt
    leapfrog_leapq(dim,qo,po,c7*dt,q12);
    to+= c7*dt;
    //kick for d7*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d7*dt,a,p12);
    //drift for c8*dt
    leapfrog_leapq(dim,q12,p12,c8*dt,qo);
    to+= c8*dt;
    //p8=p7
    for (kk=0; kk < dim; kk++) *(po+kk)= *(p12+kk);
    //save
    save_qp(dim,qo,po,result);
    result+= 2 * dim;
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //Free allocated memory
  free(qo);
  free(po);
  free(q12);
  free(a);
  //We're done
}

double leapfrog_estimate_step(void (*func)(double t, double *q, double *a,int nargs, struct potentialArg *),
			      int dim, double *qo,double *po,
			      double dt, double *t,
			      int nargs,struct potentialArg * potentialArgs,
			      double rtol,double atol){
  //return dt;
  //scalars
  double err= 2.;
  double max_val_q, max_val_p;
  double to= *t;
  double init_dt= dt;
  //allocate and initialize
  double *q11= (double *) malloc ( dim * sizeof(double) );
  double *q12= (double *) malloc ( dim * sizeof(double) );
  double *p11= (double *) malloc ( dim * sizeof(double) );
  double *p12= (double *) malloc ( dim * sizeof(double) );
  double *qtmp= (double *) malloc ( dim * sizeof(double) );
  double *ptmp= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *scale= (double *) malloc ( 2 * dim * sizeof(double) );
  int ii;
  //find maximum values
  max_val_q= fabs(*qo);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(qo+ii)) > max_val_q )
      max_val_q= fabs(*(qo+ii));
  max_val_p= fabs(*po);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(po+ii)) > max_val_p )
      max_val_p= fabs(*(po+ii));
  //set up scale
  double c= fmax(atol, rtol * max_val_q);
  double s= log(exp(atol-c)+exp(rtol*max_val_q-c))+c;
  for (ii=0; ii < dim; ii++) *(scale+ii)= s;
  c= fmax(atol, rtol * max_val_p);
  s= log(exp(atol-c)+exp(rtol*max_val_p-c))+c;
  for (ii=0; ii < dim; ii++) *(scale+ii+dim)= s;
  //find good dt
  dt*= 2.;
  while ( err > 1.  && init_dt / dt < _MAX_DT_REDUCE){
    dt/= 2.;
    //do one leapfrog step with step dt, and one with step dt/2.
    //dt
    leapfrog_leapq(dim,qo,po,dt/2.,q12);
    func(to+dt/2.,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,dt,a,p11);
    leapfrog_leapq(dim,q12,p11,dt/2.,q11);
    //dt/2.
    leapfrog_leapq(dim,qo,po,dt/4.,q12);
    func(to+dt/4.,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,dt/2.,a,ptmp);
    leapfrog_leapq(dim,q12,ptmp,dt/2.,qtmp);//Take full step combining two half
    func(to+3.*dt/4.,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,dt/2.,a,p12);
    leapfrog_leapq(dim,qtmp,p12,dt/4.,q12);//Take full step combining two half   
    //Norm
    err= 0.;
    for (ii=0; ii < dim; ii++) {
      err+= exp(2.*log(fabs(*(q11+ii)-*(q12+ii)))-2.* *(scale+ii));
      err+= exp(2.*log(fabs(*(p11+ii)-*(p12+ii)))-2.* *(scale+ii+dim));
    }
    err= sqrt(err/2./dim);
  }
  //free what we allocated
  free(q11);
  free(q12);
  free(p11);
  free(qtmp);
  free(ptmp);
  free(a);
  free(scale);
  //return
  //printf("%f\n",dt);
  //fflush(stdout);
  return dt;
}

double symplec4_estimate_step(void (*func)(double t, double *q, double *a,int nargs, struct potentialArg *),
			      int dim, double *qo,double *po,
			      double dt, double *t,
			      int nargs,struct potentialArg * potentialArgs,
			      double rtol,double atol){
  //return dt;
  //coefficients
  double c1= 0.6756035959798289;
  double c4= c1;
  double c2= -0.1756035959798288;
  double c3= c2;
  double d1= 1.3512071919596578;
  double d3= d1;
  double d2= -1.7024143839193153; //d4=0
  //scalars
  double err= 2.;
  double max_val_q, max_val_p;
  double to= *t;
  double init_dt= dt;
  //allocate and initialize
  double *q11= (double *) malloc ( dim * sizeof(double) );
  double *q12= (double *) malloc ( dim * sizeof(double) );
  double *p11= (double *) malloc ( dim * sizeof(double) );
  double *p12= (double *) malloc ( dim * sizeof(double) );
  double *qtmp= (double *) malloc ( dim * sizeof(double) );
  double *ptmp= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *scale= (double *) malloc ( 2 * dim * sizeof(double) );
  int ii;
  //find maximum values
  max_val_q= fabs(*qo);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(qo+ii)) > max_val_q )
      max_val_q= fabs(*(qo+ii));
  max_val_p= fabs(*po);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(po+ii)) > max_val_p )
      max_val_p= fabs(*(po+ii));
  //set up scale
  double c= fmax(atol, rtol * max_val_q);
  double s= log(exp(atol-c)+exp(rtol*max_val_q-c))+c;
  for (ii=0; ii < dim; ii++) *(scale+ii)= s;
  c= fmax(atol, rtol * max_val_p);
  s= log(exp(atol-c)+exp(rtol*max_val_p-c))+c;
  for (ii=0; ii < dim; ii++) *(scale+ii+dim)= s;
  //find good dt
  dt*= 2.;
  while ( err > 1. && init_dt / dt < _MAX_DT_REDUCE ){
    dt/= 2.;
    //do one step with step dt, and one with step dt/2.
    /*
      dt
    */
    //drift for c1*dt
    leapfrog_leapq(dim,qo,po,c1*dt,q12);
    to+= c1*dt;
    //kick for d1*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d1*dt,a,p12);
    //drift for c2*dt
    leapfrog_leapq(dim,q12,p12,c2*dt,qtmp);
    //kick for d2*dt
    to+= c2*dt;
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d2*dt,a,ptmp);
    //drift for c3*dt
    leapfrog_leapq(dim,qtmp,ptmp,c3*dt,q12);
    to+= c3*dt;
    //kick for d3*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d3*dt,a,p11);
    //drift for c4*dt
    leapfrog_leapq(dim,q12,p11,c4*dt,q11);
    to+= c4*dt;
    //p4=p3
    //reset
    to-= dt;   
    /*
      dt/2
    */
    //drift for c1*dt/2
    leapfrog_leapq(dim,qo,po,c1*dt/2.,q12);
    to+= c1*dt/2.;
    //kick for d1*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d1*dt/2.,a,p12);
    //drift for c2*dt/2
    leapfrog_leapq(dim,q12,p12,c2*dt/2.,qtmp);
    //kick for d2*dt/2
    to+= c2*dt/2.;
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d2*dt/2.,a,ptmp);
    //drift for c3*dt/2
    leapfrog_leapq(dim,qtmp,ptmp,c3*dt/2.,q12);
    to+= c3*dt/2.;
    //kick for d3*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d3*dt/2.,a,p12);
    //drift for (c4+c1)*dt/2, skipping q4/p4
    leapfrog_leapq(dim,q12,p12,(c1+c4)*dt/2.,qtmp);
    to+= (c1+c4)*dt/2.;
    //kick for d1*dt/2
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d1*dt/2.,a,ptmp);
    //drift for c2*dt/2
    leapfrog_leapq(dim,qtmp,ptmp,c2*dt/2.,q12);
    //kick for d2*dt/2
    to+= c2*dt/2.;
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d2*dt/2.,a,p12);
    //drift for c3*dt/2
    leapfrog_leapq(dim,q12,p12,c3*dt/2.,qtmp);
    to+= c3*dt/2.;
    //kick for d3*dt
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d3*dt/2.,a,ptmp);
    //drift for c4*dt/2.
    leapfrog_leapq(dim,qtmp,ptmp,c4*dt/2.,q12);
    to+= c4*dt/2.;
    //p4=p3
    for (ii=0; ii < dim; ii++) *(p12+ii)= *(ptmp+ii);
    //Norm
    err= 0.;
    for (ii=0; ii < dim; ii++) {
      err+= exp(2.*log(fabs(*(q11+ii)-*(q12+ii)))-2.* *(scale+ii));
      err+= exp(2.*log(fabs(*(p11+ii)-*(p12+ii)))-2.* *(scale+ii+dim));
    }
    err= sqrt(err/2./dim);
    //reset
    to-= dt;
  }
  //free what we allocated
  free(q11);
  free(q12);
  free(p11);
  free(qtmp);
  free(ptmp);
  free(a);
  free(scale);
  //return
  //printf("%f\n",dt);
  //fflush(stdout);
  return dt;
}

double symplec6_estimate_step(void (*func)(double t, double *q, double *a,int nargs, struct potentialArg *),
			      int dim, double *qo,double *po,
			      double dt, double *t,
			      int nargs,struct potentialArg * potentialArgs,
			      double rtol,double atol){
  //return dt;
  //coefficients
  double c1= 0.392256805238780;
  double c8= c1;
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
  double err= 2.;
  double max_val_q, max_val_p;
  double to= *t;
  double init_dt= dt;
  //allocate and initialize
  double *q11= (double *) malloc ( dim * sizeof(double) );
  double *q12= (double *) malloc ( dim * sizeof(double) );
  double *p11= (double *) malloc ( dim * sizeof(double) );
  double *p12= (double *) malloc ( dim * sizeof(double) );
  double *qtmp= (double *) malloc ( dim * sizeof(double) );
  double *ptmp= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *scale= (double *) malloc ( 2 * dim * sizeof(double) );
  int ii;
  //find maximum values
  max_val_q= fabs(*qo);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(qo+ii)) > max_val_q )
      max_val_q= fabs(*(qo+ii));
  max_val_p= fabs(*po);
  for (ii=1; ii < dim; ii++)
    if ( fabs(*(po+ii)) > max_val_p )
      max_val_p= fabs(*(po+ii));
  //set up scale
  double c= fmax(atol, rtol * max_val_q);
  double s= log(exp(atol-c)+exp(rtol*max_val_q-c))+c;
  for (ii=0; ii < dim; ii++) *(scale+ii)= s;
  c= fmax(atol, rtol * max_val_p);
  s= log(exp(atol-c)+exp(rtol*max_val_p-c))+c;
  for (ii=0; ii < dim; ii++) *(scale+ii+dim)= s;
  //find good dt
  dt*= 2.;
  while ( err > 1. && init_dt / dt < _MAX_DT_REDUCE ){
    dt/= 2.;
    //do one step with step dt, and one with step dt/2.
    /*
      dt
    */
    //drift for c1*dt
    leapfrog_leapq(dim,qo,po,c1*dt,q12);
    to+= c1*dt;
    //kick for d1*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d1*dt,a,p12);
    //drift for c2*dt
    leapfrog_leapq(dim,q12,p12,c2*dt,qtmp);
    to+= c2*dt;
    //kick for d2*dt
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d2*dt,a,ptmp);
    //drift for c3*dt
    leapfrog_leapq(dim,qtmp,ptmp,c3*dt,q12);
    to+= c3*dt;
    //kick for d3*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d3*dt,a,p12);
    //drift for c4*dt
    leapfrog_leapq(dim,q12,p12,c4*dt,qtmp);
    to+= c4*dt;
    //kick for d4*dt
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d4*dt,a,ptmp);
    //drift for c5*dt
    leapfrog_leapq(dim,qtmp,ptmp,c5*dt,q12);
    to+= c5*dt;
    //kick for d5*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d5*dt,a,p12);
    //drift for c6*dt
    leapfrog_leapq(dim,q12,p12,c6*dt,qtmp);
    to+= c6*dt;
    //kick for d6*dt
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d6*dt,a,ptmp);
    //drift for c7*dt
    leapfrog_leapq(dim,qtmp,ptmp,c7*dt,q12);
    to+= c7*dt;
    //kick for d7*dt
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d7*dt,a,p11);
    //drift for c8*dt
    leapfrog_leapq(dim,q12,p11,c8*dt,q11);
    to+= c8*dt;
    //p8=p7
    //reset
    to-= dt;   
    /*
      dt/2
    */
    //drift for c1*dt/2
    leapfrog_leapq(dim,qo,po,c1*dt/2.,q12);
    to+= c1*dt/2.;
    //kick for d1*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,po,d1*dt/2.,a,p12);
    //drift for c2*dt/2
    leapfrog_leapq(dim,q12,p12,c2*dt/2.,qtmp);
    to+= c2*dt/2.;
    //kick for d2*dt/2
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d2*dt/2.,a,ptmp);
    //drift for c3*dt/2
    leapfrog_leapq(dim,qtmp,ptmp,c3*dt/2.,q12);
    to+= c3*dt/2.;
    //kick for d3*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d3*dt/2.,a,p12);
    //drift for c4*dt/2
    leapfrog_leapq(dim,q12,p12,c4*dt/2.,qtmp);
    to+= c4*dt/2.;
    //kick for d4*dt/2
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d4*dt/2.,a,ptmp);
    //drift for c5*dt/2
    leapfrog_leapq(dim,qtmp,ptmp,c5*dt/2.,q12);
    to+= c5*dt/2.;
    //kick for d5*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d5*dt/2.,a,p12);
    //drift for c6*dt/2
    leapfrog_leapq(dim,q12,p12,c6*dt/2.,qtmp);
    to+= c6*dt/2.;
    //kick for d6*dt
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d6*dt/2.,a,ptmp);
    //drift for c7*dt/2.
    leapfrog_leapq(dim,qtmp,ptmp,c7*dt/2.,q12);
    to+= c7*dt/2.;
    //kick for d7*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d7*dt/2.,a,p12);
    //drift for (c1+c8)*dt/2
    leapfrog_leapq(dim,q12,p12,(c1+c8)*dt/2.,qtmp);
    to+= (c1+c8)*dt/2.;
    //kick for d1*dt/2
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d1*dt/2.,a,ptmp);
    //drift for c2*dt/2
    leapfrog_leapq(dim,qtmp,ptmp,c2*dt/2.,q12);
    to+= c2*dt/2.;
    //kick for d2*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d2*dt/2.,a,p12);
    //drift for c3*dt/2
    leapfrog_leapq(dim,q12,p12,c3*dt/2.,qtmp);
    to+= c3*dt/2.;
    //kick for d3*dt/2
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d3*dt/2.,a,ptmp);
    //drift for c4*dt/2
    leapfrog_leapq(dim,qtmp,ptmp,c4*dt/2.,q12);
    to+= c4*dt/2.;
    //kick for d4*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d4*dt/2.,a,p12);
    //drift for c5*dt/2
    leapfrog_leapq(dim,q12,p12,c5*dt/2.,qtmp);
    to+= c5*dt/2.;
    //kick for d5*dt/2
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d5*dt/2.,a,ptmp);
    //drift for c6*dt/2
    leapfrog_leapq(dim,qtmp,ptmp,c6*dt/2.,q12);
    to+= c6*dt/2.;
    //kick for d6*dt/2
    func(to,q12,a,nargs,potentialArgs);
    leapfrog_leapp(dim,ptmp,d6*dt/2.,a,p12);
    //drift for c7*dt/2
    leapfrog_leapq(dim,q12,p12,c7*dt/2.,qtmp);
    to+= c7*dt/2.;
    //kick for d7*dt
    func(to,qtmp,a,nargs,potentialArgs);
    leapfrog_leapp(dim,p12,d7*dt/2.,a,ptmp);
    //drift for c8*dt/2.
    leapfrog_leapq(dim,qtmp,ptmp,c8*dt/2.,q12);
    to+= c8*dt/2.;
    //p8=p7
    for (ii=0; ii < dim; ii++) *(p12+ii)= *(ptmp+ii);
    //Norm
    err= 0.;
    for (ii=0; ii < dim; ii++) {
      err+= exp(2.*log(fabs(*(q11+ii)-*(q12+ii)))-2.* *(scale+ii));
      err+= exp(2.*log(fabs(*(p11+ii)-*(p12+ii)))-2.* *(scale+ii+dim));
    }
    err= sqrt(err/2./dim);
    //reset
    to-= dt;
  }
  //free what we allocated
  free(q11);
  free(q12);
  free(p11);
  free(qtmp);
  free(ptmp);
  free(a);
  free(scale);
  //return
  //printf("%f\n",dt);
  //fflush(stdout);
  return dt;
}
