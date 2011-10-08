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
       double *args: arguments
  Other arguments are:
       int dim: dimension
       double *yo: initial value [qo,po], dimension: 2*dim
       int nt: number of times at which the output is wanted
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       double *args: see above
       double rtol, double atol: relative and absolute tolerance levels desired
  Output:
       double *result: result (nt blocks of size 2dim)
*/
void leapfrog(void (*func)(double t, double *q, double *a,
			   int nargs, struct leapFuncArg * leapFuncArgs),
	      int dim,
	      double * yo,
	      int nt, double *t,
	      int nargs, struct leapFuncArg * leapFuncArgs,
	      double rtol, double atol,
	      double *result){
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
  //Estimate necessary stepsize
  double dt= (*(t+1))-(*t);
  double init_dt= dt;
  dt= leapfrog_estimate_step(*func,dim,qo,po,dt,t,nargs,leapFuncArgs,
			     rtol,atol);
  long ndt= (long) (init_dt/dt);
  //Integrate the system
  double to= *t;
  for (ii=0; ii < (nt-1); ii++){
    //drift half
    leapfrog_leapq(dim,qo,po,dt/2.,q12);
    //now drift full for a while
    for (jj=0; jj < (ndt-1); jj++){
      //kick
      func(to+dt/2.,q12,a,nargs,leapFuncArgs);
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
    func(to+dt/2.,q12,a,nargs,leapFuncArgs);
    leapfrog_leapp(dim,po,dt,a,po);
    //drift
    leapfrog_leapq(dim,q12,po,dt/2.,qo);
    to= to+dt;
    //save
    save_qp(dim,qo,po,result);
    result+= 2 * dim;
  }
  //Free allocated memory
  free(qo);
  free(po);
  free(q12);
  free(a);
  //We're done
}

void leapfrog_leapq(int dim, double *q,double *p,double dt,double *qn){
  int ii;
  for (ii=0; ii < dim; ii++) (*qn++)= (*q++) +dt * (*p++);
}
void leapfrog_leapp(int dim, double *p,double dt,double *a,double *pn){
  int ii;
  for (ii=0; ii< dim; ii++) (*pn++)= (*p++) + dt * (*a++);
}

inline void save_qp(int dim, double *qo, double *po, double *result){
  int ii;
  for (ii=0; ii < dim; ii++) *result++= *qo++;
  for (ii=0; ii < dim; ii++) *result++= *po++;
}

double leapfrog_estimate_step(void (*func)(double t, double *q, double *a,int nargs, struct leapFuncArg *),
			      int dim, double *qo,double *po,
			      double dt, double *t,
			      int nargs,struct leapFuncArg * leapFuncArgs,
			      double rtol,double atol){
  //return dt;
  //scalars
  double err= 2.;
  double max_val_q, max_val_p;
  double to= *t;
  //allocate and initialize
  double *qmax= (double *) malloc ( dim * sizeof(double) );
  double *pmax= (double *) malloc ( dim * sizeof(double) );
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
  //set qmax, pmax
  for (ii=0; ii < dim; ii++)
    *(qmax+ii)= max_val_q;
  for (ii=0; ii < dim; ii++)
    *(pmax+ii)= max_val_p;
  //set up scale
  for (ii=0; ii < dim; ii++) {
    *(scale+ii)= atol + rtol * *(qmax+ii);
    *(scale+ii+dim)= atol + rtol * *(pmax+ii);
  }
  //find good dt
  dt*= 2.
  while ( err > 1. ){
    dt/= 2.
    //do one leapfrog step with step dt, and one with step dt/2.
    //dt
    leapfrog_leapq(dim,qo,po,dt/2.,q12);
    func(to+dt/2.,q12,a,nargs,leapFuncArgs);
    leapfrog_leapp(dim,po,dt,a,p11);
    leapfrog_leapq(dim,q12,p11,dt/2.,q11);
    //dt/2.
    leapfrog_leapq(dim,qo,po,dt/4.,q12);
    func(to+dt/4.,q12,a,nargs,leapFuncArgs);
    leapfrog_leapp(dim,po,dt/2.,a,ptmp);
    leapfrog_leapq(dim,q12,ptmp,dt/2.,qtmp);//Take full step combining two half
    func(to+3.*dt/4.,qtmp,a,nargs,leapFuncArgs);
    leapfrog_leapp(dim,ptmp,dt/2.,a,p12);
    leapfrog_leapq(dim,qtmp,p12,dt/4.,q12);//Take full step combining two half   
    //Norm
    err= 0.;
    for (ii=0; ii < dim; ii++) {
      err+= pow((*(q11+ii)-*(q12+ii)) / *(scale+ii),2.);
      err+= pow((*(p11+ii)-*(p12+ii)) / *(scale+ii+dim),2.);
    }
    err= sqrt(err/2./dim);
  }
  //free what we allocated
  free(qmax);
  free(pmax);
  free(q12);
  free(a);
  //return
  return dt;
}
