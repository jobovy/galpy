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
/*
  Function declarations
*/
void leapfrog(void (*func)(int, double, double *, double *,
			   int, double *),
	      int,
	      double *,
	      int, double *,
	      int, double *,
	      double, double,
	      double *);
void leapfrog_leapq(int, double *,double *,double,double *);
void leapfrog_leapp(int,double *,double,double *,double *);
inline void save_qp(int dim, double *qo, double *po, double *result);
double leapfrog_estimate_step(void (*func)(int, double , double *, double *,int, double *),
			      int, double *,double *,
			      double, double,
			      int,double *,
			      double,double);

/*
Leapfrog integrator
Usage:
   Provide the acceleration function func with calling sequence
       func (im,t,q,a,nargs,args)
   where
       int dim: dimension
       double t: time
       double * q: current position (dimension: dim)
       double * a: will be set to the derivative
       int nargs: number of arguments the function takes
       double *args: arguments
  Other arguments are:
       int dim: see above
       double *yo: initial value [qo,po], dimension: 2*dim
       int nt: number of times at which the output is wanted
       double *t: times at which the output is wanted (EQUALLY SPACED)
       int nargs: see above
       double *args: see above
       double rtol, double atol: relative and absolute tolerance levels desired
  Output:
       double *result: result (nt blocks of size 2dim)
*/
void leapfrog(void (*func)(int dim, double t, double *q, double *a,
			   int nargs, double *args),
	      int dim,
	      double * yo,
	      int nt, double *t,
	      int nargs, double *args,
	      double rtol, double atol,
	      double *result){
  //Initialize
  double *qo= yo;
  double *po= yo+dim;
  double *out;
  *out= *yo;
  double *q12, *a;
  //Estimate necessary stepsize
  double dt= *(t+1)-*t;
  double init_dt= dt;
  dt= leapfrog_estimate_step(*func,dim,qo,po,dt,*t,nargs,args,rtol,atol);
  int ndt= (int) init_dt/dt;
  //Integrate the system
  double to= *t;
  int ii, jj;
  for (ii=0; ii < (nt-1); ii++){
    //drift half
    leapfrog_leapq(dim,qo,po,dt/2.,q12);
    //now drift full for a while
    for (jj=0; jj < (ndt-1); jj++){
      //kick
      func(dim,to+dt/2.,q12,a,nargs,args);
      leapfrog_leapp(dim,po,dt,a,po);
      //drift
      leapfrog_leapq(dim,q12,po,dt,qo);
      to= to+dt;
    }
    //end with one last kick and drift
    //kick
    func(dim,to+dt/2.,q12,a,nargs,args);
    leapfrog_leapp(dim,po,dt,a,po);
    //drift
    leapfrog_leapq(dim,q12,po,dt,qo);
    to= to+dt;
    //save
    save_qp(dim,qo,po,result);
  }
  result-= dim*nt;
  //We're done
}

void leapfrog_leapq(int dim, double *q,double *p,double dt,double *qn){
  int ii;
  for (ii=0; ii < dim; ii++){
    *qn= *q+dt*(*p);
    qn++;
    q++;
    p++;
  }
  qn-= dim;
  q-= dim;
  p-= dim;
}
void leapfrog_leapp(int dim, double *p,double dt,double *a,double *pn){
  int ii;
  for (ii=0; ii< dim; ii++){
    *pn= *p+dt*(*a);
    pn++;
    p++;
    a++;
  }
  pn-= dim;
  p-= dim;
  a-= dim;
}

inline void save_qp(int dim, double *qo, double *po, double *result){
  int ii;
  for (ii=0; ii < dim; ii++){
    *result= *qo;
    result++;
    qo++;
  }
  qo-= dim;
  for (ii=0; ii < dim; ii++){
    *result= *po;
    result++;
    po++;
  }
  po-= dim;
}

double leapfrog_estimate_step(void (*func)(int dim, double t, double *q, double *a,int nargs, double *args),
			      int dim, double *qo,double *po,
			      double dt, double t,
			      int nargs,double *args,
			      double rtol,double atol){
  return dt;
}
