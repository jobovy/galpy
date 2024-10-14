/*
  C implementation of IAS15 integrator
 */
/*
Copyright (c) 2023, John Weatherall
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
#ifndef __WEZ_IAS_H__
#define __WEZ_IAS_H__
#ifdef __cplusplus
extern "C" {
#endif
/*
  include
*/

#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <bovy_symplecticode.h>
/*
  Global variables
*/

/*
  Function declarations
*/
void wez_ias15(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
	      int,
	      double *,
	      int, double, double *,
	      int, struct potentialArg *,
	      double, double,
	      double *,int *);

static inline void save_ias15(int dim, double *x_steps, double *v_steps, double *t_steps, int maxsteps, int steps, int nt, double dt, double *result){
  int ii;
  int jj;

  double *x_steps_short = (double *) malloc ( (steps + 1) * sizeof(double) ); //storing the result before interpolation
  double *v_steps_short = (double *) malloc ( (steps + 1) * sizeof(double) ); //storing the result before interpolation

  gsl_interp_accel *acc_x = gsl_interp_accel_alloc();
  gsl_spline *spline_x = gsl_spline_alloc(gsl_interp_cspline, (steps + 1));

  gsl_interp_accel *acc_v = gsl_interp_accel_alloc();
  gsl_spline *spline_v = gsl_spline_alloc(gsl_interp_cspline, (steps + 1));
  
  for (ii=0; ii < dim; ii++){  
    memcpy(x_steps_short, &x_steps[ii * (maxsteps + 1)], (steps + 1) * sizeof(double));
    memcpy(v_steps_short, &v_steps[ii * (maxsteps + 1)], (steps + 1) * sizeof(double));

    gsl_spline_init (spline_x, t_steps, x_steps_short, steps + 1);
    gsl_spline_init (spline_v, t_steps, v_steps_short, steps + 1);

    for (jj = 0; jj <= nt; jj += 1){
        //if (ii == 0){
        //  printf("%d\n", jj);
        //  printf("%f\n", jj * dt);
        //}
        result[ii + (2 * dim * jj)] = gsl_spline_eval(spline_x, jj * dt, acc_x);
        result[ii + (2 * dim * jj) + dim] = gsl_spline_eval(spline_v, jj * dt, acc_v);
      }
  }

  gsl_spline_free (spline_x);
  gsl_interp_accel_free (acc_x);
  gsl_spline_free (spline_v);
  gsl_interp_accel_free (acc_v);
  free(x_steps_short);
  free(v_steps_short);
}

static inline void save_dummy_ias15(int dim, double *x, double *v, double t, double *x_steps, double *v_steps, double *t_steps, int steps, int maxsteps){
  int ii;
  for (ii=0; ii < dim; ii++){
    x_steps[ii * (maxsteps + 1) + steps] = *x++;
  }
  for (ii=0; ii < dim; ii++){
    v_steps[ii * (maxsteps + 1) + steps] = *v++;
  }
  t_steps[steps] = t;
}

void update_velocity(double *, double *, int, double, double, double *, double *);
void update_position(double *, double *, double *, int, double, double, double *, double *);
void update_Gs_from_Bs(int, double *, double *);
void update_Gs_from_Fs(int, int, double *, double * );
void update_Bs_from_Gs(int, int, double *, double *, double);
void next_sequence_Bs(double, double *, double *, double *, int);
static double seventhroot(double);

#ifdef __cplusplus
}
#endif
#endif /* wez_ias15.h */
