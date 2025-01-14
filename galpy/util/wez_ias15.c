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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <bovy_symplecticode.h>
#include <wez_ias15.h>
#include "signal.h"

const double integrator_error_threshold = 1e-16; //e_deltab from paper
const double precision_parameter = 1e-9; //e_b from paper
const int order = 7;

const double h[8] = {
    0.0,
    0.05626256053692215,
    0.18024069173689236,
    0.35262471711316964,
    0.54715362633055538,
    0.73421017721541053,
    0.88532094683909577,
    0.97752061356128750
};

const double c_0_0 = 1;
const double c_1_0 = -0.05626256053692215;
const double c_1_1 = 1;
const double c_2_0 = 0.01014080283006363;
const double c_2_1 = -0.23650325227381452;
const double c_2_2 = 1;
const double c_3_0 = -0.003575897729251617;
const double c_3_1 = 0.09353769525946207;
const double c_3_2 = -0.5891279693869842;
const double c_3_3 = 1;
const double c_4_0 = 0.001956565409947221;
const double c_4_1 = -0.05475538688906869;
const double c_4_2 = 0.4158812000823069;
const double c_4_3 = -1.1362815957175396;
const double c_4_4 = 1;
const double c_5_0 = -0.0014365302363708915;
const double c_5_1 = 0.042158527721268706;
const double c_5_2 = -0.3600995965020568;
const double c_5_3 = 1.250150711840691;
const double c_5_4 = -1.87049177293295;
const double c_5_5 = 1;
const double c_6_0 = 0.0012717903090268678;
const double c_6_1 = -0.03876035791590677;
const double c_6_2 = 0.360962243452846;
const double c_6_3 = -1.466884208400427;
const double c_6_4 = 2.906136259308429;
const double c_6_5 = -2.7558127197720457;
const double c_6_6 = 1;
const double c_7_0 = -0.0012432012432012432;
const double c_7_1 = 0.03916083916083916;
const double c_7_2 = -0.3916083916083916;
const double c_7_3 = 1.7948717948717947;
const double c_7_4 = -4.3076923076923075;
const double c_7_5 = 5.6;
const double c_7_6 = -3.7333333333333334;
const double c_7_7 = 1;
const double c_8_0 = 0.0012432012432012432;
const double c_8_1 = -0.0404040404040404;
const double c_8_2 = 0.4307692307692308;
const double c_8_3 = -2.1864801864801864;
const double c_8_4 = 6.102564102564102;
const double c_8_5 = -9.907692307692308;
const double c_8_6 = 9.333333333333332;
const double c_8_7 = -4.733333333333333;
const double c_8_8 = 1;

const double d_0_0 = 1;
const double d_1_0 = 0.05626256053692215;
const double d_1_1 = 1;
const double d_2_0 = 0.0031654757181708297;
const double d_2_1 = 0.23650325227381452;
const double d_2_2 = 1;
const double d_3_0 = 0.00017809776922174343;
const double d_3_1 = 0.04579298550602792;
const double d_3_2 = 0.5891279693869842;
const double d_3_3 = 1;
const double d_4_0 = 1.002023652232913e-05;
const double d_4_1 = 0.008431857153525702;
const double d_4_2 = 0.25353406905456927;
const double d_4_3 = 1.1362815957175396;
const double d_4_4 = 1;
const double d_5_0 = 5.637641639318209e-07;
const double d_5_1 = 0.001529784002500466;
const double d_5_2 = 0.09783423653244401;
const double d_5_3 = 0.8752546646840911;
const double d_5_4 = 1.87049177293295;
const double d_5_5 = 1;
const double d_6_0 = 3.171881540176138e-08;
const double d_6_1 = 0.0002762930909826477;
const double d_6_2 = 0.03602855398373646;
const double d_6_3 = 0.5767330002770787;
const double d_6_4 = 2.2485887607691595;
const double d_6_5 = 2.7558127197720457;
const double d_6_6 = 1;
const double d_7_0 = 1.7845817717010581e-09;
const double d_7_1 = 4.9830976656238314e-05;
const double d_7_2 = 0.012980851747494276;
const double d_7_3 = 0.3515901065098413;
const double d_7_4 = 2.2276697528059834;
const double d_7_5 = 4.688367487148971;
const double d_7_6 = 3.7333333333333334;
const double d_7_7 = 1;
const double d_8_0 = 1.0040513996341856e-10;
const double d_8_1 = 8.98335428421703e-06;
const double d_8_2 = 0.004627200152004401;
const double d_8_3 = 0.20535465350630017;
const double d_8_4 = 1.987167910494932;
const double d_8_5 = 6.378379695658343;
const double d_8_6 = 8.337777777777777;
const double d_8_7 = 4.733333333333333;
const double d_8_8 = 1;

const double r_1_0 = 17.773808914078;
const double r_2_0 = 5.548136718537217;
const double r_2_1 = 8.065938648381888;
const double r_3_0 = 2.835876078644439;
const double r_3_1 = 3.3742499769626355;
const double r_3_2 = 5.801001559264062;
const double r_4_0 = 1.8276402675175978;
const double r_4_1 = 2.0371118353585844;
const double r_4_2 = 2.725442211808226;
const double r_4_3 = 5.140624105810932;
const double r_5_0 = 1.3620078160624696;
const double r_5_1 = 1.4750402175604116;
const double r_5_2 = 1.8051535801402514;
const double r_5_3 = 2.620644926387035;
const double r_5_4 = 5.3459768998711095;
const double r_6_0 = 1.1295338753367898;
const double r_6_1 = 1.2061876660584456;
const double r_6_2 = 1.418278263734739;
const double r_6_3 = 1.8772424961868102;
const double r_6_4 = 2.957116017290456;
const double r_6_5 = 6.617662013702422;
const double r_7_0 = 1.0229963298234868;
const double r_7_1 = 1.0854721939386425;
const double r_7_2 = 1.2542646222818779;
const double r_7_3 = 1.6002665494908161;
const double r_7_4 = 2.3235983002196945;
const double r_7_5 = 4.109975778344558;
const double r_7_6 = 10.846026190236847;
const double r_8_0 = 1.0;
const double r_8_1 = 1.0596167516347892;
const double r_8_2 = 1.2198702593798945;
const double r_8_3 = 1.5446990739910793;
const double r_8_4 = 2.2082544062281713;
const double r_8_5 = 3.762371295948582;
const double r_8_6 = 8.719988284145643;
const double r_8_7 = 44.48519992867184;


/*
IAS15 integrator
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
void wez_ias15(void (*func)(double t, double *q, double *a, int nargs, struct potentialArg * potentialArgs),
    int dim,
    double * yo,
    int nt,
    double dt,
    double *t,
    int nargs,
    struct potentialArg * potentialArgs,
    double rtol,
    double atol,
    double *result,
    int * err){
  //Declare and initialize
  double *x= (double *) malloc ( dim * sizeof(double) );
  double *v= (double *) malloc ( dim * sizeof(double) );
  double *a= (double *) malloc ( dim * sizeof(double) );
  double *xs= (double *) malloc ( dim * sizeof(double) ); //x substep
  double *vs= (double *) malloc ( dim * sizeof(double) ); //v substep

  double *Bs= (double *) malloc ( (order * dim) * sizeof(double) );
  double *Es= (double *) malloc ( (order * dim) * sizeof(double) );
  double *BDs= (double *) malloc ( (order * dim) * sizeof(double) );
  double *Gs= (double *) malloc ( (order * dim) * sizeof(double) );
  double *Fs= (double *) malloc ( ((order + 1) * dim) * sizeof(double) );

  double hs;

  int ii, jj, kk;

  for (ii=0; ii < dim; ii++) {
    *x++= *(yo+ii);
    *v++= *(yo+dim+ii);
  }

  x-= dim;
  v-= dim;

  for(int i=0; i < (order * dim); i++){
    Bs[i] = 0;
    Es[i] = 0;
    BDs[i] = 0;
    Gs[i] = 0;
  }

  for(int i=0; i < (order + 1 * dim); i++){
    Fs[i] = 0;
  }

  double diff_G;

  save_ias15(dim, x, v, result);
  result+= 2 * dim;

  *err= 0;

  //Estimate necessary stepsize, use the returned time interval if the user does not provide
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt = init_dt; //Note in this case of dynamic timestepping, this makes little difference
  }

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
  //WHILE THERE IS TIME REMAINING, INTEGRATE A DYNAMIC TIMESTEP FORWARD AND SUBTRACT FROM THE TIME REMAINING
  double time_remaining = fabs(nt * dt);
  int steps = 1;

  while(time_remaining > 0) {
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

    double to_temp;
    double dt_temp;
    dt_temp = dt;

    to_temp = to + dt_temp;

    func(to_temp,x,a,nargs,potentialArgs);
    for (int i=0; i < dim; i++){
      Fs[i * (order + 1)] = a[i];
    }
    //update G from B
    update_Gs_from_Bs(dim, Gs, Bs);

    int iterations = 0;
    double integrator_error = integrator_error_threshold + 1; //init value above the threshold
    while(true){
      if(iterations == 12){
        //See paper for dynamic vs static treatment of predictor–corrector loop.
        break;
      }
      if (integrator_error < integrator_error_threshold){
        break;
      }

      //at start of each step reset xs
      //for (int l=0; l < dim; l++) *(xs+l)= *(x+l);

      double max_delta_B6 = 0.0; //also = max delta G
      double max_a = 0.0;
      for (int k=1; k < (order + 1); k++){
          //update position, update force, update G, update B
          update_position(xs, x, v, dim, h[k], dt_temp, Fs, Bs);

          func(to_temp,xs,a,nargs,potentialArgs);
          for (int i=0; i < dim; i++){
            Fs[i * (order + 1) + k] = a[i];

            diff_G = Gs[i * order + (k-1)];
            update_Gs_from_Fs(k, i, Gs, Fs);
            diff_G = Gs[i * order + (k-1)] - diff_G;

            update_Bs_from_Gs(k, i, Bs, Gs, diff_G);

            if (k == order){ //on last step, update max delta B6 and a, using "global" strategy in paper.
              double abs_delta_B6 = fabs(diff_G);
              double abs_a = fabs(Fs[i * (order + 1)]); //accel at beginning of step
              if (abs_delta_B6 > max_delta_B6){
                max_delta_B6 = abs_delta_B6;
              }
              if (abs_a > max_a){
                max_a = abs_a;
              }
            }
        }
      }

      integrator_error = max_delta_B6/max_a;
      iterations += 1;
    }

    //global error strategy for timestep
    double max_B6 = 0.0;
    double max_a = 0.0;
    for (int i=0; i < dim; i++){
      double abs_B6 = fabs(Bs[i * order + 6]);
      double abs_a = fabs(Fs[i * (order + 1)]);
      if (abs_B6 > max_B6){
        max_B6 = abs_B6;
      }
      if (abs_a > max_a){
        max_a = abs_a;
      }
    };

    //fix for inf values issue
    double correction_factor = seventhroot(precision_parameter / (max_B6/max_a));
    double dt_required = dt_temp * correction_factor;

    if(fabs(dt_temp) > fabs(dt_required)){
      //rejected, try again with dt required
      dt = dt_required;
    } else {
      //accepted, update position/velocity and do next timestep with dt required
      time_remaining -= fabs(dt); //will eventually get negative as we stepped forward the minimum of dt and time_remaining

      if (init_dt > 0){
        //estimate the function over the interval for any points in the t array
        while(to < t[steps] && t[steps] <= to_temp && steps < nt){
          //hs = (fabs(t[steps]) - fabs(to))/(fabs(to_temp) - fabs(to));
          hs = (fabs(t[steps] - to))/(fabs(to_temp - to));

          update_position(xs, x, v, dim, hs, dt_temp, Fs, Bs);
          update_velocity(vs, v, dim, hs, dt_temp, Fs, Bs);

          save_ias15(dim, xs, vs, result);
          result+= 2 * dim;

          steps += 1;
        }
      } else {
        //estimate the function over the interval for any points in the t array
        while(to > t[steps] && t[steps] >= to_temp && steps < nt){
          hs = (fabs(t[steps] - to))/(fabs(to_temp - to));

          update_position(xs, x, v, dim, hs, dt_temp, Fs, Bs);
          update_velocity(vs, v, dim, hs, dt_temp, Fs, Bs);

          save_ias15(dim, xs, vs, result);
          result+= 2 * dim;

          steps += 1;
        }
      }


      update_position(x, x, v, dim, 1, dt_temp, Fs, Bs);
      update_velocity(v, v, dim, 1, dt_temp, Fs , Bs);
      next_sequence_Bs(1, Bs, Es, BDs, dim);

      to = to_temp;
      dt = dt_required;
    }
  }
  // Back to default handler
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  //Free allocated memory
  free(x);
  free(v);
  free(a);
  free(xs);
  free(vs);
  free(Fs);
  free(Gs);
  free(Bs);
  free(Es);
  free(BDs);
}

void update_velocity(double *v, double *v1, int dim, double h_n, double T, double * Fs , double * Bs){
     for (int ii=0; ii < dim; ii++){
        *(v+ii) = *(v1+ii) +
        (h_n * T *
        (Fs[ii * (order + 1) + 0] + h_n *
        (Bs[ii * order + 0]/2 + h_n *
        (Bs[ii * order + 1]/3 + h_n *
        (Bs[ii * order + 2]/4 + h_n *
        (Bs[ii * order + 3]/5 + h_n *
        (Bs[ii * order + 4]/6 + h_n *
        (Bs[ii * order + 5]/7 + h_n *
        (Bs[ii * order + 6]/8 //+ h_n *
        //(Bs[ii * order + 7]/9
        )))))))));
     }
}


void update_position(double *y, double *y1, double *v, int dim, double h_n, double T, double * Fs, double * Bs){
    for (int ii=0; ii < dim; ii++){
      *(y+ii) = *(y1+ii) +
        (*(v+ii) * h_n * T) +
        ((h_n * T)*(h_n * T)) *
        (Fs[ii * (order + 1) + 0]/2 + h_n *
        (Bs[ii * order + 0]/6 + h_n *
        (Bs[ii * order + 1]/12 + h_n *
        (Bs[ii * order + 2]/20 + h_n *
        (Bs[ii * order + 3]/30 + h_n *
        (Bs[ii * order + 4]/42 + h_n *
        (Bs[ii * order + 5]/56 + h_n *
        (Bs[ii * order + 6]/72 //+ h_n *
        //(Bs[ii * order + 7]/90)
        ))))))));
    }
}

void update_Gs_from_Fs(int current_truncation_order, int i, double * Gs, double * Fs){
  //int i;
  int j;
  int h;
  //for (i = 0; i < dim; i++){
  j = i * order;
  h = i * (order + 1);

  if (current_truncation_order == 1){ Gs[j    ] = (Fs[h + 1] - Fs[h]) * r_1_0; }
  if (current_truncation_order == 2){ Gs[j + 1] = ((Fs[h + 2] - Fs[h]) * r_2_0 - Gs[j]) * r_2_1; }
  if (current_truncation_order == 3){ Gs[j + 2] = (((Fs[h + 3] - Fs[h]) * r_3_0 - Gs[j]) * r_3_1 - Gs[j + 1]) * r_3_2; }
  if (current_truncation_order == 4){ Gs[j + 3] = ((((Fs[h + 4] - Fs[h]) * r_4_0 - Gs[j]) * r_4_1 - Gs[j + 1]) * r_4_2 - Gs[j + 2]) * r_4_3; }
  if (current_truncation_order == 5){ Gs[j + 4] = (((((Fs[h + 5] - Fs[h]) * r_5_0 - Gs[j]) * r_5_1 - Gs[j + 1]) * r_5_2 - Gs[j + 2]) * r_5_3 - Gs[j + 3]) * r_5_4; }
  if (current_truncation_order == 6){ Gs[j + 5] = ((((((Fs[h + 6] - Fs[h]) * r_6_0 - Gs[j]) * r_6_1 - Gs[j + 1]) * r_6_2 - Gs[j + 2]) * r_6_3 - Gs[j + 3]) * r_6_4 - Gs[j + 4]) * r_6_5; }
  if (current_truncation_order == 7){ Gs[j + 6] = (((((((Fs[h + 7] - Fs[h]) * r_7_0 - Gs[j]) * r_7_1 - Gs[j + 1]) * r_7_2 - Gs[j + 2]) * r_7_3 - Gs[j + 3]) * r_7_4 - Gs[j + 4]) * r_7_5 - Gs[j + 5]) * r_7_6; }
    //if (current_truncation_order == 7){ Gs[j + 7] = ((((((((Fs[j + 8] - Fs[j + 0]) * r_8_0 - Gs[j + 0]) * r_8_1 - Gs[j + 1]) * r_8_2 - Gs[j + 2]) * r_8_3 - Gs[j + 3]) * r_8_4 - Gs[j + 4]) * r_8_5 - Gs[j + 5]) * r_8_6 - Gs[j + 6]) * r_8_7; }
  //}
}

void update_Gs_from_Bs(int dim, double * Gs, double * Bs){
  int i;
  int j;
  for (i = 0; i < dim; i++){
    j = i * order;
    Gs[j    ] = d_0_0 * Bs[j    ] + d_1_0 * Bs[j + 1] + d_2_0 * Bs[j + 2] + d_3_0 * Bs[j + 3] + d_4_0 * Bs[j + 4] + d_5_0 * Bs[j + 5] + d_6_0 * Bs[j + 6];// + d_7_0 * Bs[j + 7];
    Gs[j + 1] =                     d_1_1 * Bs[j + 1] + d_2_1 * Bs[j + 2] + d_3_1 * Bs[j + 3] + d_4_1 * Bs[j + 4] + d_5_1 * Bs[j + 5] + d_6_1 * Bs[j + 6];// + d_7_1 * Bs[j + 7];
    Gs[j + 2] =                                         d_2_2 * Bs[j + 2] + d_3_2 * Bs[j + 3] + d_4_2 * Bs[j + 4] + d_5_2 * Bs[j + 5] + d_6_2 * Bs[j + 6];// + d_7_2 * Bs[j + 7];
    Gs[j + 3] =                                                             d_3_3 * Bs[j + 3] + d_4_3 * Bs[j + 4] + d_5_3 * Bs[j + 5] + d_6_3 * Bs[j + 6];// + d_7_3 * Bs[j + 7];
    Gs[j + 4] =                                                                                 d_4_4 * Bs[j + 4] + d_5_4 * Bs[j + 5] + d_6_4 * Bs[j + 6];// + d_7_4 * Bs[j + 7];
    Gs[j + 5] =                                                                                                     d_5_5 * Bs[j + 5] + d_6_5 * Bs[j + 6];// + d_7_5 * Bs[j + 7];
    Gs[j + 6] =                                                                                                                         d_6_6 * Bs[j + 6];// + d_7_6 * Bs[j + 7];
  }
}

void next_sequence_Bs(double Q, double * Bs, double *Es, double * BDs, int dim){
  int i;
  int j;
  for (i = 0; i < dim; i++){
    j = i * order;

    BDs[j    ] = Bs[j    ] - Es[j    ];
    BDs[j + 1] = Bs[j + 1] - Es[j + 1];
    BDs[j + 2] = Bs[j + 2] - Es[j + 2];
    BDs[j + 3] = Bs[j + 3] - Es[j + 3];
    BDs[j + 4] = Bs[j + 4] - Es[j + 4];
    BDs[j + 5] = Bs[j + 5] - Es[j + 5];
    BDs[j + 6] = Bs[j + 6] - Es[j + 6];

    Es[j    ] = (Bs[j    ] + 2 * Bs[j + 1] + 3 * Bs[j + 2] + 4 * Bs[j + 3] + 5 * Bs[j + 4] + 6 * Bs[j + 5] + 7 * Bs[j + 6]);
    Es[j + 1] = (Bs[j + 1] + 3 * Bs[j + 2] + 6 * Bs[j + 3] + 10 * Bs[j + 4] + 15 * Bs[j + 5] + 21 * Bs[j + 6]);
    Es[j + 2] = (Bs[j + 2] + 4 * Bs[j + 3] + 10 * Bs[j + 4] + 20 * Bs[j + 5] + 35 * Bs[j + 6]);
    Es[j + 3] = (Bs[j + 3] + 5 * Bs[j + 4] + 15 * Bs[j + 5] + 35 * Bs[j + 6]);
    Es[j + 4] = (Bs[j + 4] + 6 * Bs[j + 5] + 21 * Bs[j + 6]);
    Es[j + 5] = (Bs[j + 5] + 7 * Bs[j + 6]);
    Es[j + 6] = (Bs[j + 6]);

    Bs[j    ] = Es[j    ] + BDs[j    ];
    Bs[j + 1] = Es[j + 1] + BDs[j + 1];
    Bs[j + 2] = Es[j + 2] + BDs[j + 2];
    Bs[j + 3] = Es[j + 3] + BDs[j + 3];
    Bs[j + 4] = Es[j + 4] + BDs[j + 4];
    Bs[j + 5] = Es[j + 5] + BDs[j + 5];
    Bs[j + 6] = Es[j + 6] + BDs[j + 6];
  }
}

void update_Bs_from_Gs(int current_truncation_order, int i, double * Bs, double * Gs, double diff_G){
  int j = i * order;
  if (current_truncation_order == 1){
    Bs[j    ] += diff_G;
  }
  if (current_truncation_order == 2){
    Bs[j    ] +=  diff_G * c_1_0;
    Bs[j + 1] +=  diff_G;
  }
  if (current_truncation_order == 3){
    Bs[j    ] +=  diff_G * c_2_0;
    Bs[j + 1] +=  diff_G * c_2_1;
    Bs[j + 2] +=  diff_G;
  }
  if (current_truncation_order == 4){
    Bs[j    ] +=  diff_G * c_3_0;
    Bs[j + 1] +=  diff_G * c_3_1;
    Bs[j + 2] +=  diff_G * c_3_2;
    Bs[j + 3] +=  diff_G;
  }
  if (current_truncation_order == 5){
    Bs[j    ] +=  diff_G * c_4_0;
    Bs[j + 1] +=  diff_G * c_4_1;
    Bs[j + 2] +=  diff_G * c_4_2;
    Bs[j + 3] +=  diff_G * c_4_3;
    Bs[j + 4] +=  diff_G;
  }
  if (current_truncation_order == 6){
    Bs[j    ] +=  diff_G * c_5_0;
    Bs[j + 1] +=  diff_G * c_5_1;
    Bs[j + 2] +=  diff_G * c_5_2;
    Bs[j + 3] +=  diff_G * c_5_3;
    Bs[j + 4] +=  diff_G * c_5_4;
    Bs[j + 5] +=  diff_G ;
  }
  if (current_truncation_order == 7){
    Bs[j    ] +=  diff_G * c_6_0;
    Bs[j + 1] +=  diff_G * c_6_1;
    Bs[j + 2] +=  diff_G * c_6_2;
    Bs[j + 3] +=  diff_G * c_6_3;
    Bs[j + 4] +=  diff_G * c_6_4;
    Bs[j + 5] +=  diff_G * c_6_5;
    Bs[j + 6] +=  diff_G;
  }
}

static double seventhroot(double a){
    // Without scaling, this is only accurate for arguments in [1e-7, 1e2]
    // With scaling: [1e-14, 1e8]
    double scale = 1;
    while(a<1e-7 && isnormal(a)){
        scale *= 0.1;
        a *= 1e7;
    }
    while(a>1e2 && isnormal(a)){
        scale *= 10;
        a *= 1e-7;
    }
    double x = 1.;
    for (int k=0; k<20;k++){  // A smaller number should be ok too.
        double x6 = x*x*x*x*x*x;
        x += (a/x6-x)/7.;
    }
    return x*scale;
}
