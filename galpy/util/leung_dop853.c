/*
C implementations of Dormand-Prince 8(5,3)
 */
/*
Copyright (c) 2018, Henry Leung
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
#include <signal.h>
#include "bovy_symplecticode.h"
#include "leung_dop853.h"
#define _MAX_DT_REDUCE 10000.
#define uround 2.3e-16

/* time increment coefficients */
double c2 = 0.526001519587677318785587544488e-1;
double c3 = 0.789002279381515978178381316732e-1;
double c4 = 0.118350341907227396726757197510;
double c5 = 0.281649658092772603273242802490;
double c6 = 0.333333333333333333333333333333;
double c7 = 0.25;
double c8 = 0.307692307692307692307692307692;
double c9 = 0.651282051282051282051282051282;
double c10 = 0.6;
double c11 = 0.857142857142857142857142857142;
double c12 = 1;
double c13 = 1;
double c14 = 0.1;
double c15 = 0.2;
double c16 = 0.777777777777777777777777777778;

/* slope calculation coefficients */
double a21 = 5.26001519587677318785587544488e-2;
double a31 = 1.97250569845378994544595329183e-2;
double a32 = 5.91751709536136983633785987549e-2;
double a41 = 2.95875854768068491816892993775e-2;
double a43 = 8.87627564304205475450678981324e-2;
double a51 = 2.41365134159266685502369798665e-1;
double a53 = -8.84549479328286085344864962717e-1;
double a54 = 9.24834003261792003115737966543e-1;
double a61 = 3.7037037037037037037037037037e-2;
double a64 = 1.70828608729473871279604482173e-1;
double a65 = 1.25467687566822425016691814123e-1;
double a71 = 3.7109375e-2;
double a74 = 1.70252211019544039314978060272e-1;
double a75 = 6.02165389804559606850219397283e-2;
double a76 = -1.7578125e-2;
double a81 = 3.70920001185047927108779319836e-2;
double a84 = 1.70383925712239993810214054705e-1;
double a85 = 1.07262030446373284651809199168e-1;
double a86 = -1.53194377486244017527936158236e-2;
double a87 = 8.27378916381402288758473766002e-3;
double a91 = 6.24110958716075717114429577812e-1;
double a94 = -3.36089262944694129406857109825e0;
double a95 = -8.68219346841726006818189891453e-1;
double a96 = 2.75920996994467083049415600797e1;
double a97 = 2.01540675504778934086186788979e1;
double a98 = -4.34898841810699588477366255144e1;
double a101 = 4.77662536438264365890433908527e-1;
double a104 = -2.48811461997166764192642586468e0;
double a105 = -5.90290826836842996371446475743e-1;
double a106 = 2.12300514481811942347288949897e1;
double a107 = 1.52792336328824235832596922938e1;
double a108 = -3.32882109689848629194453265587e1;
double a109 = -2.03312017085086261358222928593e-2;
double a111 = -9.3714243008598732571704021658e-1;
double a114 = 5.18637242884406370830023853209e0;
double a115 = 1.09143734899672957818500254654e0;
double a116 = -8.14978701074692612513997267357e0;
double a117 = -1.85200656599969598641566180701e1;
double a118 = 2.27394870993505042818970056734e1;
double a119 = 2.49360555267965238987089396762e0;
double a1110 = -3.0467644718982195003823669022e0;
double a121 = 2.27331014751653820792359768449e0;
double a124 = -1.05344954667372501984066689879e1;
double a125 = -2.00087205822486249909675718444e0;
double a126 = -1.79589318631187989172765950534e1;
double a127 = 2.79488845294199600508499808837e1;
double a128 = -2.85899827713502369474065508674e0;
double a129 = -8.87285693353062954433549289258e0;
double a1210 = 1.23605671757943030647266201528e1;
double a1211 = 6.43392746015763530355970484046e-1;
double a141 = 5.61675022830479523392909219681e-2;
double a147 = 2.53500210216624811088794765333e-1;
double a148 = -2.46239037470802489917441475441e-1;
double a149 = -1.24191423263816360469010140626e-1;
double a1410 = 1.5329179827876569731206322685e-1;
double a1411 = 8.20105229563468988491666602057e-3;
double a1412 = 7.56789766054569976138603589584e-3;
double a1413 = -8.298e-3;
double a151 = 3.18346481635021405060768473261e-2;
double a156 = 2.83009096723667755288322961402e-2;
double a157 = 5.35419883074385676223797384372e-2;
double a158 = -5.49237485713909884646569340306e-2;
double a1511 = -1.08347328697249322858509316994e-4;
double a1512 = 3.82571090835658412954920192323e-4;
double a1513 = -3.40465008687404560802977114492e-4;
double a1514 = 1.41312443674632500278074618366e-1;
double a161 = -4.28896301583791923408573538692e-1;
double a166 = -4.69762141536116384314449447206e0;
double a167 = 7.68342119606259904184240953878e0;
double a168 = 4.06898981839711007970213554331e0;
double a169 = 3.56727187455281109270669543021e-1;
double a1613 = -1.39902416515901462129418009734e-3;
double a1614 = 2.9475147891527723389556272149e0;
double a1615 = -9.15095847217987001081870187138e0;

/*Final assembly coefficients */
double b1 = 5.42937341165687622380535766363e-2;
double b6 = 4.45031289275240888144113950566;
double b7 = 1.89151789931450038304281599044;
double b8 = -5.8012039600105847814672114227;
double b9 = 3.1116436695781989440891606237e-1;
double b10 = -1.52160949662516078556178806805e-1;
double b11 = 2.01365400804030348374776537501e-1;
double b12 = 4.47106157277725905176885569043e-2;
double bhh1 = 0.244094488188976377952755905512;
double bhh2 = 0.733846688281611857341361741547;
double bhh3 = 0.220588235294117647058823529412e-1;

/* Dense output coefficients */
double d41 = -0.84289382761090128651353491142e+1;
double d46 = 0.56671495351937776962531783590;
double d47 = -0.30689499459498916912797304727e+1;
double d48 = 0.23846676565120698287728149680e+1;
double d49 = 0.21170345824450282767155149946e+1;
double d410 = -0.87139158377797299206789907490;
double d411 = 0.22404374302607882758541771650e+1;
double d412 = 0.63157877876946881815570249290;
double d413 = -0.88990336451333310820698117400e-1;
double d414 = 0.18148505520854727256656404962e+2;
double d415 = -0.91946323924783554000451984436e+1;
double d416 = -0.44360363875948939664310572000e+1;
double d51 = 0.10427508642579134603413151009e+2;
double d56 = 0.24228349177525818288430175319e+3;
double d57 = 0.16520045171727028198505394887e+3;
double d58 = -0.37454675472269020279518312152e+3;;
double d59 = -0.22113666853125306036270938578e+2;
double d510 = 0.77334326684722638389603898808e+1;
double d511 = -0.30674084731089398182061213626e+2;
double d512 = -0.93321305264302278729567221706e+1;
double d513 = 0.15697238121770843886131091075e+2;
double d514 = -0.31139403219565177677282850411e+2;
double d515 = -0.93529243588444783865713862664e+1;
double d516 = 0.35816841486394083752465898540e+2;
double d61 = 0.19985053242002433820987653617e+2;
double d66 = -0.38703730874935176555105901742e+3;
double d67 = -0.18917813819516756882830838328e+3;
double d68 = 0.52780815920542364900561016686e+3;;
double d69 = -0.11573902539959630126141871134e+2;
double d610 = 0.68812326946963000169666922661e+1;
double d611 = -0.10006050966910838403183860980e+1;
double d612 = 0.77771377980534432092869265740;
double d613 = -0.27782057523535084065932004339e+1;
double d614 = -0.60196695231264120758267380846e+2;
double d615 = 0.84320405506677161018159903784e+2;
double d616 = 0.11992291136182789328035130030e+2;
double d71 = -0.25693933462703749003312586129e+2;
double d76 = -0.15418974869023643374053993627e+3;
double d77 = -0.23152937917604549567536039109e+3;
double d78 = 0.35763911791061412378285349910e+3;
double d79 = 0.93405324183624310003907691704e+2;
double d710 = -0.37458323136451633156875139351e+2;
double d711 = 0.10409964950896230045147246184e+3;
double d712 = 0.29840293426660503123344363579e+2;
double d713 = -0.43533456590011143754432175058e+2;
double d714 = 0.96324553959188282948394950600e+2;
double d715 = -0.39177261675615439165231486172e+2;
double d716 = -0.14972683625798562581422125276e+3;

/* Error calculation coefficients */
double er1 = 0.1312004499419488073250102996e-1;
double er6 = -0.1225156446376204440720569753e+1;
double er7 = -0.4957589496572501915214079952;
double er8 = 0.1664377182454986536961530415e+1;
double er9 = -0.3503288487499736816886487290;
double er10 = 0.3341791187130174790297318841;
double er11 = 0.8192320648511571246570742613e-1;
double er12 = -0.2235530786388629525884427845e-1;

/*
Core of DOP8(5, 3) integration
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
	   double *x0: initial value, dimension: dim
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
void dop853(void(*func)(double t, double *q, double *a, int nargs, struct potentialArg * potentialArgs),
	int dim,
	double * y0,
	int nt,
	double dt,
	double *t,
	int nargs,
	struct potentialArg * potentialArgs,
	double rtol,
	double atol,
	double *result,
	int *err_)
{
	rtol = exp(rtol);
	atol = exp(atol);
	// Same initial parameters also used in my python version
	double safe = 0.9;
	double beta = 0.0;
	double facold = 1.0e-4;
	double expo1 = 1.0 / 8.0 - beta * 0.2;
	double facc1 = 1.0 / 0.333;
	double facc2 = 1.0 / 6.0;
	double hmax;
	double pos_neg;
	hmax = t[nt - 1] - t[0];
	pos_neg = custom_sign(1.0, hmax);  // a check to see integrate forward or backward

	// Declare and initialize of others
	double *yy1 = (double *)malloc(dim * sizeof(double));
	double *yy_temp = (double *)malloc(dim * sizeof(double));
	double *k1 = (double *)malloc(dim * sizeof(double));
	double *k2 = (double *)malloc(dim * sizeof(double));
	double *k3 = (double *)malloc(dim * sizeof(double));
	double *k4 = (double *)malloc(dim * sizeof(double));
	double *k5 = (double *)malloc(dim * sizeof(double));
	double *k6 = (double *)malloc(dim * sizeof(double));
	double *k7 = (double *)malloc(dim * sizeof(double));
	double *k8 = (double *)malloc(dim * sizeof(double));
	double *k9 = (double *)malloc(dim * sizeof(double));
	double *k10 = (double *)malloc(dim * sizeof(double));
	double *rcont1 = (double*)malloc(dim * sizeof(double));
	double *rcont2 = (double*)malloc(dim * sizeof(double));
	double *rcont3 = (double*)malloc(dim * sizeof(double));
	double *rcont4 = (double*)malloc(dim * sizeof(double));
	double *rcont5 = (double*)malloc(dim * sizeof(double));
	double *rcont6 = (double*)malloc(dim * sizeof(double));
	double *rcont7 = (double*)malloc(dim * sizeof(double));
	double *rcont8 = (double*)malloc(dim * sizeof(double));
	int i;
	double hnew, ydiff, bspl;
	double dnf, dny, sk, h, h1, der2, der12;
	double sqr, err, err2, erri, deno;
	double fac, fac11;
	double s, s1;
	save_dop853(dim, y0, result);  // save first result which is the initials
	result += dim;  // shift to next memory

	#ifndef _WIN32
		struct sigaction action;
		memset(&action, 0, sizeof(struct sigaction));
		action.sa_handler = handle_sigint;
		sigaction(SIGINT, &action, NULL);
	#else
		SetConsoleCtrlHandler(CtrlHandler, TRUE);
	#endif

	// calculate k1
	func(t[0], y0, k1, nargs, potentialArgs);

	// start to estimate initial time step
	dnf = 0.0;
	dny = 0.0;
	for (i = 0; i < dim; i++)  // this loop only be vectorized with /fp:fast
	{
		sk = atol + rtol * fabs(y0[i]);
		sqr = k1[i] / sk;
		dnf += sqr * sqr;
		sqr = y0[i] / sk;
		dny += sqr * sqr;
	}

	h = custom_sign(min(sqrt(dny / dnf) * 0.01, fabs(hmax)), pos_neg);
	for (i = 0; i < dim; i++) k3[i] = y0[i] + h * k1[i]; // perform an explicit Euler step
	func(t[0] + h, k3, k2, nargs, potentialArgs);
	der2 = 0.0; // estimate the second derivative of the solution
	for (i = 0; i < dim; i++)  // this loop only be vectorized with /fp:fast
	{
		sk = atol + rtol * fabs(y0[i]);
		sqr = (k2[i] - k1[i]) / sk;
		der2 += sqr * sqr;
	}
	der2 = sqrt(der2) / h;
	der12 = max(fabs(der2), sqrt(dnf));
	h1 = pow(0.01 / der12, 1.0 / 8.0);
	h = custom_sign(min(100.0 * fabs(h), min(fabs(h1), fabs(hmax))), pos_neg);
	// finished estimate initial time step

	int reject = 0;
	double t_current = (double) t[0];  // store current integration time internally(not the current time wanted by user!!)
	double t_old = (double) t[0];
	double t_old_older = t_old;
	int finished_user_t_ii = 0;  // times indices wanted by user

	// basic integration step
	while (finished_user_t_ii < nt - 1)  // check if the current computed time indices less than total inices needed
	{
		if (interrupted) {
			*err_ = -10;
			interrupted = 0; // need to reset, bc library and vars stay in memory
#ifdef USING_COVERAGE
			__gcov_dump();
// LCOV_EXCL_START
			__gcov_reset();
#endif
			break;
// LCOV_EXCL_STOP
		}
		h = pos_neg * max(fabs(h), 1e3 * uround);  // keep time step not too small

		// the twelve stages
		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * a21 * k1[i];
		func(t_current + c2 * h, yy1, k2, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a31*k1[i] + a32 * k2[i]);
		func(t_current + c3 * h, yy1, k3, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a41*k1[i] + a43 * k3[i]);
		func(t_current + c4 * h, yy1, k4, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a51*k1[i] + a53 * k3[i] + a54 * k4[i]);
		func(t_current + c5 * h, yy1, k5, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a61*k1[i] + a64 * k4[i] + a65 * k5[i]);
		func(t_current + c6 * h, yy1, k6, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a71*k1[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
		func(t_current + c7 * h, yy1, k7, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a81*k1[i] + a84 * k4[i] + a85 * k5[i] + a86 * k6[i] + a87 * k7[i]);
		func(t_current + c8 * h, yy1, k8, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a91*k1[i] + a94 * k4[i] + a95 * k5[i] + a96 * k6[i] + a97 * k7[i] + a98 * k8[i]);
		func(t_current + c9 * h, yy1, k9, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a101*k1[i] + a104 * k4[i] + a105 * k5[i] + a106 * k6[i] + a107 * k7[i] + a108 * k8[i] + a109 * k9[i]);
		func(t_current + c10 * h, yy1, k10, nargs, potentialArgs);

		for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a111*k1[i] + a114 * k4[i] + a115 * k5[i] + a116 * k6[i] + a117 * k7[i] + a118 * k8[i] + a119 * k9[i] + a1110 * k10[i]);
		func(t_current + c11 * h, yy1, k2, nargs, potentialArgs);

		for (i = 0; i < dim; i++)
			yy1[i] = y0[i] + h * (a121*k1[i] + a124 * k4[i] + a125 * k5[i] + a126 * k6[i] + a127 * k7[i] + a128 * k8[i] + a129 * k9[i] + a1210 * k10[i] + a1211 * k2[i]);

		t_old_older = t_old;
		t_old = t_current;
		t_current = t_current + h;

		func(t_current, yy1, k3, nargs, potentialArgs);

		for (i = 0; i < dim; i++)
		{
			k4[i] = b1 * k1[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] + b9 * k9[i] +
				b10 * k10[i] + b11 * k2[i] + b12 * k3[i];
			k5[i] = y0[i] + h * k4[i];
		}

		// error estimation
		err = (double) 0.0;
		err2 = (double) 0.0;
		for (i = 0; i < dim; i++)
		{
			sk = atol + rtol * max(fabs(y0[i]), fabs(k5[i]));
			erri = k4[i] - bhh1 * k1[i] - bhh2 * k9[i] - bhh3 * k3[i];
			sqr = erri / sk;
			err2 += sqr * sqr;
			erri = er1 * k1[i] + er6 * k6[i] + er7 * k7[i] + er8 * k8[i] + er9 * k9[i] + er10 * k10[i] + er11 * k2[i] + er12 * k3[i];
			sqr = erri / sk;
			err += sqr * sqr;
		}
		deno = err + 0.01 * err2;
		if (deno <= 0.0) deno = (double) 1.0;
		err = fabs(h) * err * sqrt(1.0 / (deno* (double)dim));

		// computation of hnew
		fac11 = pow(err, expo1);

		// Lund-stabilization
		fac = fac11 / pow(facold, beta);

		// we require fac1 <= hnew/h <= fac2
		fac = max(facc2, min(facc1, fac / safe));
		hnew = h / fac;

		if (err <= 1.0)  // step accepted
		{
			facold = max(err, 1.0e-4);
			func(t_current, k5, k4, nargs, potentialArgs);

			// final preparation for dense output
			for (i = 0; i < dim; i++)
			{
				rcont1[i] = y0[i];
				ydiff = k5[i] - y0[i];
				rcont2[i] = ydiff;
				bspl = h * k1[i] - ydiff;
				rcont3[i] = bspl;
				rcont4[i] = ydiff - h * k4[i] - bspl;
				rcont5[i] = d41 * k1[i] + d46 * k6[i] + d47 * k7[i] + d48 * k8[i] + d49 * k9[i] + d410 * k10[i] + d411 * k2[i] + d412 * k3[i];
				rcont6[i] = d51 * k1[i] + d56 * k6[i] + d57 * k7[i] + d58 * k8[i] + d59 * k9[i] + d510 * k10[i] + d511 * k2[i] + d512 * k3[i];
				rcont7[i] = d61 * k1[i] + d66 * k6[i] + d67 * k7[i] + d68 * k8[i] + d69 * k9[i] + d610 * k10[i] + d611 * k2[i] + d612 * k3[i];
				rcont8[i] = d71 * k1[i] + d76 * k6[i] + d77 * k7[i] + d78 * k8[i] + d79 * k9[i] + d710 * k10[i] + d711 * k2[i] + d712 * k3[i];
			}

			// the next three function evaluations
			for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a141*k1[i] + a147 * k7[i] + a148 * k8[i] + a149 * k9[i] + a1410 * k10[i] + a1411 * k2[i] + a1412 * k3[i] + a1413 * k4[i]);
			func(t_old + c14 * h, yy1, k10, nargs, potentialArgs);


			for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a151*k1[i] + a156 * k6[i] + a157 * k7[i] + a158 * k8[i] + a1511 * k2[i] + a1512 * k3[i] + a1513 * k4[i] + a1514 * k10[i]);
			func(t_old + c15 * h, yy1, k2, nargs, potentialArgs);

			for (i = 0; i < dim; i++) yy1[i] = y0[i] + h * (a161*k1[i] + a166 * k6[i] + a167 * k7[i] + a168 * k8[i] + a169 * k9[i] + a1613 * k4[i] + a1614 * k10[i] + a1615 * k2[i]);
			func(t_old + c16 * h, yy1, k3, nargs, potentialArgs);

			for (i = 0; i < dim; i++)
			{
				rcont5[i] = h * (rcont5[i] + d413 * k4[i] + d414 * k10[i] + d415 * k2[i] + d416 * k3[i]);
				rcont6[i] = h * (rcont6[i] + d513 * k4[i] + d514 * k10[i] + d515 * k2[i] + d516 * k3[i]);
				rcont7[i] = h * (rcont7[i] + d613 * k4[i] + d614 * k10[i] + d615 * k2[i] + d616 * k3[i]);
				rcont8[i] = h * (rcont8[i] + d713 * k4[i] + d714 * k10[i] + d715 * k2[i] + d716 * k3[i]);
				k1[i] = k4[i];
				y0[i] = k5[i];
			}

			// loop for dense output in this time slot
			while ((finished_user_t_ii < nt - 1) && (pos_neg * t[finished_user_t_ii + 1] < pos_neg * t_current))
			{
				s = (t[finished_user_t_ii + 1] - t_old) / h;
				s1 = 1.0 - s;
				for (i = 0; i < dim; i++) yy_temp[i] = rcont1[i] + s * (rcont2[i] + s1 * (rcont3[i] + s * (rcont4[i] + s1 * (rcont5[i] + s * (rcont6[i] + s1 * (rcont7[i] + s * rcont8[i]))))));
				save_dop853(dim, yy_temp, result);  // save first result
				result += dim;
				finished_user_t_ii++;
			}

			hnew = (fabs(hnew) > fabs(hmax)) ? pos_neg * hmax : hnew;
			if (reject)
			    hnew = pos_neg * min(fabs(hnew), fabs(h));
			reject = 0;
		}
		else
		{
			// step rejected since error too big
			hnew = h / min(facc1, fac11 / safe);
			reject = 1;

			// reverse time increment since error rejected
			t_current = t_old;
			t_old = t_old_older;
		}

		h = hnew;  // current h
	}

	//Free allocated memory
	free(k10);
	free(k9);
	free(k8);
	free(k7);
	free(k6);
	free(k5);
	free(k4);
	free(k3);
	free(k2);
	free(k1);
	free(yy1);
	free(yy_temp);
	free(rcont8);
	free(rcont7);
	free(rcont6);
	free(rcont5);
	free(rcont4);
	free(rcont3);
	free(rcont2);
	free(rcont1);
	//We're done
}
