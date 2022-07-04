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

#ifndef __LEUNG_DOP853__
#define __LEUNG_DOP853__
#ifdef __cplusplus
extern "C" {
#endif
#include "signal.h"
#include <galpy_potentials.h>
/* Global variables */
extern volatile sig_atomic_t interrupted;
#ifndef _WIN32
void handle_sigint(int);
#else
#include "windows.h"
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType);
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

/* Function declarations */
static inline void save_dop853(
	int dim,
	double *y0,
	double *result)
{
	int ii;
	for (ii = 0; ii < dim; ii++) *result++ = *y0++;
}

static inline double custom_sign(
	double x,
	double y)
{
	return (y > 0.0) ? fabs(x) : -fabs(x);

}

void dop853 (
	void(*func)(double, double *, double *, int, struct potentialArg *),
	int,
	double *,
	int,
	double,
	double *,
	int,
	struct potentialArg *,
	double,
	double,
	double *,
	int *
);
#ifdef __cplusplus
}
#endif
#endif
