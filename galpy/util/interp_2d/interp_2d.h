#ifndef __INTERP_2D_H__
#define __INTERP_2D_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_math.h>
#include <gsl/gsl_spline.h>

#include "cubic_bspline_2d_coeffs.h"
#include "cubic_bspline_2d_interpol.h"

enum
{
    INTERP_2D_LINEAR=0,
    INTERP_2D_CUBIC_BSPLINE=1
};

typedef struct
{
    int size1;
    int size2;
    double * xa;
    double * ya;
    double * za;
    int type;        
}interp_2d;

interp_2d * interp_2d_alloc(int size1, int size2);
void interp_2d_free(interp_2d * i2d);

void interp_2d_init(interp_2d * i2d, const double * xa, const double * ya, const double * za, int type);

double interp_2d_eval(interp_2d  * i2d, double x, double y, gsl_interp_accel * accx, gsl_interp_accel * accy);
void interp_2d_eval_grad(interp_2d * i2d, double x, double y, double * grad, gsl_interp_accel * accx, gsl_interp_accel * accy);
double interp_2d_eval_cubic_bspline(interp_2d * i2d, double x, double y, gsl_interp_accel * accx,gsl_interp_accel * accy);

#ifdef __cplusplus
}
#endif
#endif

