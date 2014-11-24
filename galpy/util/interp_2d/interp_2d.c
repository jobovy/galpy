#include "interp_2d.h"

interp_2d * interp_2d_alloc(int size1, int size2)
{
    interp_2d * i2d = (interp_2d *)malloc(sizeof(interp_2d));
    
    i2d->size1 = size1;
    i2d->size2 = size2;
    i2d->xa = (double *)malloc(size1*sizeof(double));
    i2d->ya = (double *)malloc(size2*sizeof(double));
    i2d->za = (double *)malloc(size1*size2*sizeof(double));
    
    return i2d;
}

void interp_2d_free(interp_2d * i2d)
{
    free(i2d->xa);
    free(i2d->ya);
    free(i2d->za);
    free(i2d);
}

void interp_2d_init(interp_2d * i2d, const double * xa, const double * ya, const double * za, int type)
{
    i2d->type = type;
    memcpy(i2d->xa,xa,(i2d->size1)*sizeof(double));
    memcpy(i2d->ya,ya,(i2d->size2)*sizeof(double));
    memcpy(i2d->za,za,(i2d->size1)*(i2d->size2)*sizeof(double));
    
// LCOV_EXCL_START
    if(type==INTERP_2D_CUBIC_BSPLINE)
    {
        samples_to_coefficients(i2d->za,i2d->size1,i2d->size2);
    }
// LCOV_EXCL_STOP
}
// LCOV_EXCL_START
double interp_2d_eval_linear(interp_2d * i2d, double x, double y, 
			     gsl_interp_accel * accx, gsl_interp_accel * accy)
{
    int size1 = i2d->size1;
    int size2 = i2d->size2;
    double * xa = i2d->xa;
    double * ya = i2d->ya;
    double * za = i2d->za;
    
    //x = (x > xa[size1-1]) ? xa[size1-1] : x;
    //y = (y > ya[size2-1]) ? ya[size2-1] : y;
    
    int ix = gsl_interp_accel_find(accx, xa, size1, x);
    int iy = gsl_interp_accel_find(accy, ya, size2, y);
    
    double z00 = za[ix*size2+iy];
    double z01 = za[ix*size2+iy+1];
    double z10 = za[(ix+1)*size2+iy];
    double z11 = za[(ix+1)*size2+iy+1];
    
    double denom = (xa[ix+1]-xa[ix])*(ya[iy+1]-ya[iy]);
    
    return z00*(xa[ix+1]-x)*(ya[iy+1]-y)/denom + z10*(x-xa[ix])*(ya[iy+1]-y)/denom + z01*(xa[ix+1]-x)*(y-ya[iy])/denom + z11*(x-xa[ix])*(y-ya[iy])/denom;
}

void interp_2d_eval_grad_linear(interp_2d * i2d, double x, double y, double * grad, 
				gsl_interp_accel * accx, gsl_interp_accel * accy)
{
    int size1 = i2d->size1;
    int size2 = i2d->size2;
    double * xa = i2d->xa;
    double * ya = i2d->ya;
    double * za = i2d->za;
    
    //x = (x > xa[size1-1]) ? xa[size1-1] : x;
    //y = (y > ya[size2-1]) ? ya[size2-1] : y;
    
    int ix = gsl_interp_accel_find(accx, xa, size1, x);
    int iy = gsl_interp_accel_find(accy, ya, size2, y);
    
    double z00 = za[ix*size2+iy];
    double z01 = za[ix*size2+iy+1];
    double z10 = za[(ix+1)*size2+iy];
    double z11 = za[(ix+1)*size2+iy+1];
    
    double denom = (xa[ix+1]-xa[ix])*(ya[iy+1]-ya[iy]);
    
    grad[0] = -1*z00*(ya[iy+1]-y)/denom + z10*(ya[iy+1]-y)/denom + -1*z01*(y-ya[iy])/denom + z11*(y-ya[iy])/denom;
    grad[1] = -1*z00*(xa[ix+1]-x)/denom - z10*(x-xa[ix])/denom + z01*(xa[ix+1]-x)/denom + z11*(x-xa[ix])/denom;
    
    return;
}
// LCOV_EXCL_STOP
double interp_2d_eval_cubic_bspline(interp_2d * i2d, double x, double y, 
				    gsl_interp_accel * accx, 
				    gsl_interp_accel * accy)
{
    int size1 = i2d->size1;
    int size2 = i2d->size2;
    double * xa = i2d->xa;
    double * ya = i2d->ya;
    double * za = i2d->za;
    
    x = (x > xa[size1-1]) ? xa[size1-1] : x;
    x = (x < xa[0]) ? xa[0] : x;
    y = (y > ya[size2-1]) ? ya[size2-1] : y;
    y = (y < ya[0]) ? ya[0] : y;

    int ix = gsl_interp_accel_find(accx, xa, size1, x);
    int iy = gsl_interp_accel_find(accy, ya, size2, y);
    
    double x_norm = ix + (x-xa[ix])/(xa[ix+1]-xa[ix]);
    double y_norm = iy + (y-ya[iy])/(ya[iy+1]-ya[iy]);
    
    return cubic_bspline_2d_interpol(za,size1,size2,x_norm,y_norm);
}
// LCOV_EXCL_START
void interp_2d_eval_grad_cubic_bspline(interp_2d * i2d, double x, double y, 
				       double * grad, 
				       gsl_interp_accel * accx, 
				       gsl_interp_accel * accy)
{
    int size1 = i2d->size1;
    int size2 = i2d->size2;
    double * xa = i2d->xa;
    double * ya = i2d->ya;
    double * za = i2d->za;
    
    int ix = gsl_interp_accel_find(accx, xa, size1, x);
    int iy = gsl_interp_accel_find(accy, ya, size2, y);
    
    double x_norm = ix + (x-xa[ix])/(xa[ix+1]-xa[ix]);
    double y_norm = iy + (y-ya[iy])/(ya[iy+1]-ya[iy]);    
    
    grad[0] = cubic_bspline_2d_interpol_dx(za,size1,size2,x_norm,y_norm)/(xa[ix+1]-xa[ix]);
    grad[1] = cubic_bspline_2d_interpol_dy(za,size1,size2,x_norm,y_norm)/(ya[iy+1]-ya[iy]);
    
    return;
}

double interp_2d_eval(interp_2d * i2d, double x, double y, 
		      gsl_interp_accel * accx, 
		      gsl_interp_accel * accy)
{
  return (i2d->type == INTERP_2D_CUBIC_BSPLINE ? interp_2d_eval_cubic_bspline(i2d,x,y,accx,accy) :  interp_2d_eval_linear(i2d,x,y,accx,accy) );
}

void interp_2d_eval_grad(interp_2d * i2d, double x, double y, double * grad, 
			 gsl_interp_accel * accx, 
			 gsl_interp_accel * accy)
{
    if(i2d->type == INTERP_2D_CUBIC_BSPLINE)
    {
      interp_2d_eval_grad_cubic_bspline(i2d,x,y,grad,accx,accy);
    }
    else
    {
      interp_2d_eval_grad_linear(i2d,x,y,grad,accx,accy);
    }
    return;
}
// LCOV_EXCL_STOP
