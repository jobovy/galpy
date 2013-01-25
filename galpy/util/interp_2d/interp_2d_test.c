#include <gsl/gsl_vector.h>
#include <stdio.h>

#include "interp_2d.h"
#include "discretize_utils.h"

double sigma = 2.0;
double mu = 0.7;

int nx = 100;
 int ny = 100;
 double xlo = 0.1;
 double xhi = 10.0;
 double ylo = 0.1;
 double yhi = 10.0;
 double expx = 2.5;
 double expy = 2.5;
 
int nx_test = 1000;
int ny_test = 1000;
double xlo_test = 0.2;
double xhi_test = 9.9;
double ylo_test = 0.2;
double yhi_test = 9.9;
double expx_test = 2.5;
double expy_test = 2.5;

int type=INTERP_2D_CUBIC_BSPLINE;

double f(double x, double y)
{
    //return x*y+2*x+8*y+pow(x,3)+pow(x*y,2);
    //return 8.*x+13.*y + 12.*x*y;
    return pow( pow(x,mu)*pow(y,1.0-mu), 1.0-sigma)/(1.0-sigma);
}

double dfdx(double x, double y)
{
    //return y + 2 + 3*pow(x,2) + 2*x*pow(y,2);
    //return 8. + 12.*y; 
    return pow( pow(x,mu)*pow(y,1.0-mu), -1.0*sigma)*mu*pow(x,mu-1.0)*pow(y,1.0-mu);
}

double dfdy(double x, double y)
{
    //return x + 8 + 2*y*pow(x,2);
    //return 13. + 12.*x;
    return pow( pow(x,mu)*pow(y,1.0-mu), -1.0*sigma)*(1.0-mu)*pow(x,mu)*pow(y,-1.0*mu);
}

int main()

{
    interp_2d * i2d = interp_2d_alloc(nx,ny);
    
    gsl_vector * x = gsl_vector_alloc(nx);
    gsl_vector * y = gsl_vector_alloc(ny);
    gsl_vector * z = gsl_vector_alloc(nx*ny);
  
    expspace(xlo,xhi,nx,expx,x);
    expspace(ylo,yhi,ny,expy,y);
    
    int i,j;
    for(i=0; i<nx; i++)
    {
        for(j=0; j<ny; j++)
        {
            gsl_vector_set(z,i*ny+j, f(gsl_vector_get(x,i), gsl_vector_get(y,j)));
        }
    }
    
    interp_2d_init(i2d,x->data,y->data,z->data,type);
    
    gsl_vector * xx = gsl_vector_alloc(nx_test);
    gsl_vector * yy = gsl_vector_alloc(ny_test);
    
    expspace(xlo_test,xhi_test,nx_test,expx_test,xx);
    expspace(ylo_test,yhi_test,ny_test,expy_test,yy);
    
    gsl_interp_accel * acc = gsl_interp_accel_alloc();
    
    double sum=0., sum_dx=0., sum_dy=0.;
    double min=+HUGE_VAL, min_dx=+HUGE_VAL, min_dy=+HUGE_VAL;
    double max=-HUGE_VAL, max_dx=-HUGE_VAL, max_dy=-HUGE_VAL;
    
    double tmp_grad[2];
    for(i=0; i<nx_test; i++)
    {
        for(j=0; j<ny_test; j++)
        {
            double actual = f(gsl_vector_get(xx,i), gsl_vector_get(yy,j));
            double est = interp_2d_eval( i2d, gsl_vector_get(xx,i), gsl_vector_get(yy,j), acc);
            double err = est/actual-1.;
            sum += fabs(err);
            min = (fabs(err)<min ? fabs(err) : min);
            max = (fabs(err)>max ? fabs(err) : max);
            
            interp_2d_eval_grad( i2d, gsl_vector_get(xx,i), gsl_vector_get(yy,j), tmp_grad, acc);
            double actual_dx = dfdx(gsl_vector_get(xx,i), gsl_vector_get(yy,j));
            double actual_dy = dfdy(gsl_vector_get(xx,i), gsl_vector_get(yy,j));
            double est_dx = tmp_grad[0];
            double est_dy = tmp_grad[1];
            double err_dx = est_dx/actual_dx-1.;
            double err_dy = est_dy/actual_dy-1.;
            sum_dx += fabs(err_dx);
            sum_dy += fabs(err_dy); 
            min_dx = (fabs(err_dx)<min_dx ? fabs(err_dx) : min_dx);
            max_dx = (fabs(err_dx)>max_dx ? fabs(err_dx) : max_dx);
            min_dy = (fabs(err_dy)<min_dy ? fabs(err_dy) : min_dy);
            max_dy = (fabs(err_dy)>max_dy ? fabs(err_dy) : max_dy);   
        }
    }
    
    printf("\nPercent abs. errors in f(x,y)     : min = %e, max = %e, avg = %e\n",100*min,100*max,100*sum/(nx_test*ny_test));
    printf("Percent abs. errors in df(x,y)/dx : min = %e, max = %e, avg = %e\n",100*min_dx,100*max_dx,100*sum_dx/(nx_test*ny_test));
    printf("Percent abs. errors in df(x,y)/dy : min = %e, max = %e, avg = %e\n\n",100*min_dy,100*max_dy,100*sum_dy/(nx_test*ny_test));
    
    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(z);
    gsl_vector_free(xx);
    gsl_vector_free(yy);
    
    gsl_interp_accel_free(acc);

    interp_2d_free(i2d);
    
    return 0;
    
}
