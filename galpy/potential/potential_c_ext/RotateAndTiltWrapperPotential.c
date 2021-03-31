#include <stdbool.h>
#include <math.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>
//RotateAndTiltWrapperPotential
static inline void rotate(double *x, double *y, double *z, double *rot){
  double xp,yp,zp;
  xp= *(rot)   * *x + *(rot+1) * *y + *(rot+2) * *z;
  yp= *(rot+3) * *x + *(rot+4) * *y + *(rot+5) * *z;
  zp= *(rot+6) * *x + *(rot+7) * *y + *(rot+8) * *z;
  *x= xp;
  *y= yp;
  *z= zp;
}
static inline void rotate_force(double *Fx, double *Fy, double *Fz,
				double *rot){
  double Fxp,Fyp,Fzp;
  Fxp= *(rot)   * *Fx + *(rot+3) * *Fy + *(rot+6) * *Fz;
  Fyp= *(rot+1) * *Fx + *(rot+4) * *Fy + *(rot+7) * *Fz;
  Fzp= *(rot+2) * *Fx + *(rot+5) * *Fy + *(rot+8) * *Fz;
  *Fx= Fxp;
  *Fy= Fyp;
  *Fz= Fzp;
}
double RotateAndTiltWrapperPotentialRforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
   double * args = potentialArgs->args;
   // change the zvector, calculate Rforce

   double * rot = args+1;
   double x, y;
   cyl_to_rect(R,phi,&x, &y);
   rotate(&x,&y,&z,rot);
   rect_to_cyl(x,y,&R, &phi);
   return *args * calcRforce(R,z,phi,t, potentialArgs->nwrapped,
      potentialArgs->wrappedPotentialArg);
}
double RotateAndTiltWrapperPotentialphiforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //calculate phiforce
    double * rot = args+1;
    double x, y;
    cyl_to_rect(R,phi,&x, &y);
    rotate(&x,&y,&z,rot);
    rect_to_cyl(x,y,&R, &phi);
    return *args * calcPhiforce(R,z,phi,t, potentialArgs->nwrapped,
       potentialArgs->wrappedPotentialArg);
}
double RotateAndTiltWrapperPotentialzforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //calculate zforce
    double * rot = args+1;
    double x, y;
    cyl_to_rect(R,phi,&x, &y);
    rotate(&x,&y,&z,rot);
    rect_to_cyl(x,y,&R, &phi);
    return *args * calczforce(R,z,phi,t, potentialArgs->nwrapped,
       potentialArgs->wrappedPotentialArg);
}
double RotateAndTiltWrapperPotentialPlanarRforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //calculate Rforce
    double * rot = args+1;
    double x, y;
    cyl_to_rect(R,phi,&x, &y);
    rotate(&x,&y,&z,rot);
    rect_to_cyl(x,y,&R, &phi);
    return *args * calcPlanarRforce(R,phi,t, potentialArgs->nwrapped,
       potentialArgs->wrappedPotentialArg);
}
double RotateAndTiltWrapperPotentialPlanarphiforce(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //calculate phiforce
    double * rot = args+1;
    double x, y;
    cyl_to_rect(R,phi,&x, &y);
    rotate(&x,&y,&z,rot);
    rect_to_cyl(x,y,&R, &phi);
    return *args * calcPlanarphiforce(R,phi,t, potentialArgs->nwrapped,
       potentialArgs->wrappedPotentialArg);
}
double RotateAndTiltWrapperPotentialPlanarR2deriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //calculate R2deriv
    double * rot = args+1;
    double x, y;
    cyl_to_rect(R,phi,&x, &y);
    rotate(&x,&y,&z,rot);
    rect_to_cyl(x,y,&R, &phi);
    return *args * calcPlanarR2deriv(R,phi,t, potentialArgs->nwrapped,
       potentialArgs->wrappedPotentialArg);
}
double RotateAndTiltWrapperPotentialPlanarphi2deriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //calculate phi2deriv
    double * rot = args+1;
    double x, y;
    cyl_to_rect(R,phi,&x, &y);
    rotate(&x,&y,&z,rot);
    rect_to_cyl(x,y,&R, &phi);
    return *args * calcPlanarphi2deriv(R,phi,t, potentialArgs->nwrapped,
       potentialArgs->wrappedPotentialArg);
}
double RotateAndTiltWrapperPotentialPlanarRphideriv(double R, double z, double phi,
        double t,
        struct potentialArg * potentialArgs){
    double * args= potentialArgs->args;
    //calculate Rphideriv
    double * rot = args+1;
    double x, y;
    cyl_to_rect(R,phi,&x, &y);
    rotate(&x,&y,&z,rot);
    rect_to_cyl(x,y,&R, &phi);
    return *args * calcPlanarRphideriv(R,phi,t, potentialArgs->nwrapped,
       potentialArgs->wrappedPotentialArg);
}
