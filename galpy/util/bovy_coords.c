#include <math.h>
#include <bovy_coords.h>
/*
NAME: cyl_to_rect
PURPOSE: convert 2D (R,phi) to (x,y) [mainly used in the context of cylindrical coordinates, hence the name)
INPUT:
   double R - cylindrical radius
   double phi - azimuth (rad)
OUTPUT (as arguments):
   double *x - x
   double *y - y
 */
void cyl_to_rect(double R, double phi,double *x, double *y){
  *x= R * cos ( phi );
  *y= R * sin ( phi );
}
/*
NAME: rect_to_cyl
PURPOSE: convert 2D (x,y) to (R,phi) [mainly used in the context of cylindrical coordinates, hence the name)
INPUT:
   double x - x
   double y - y
OUTPUT (as arguments):
   double *R - cylindrical radius
   double *phi - azimuth (rad)
 */
void rect_to_cyl(double x, double y,double *R, double *phi){
  *R = sqrt (x*x+y*y);
  *phi = atan2 ( y , x );
}
/*
NAME: polar_to_rect_galpy
PURPOSE: convert (R,vR,vT,phi) to (x,y,vx,vy)
INPUT:
   double * vxvv - (R,vR,vT,phi)
OUTPUT:
   performed in-place
HISTORY: 2012-12-24 - Written - Bovy (UofT)
 */
void polar_to_rect_galpy(double *vxvv){
  double R,phi,cp,sp,vR,vT;
  R  = *vxvv;
  phi= *(vxvv+3);
  cp = cos ( phi );
  sp = sin ( phi );
  vR = *(vxvv+1);
  vT = *(vxvv+2);
  *vxvv    = R * cp;
  *(vxvv+1)= R * sp;
  *(vxvv+2)= vR * cp - vT * sp;
  *(vxvv+3)= vR * sp + vT * cp;
}
/*
NAME: rect_to_polar_galpy
PURPOSE: convert (x,y,vx,vy) to (R,vR,vT,phi)
INPUT:
   double * vxvv - (x,y,vx,vy)
OUTPUT (as arguments):
   performed in-place
HISTORY: 2012-12-24 - Written - Bovy (UofT)
 */
void rect_to_polar_galpy(double *vxvv){
  double x,y,vx,vy,cp,sp;
  x = *vxvv;
  y = *(vxvv+1);
  vx= *(vxvv+2);
  vy= *(vxvv+3);
  *(vxvv+3)= atan2( y , x );
  cp = cos ( *(vxvv+3) );
  sp = sin ( *(vxvv+3) );
  *vxvv    = sqrt ( x * x + y * y );
  *(vxvv+1)=  vx * cp + vy * sp;
  *(vxvv+2)= -vx * sp + vy * cp;
}
/*
NAME: cyl_to_rect_galpy
PURPOSE: convert (R,vR,vT,z,vz,phi) to (x,y,z,vx,vy,vz)
INPUT:
   double * vxvv - (R,vR,vT,z,vz,phi)
OUTPUT:
   performed in-place
HISTORY: 2012-12-24 - Written - Bovy (UofT)
 */
void cyl_to_rect_galpy(double *vxvv){
  double R,phi,cp,sp,vR,vT;
  R  = *vxvv;
  phi= *(vxvv+5);
  cp = cos ( phi );
  sp = sin ( phi );
  vR = *(vxvv+1);
  vT = *(vxvv+2);
  *vxvv    = R * cp;
  *(vxvv+1)= R * sp;
  *(vxvv+2)= *(vxvv+3);
  *(vxvv+5)= *(vxvv+4);
  *(vxvv+3)= vR * cp - vT * sp;
  *(vxvv+4)= vR * sp + vT * cp;
}
/*
NAME: rect_to_cyl_galpy
PURPOSE: convert (x,y,z,vx,vy,vz) to (R,vR,vT,z,vzphi)
INPUT:
   double * vxvv - (x,y,z,vx,vy,vz)
OUTPUT (as arguments):
   performed in-place
HISTORY: 2012-12-24 - Written - Bovy (UofT)
 */
void rect_to_cyl_galpy(double *vxvv){
  double x,y,vx,vy,cp,sp;
  x = *vxvv;
  y = *(vxvv+1);
  vx= *(vxvv+3);
  vy= *(vxvv+4);
  *(vxvv+3)= *(vxvv+2);
  *(vxvv+4)= *(vxvv+5);
  *(vxvv+5)= atan2( y , x );
  cp = cos ( *(vxvv+5) );
  sp = sin ( *(vxvv+5) );
  *vxvv    = sqrt ( x * x + y * y );
  *(vxvv+1)=  vx * cp + vy * sp;
  *(vxvv+2)= -vx * sp + vy * cp;
}
