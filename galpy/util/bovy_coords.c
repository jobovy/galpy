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
NAME: cyl_to_rect_vec
PURPOSE: convert 2D (vR,vT) to (vx,vy) [mainly used in the context of cylindrical coordinates, hence the name)
INPUT:
   double vR - cylindrical radial velocity
   double vT - cylindrical rotational velocity
   double phi - azimuth (rad)
OUTPUT (as arguments):
   double *vx - vx
   double *vy - vy
 */
void cyl_to_rect_vec(double vR, double vT, double phi,double * vx, double * vy){
  double cp= cos(phi);
  double sp= sin(phi);
  *vx= vR * cp - vT * sp;
  *vy= vR * sp + vT * cp;
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
/*
NAME: cyl_to_sos_galpy
PURPOSE: convert (R,vR,vT,z,vz,phi,t) to (x,y,vx,vy,A,t,psi) coordinates for SOS integration
INPUT:
   double * vxvv - (R,vR,vT,z,vz,phi,t)
OUTPUT:
   performed in-place
HISTORY: 2023-03-19 - Written - Bovy (UofT)
 */
void cyl_to_sos_galpy(double *vxvv){
  double R,cp,sp,vR,vT;
  R  = *vxvv;
  cp = cos ( *(vxvv+5) );
  sp = sin ( *(vxvv+5) );
  vR = *(vxvv+1);
  vT = *(vxvv+2);
  *(vxvv+5)= *(vxvv+6);
  *(vxvv+6)= atan2( *(vxvv+3) , *(vxvv+4) );
  *(vxvv+4)= sqrt( *(vxvv+3) * *(vxvv+3) + *(vxvv+4) * *(vxvv+4) );
  *vxvv    = R * cp;
  *(vxvv+1)= R * sp;
  *(vxvv+2)= vR * cp - vT * sp;
  *(vxvv+3)= vR * sp + vT * cp;
}
/*
NAME: sos_to_cyl_galpy
PURPOSE: convert (x,y,vx,vy,A,t,psi) to (R,vR,vT,z,vz,phi,t)
INPUT:
   double * vxvv - (x,y,vx,vy,A,t,psi)
OUTPUT (as arguments):
   performed in-place
HISTORY: 2023-03-19 - Written - Bovy (UofT)
 */
void sos_to_cyl_galpy(double *vxvv){
  double x,y,vx,vy,phi,cp,sp;
  x = *(vxvv  );
  y = *(vxvv+1);
  vx= *(vxvv+2);
  vy= *(vxvv+3);
  phi= atan2( y , x );
  cp = cos ( phi );
  sp = sin ( phi );
  *(vxvv  )= sqrt ( x * x + y * y );
  *(vxvv+1)=  vx * cp + vy * sp;
  *(vxvv+2)= -vx * sp + vy * cp;
  *(vxvv+3)= *(vxvv+4) * sin ( *(vxvv+6) );
  *(vxvv+4)= *(vxvv+4) * cos ( *(vxvv+6) );
  *(vxvv+6)= *(vxvv+5);
  *(vxvv+5)= phi;
}
/*
NAME: polar_to_sos_galpy
PURPOSE: convert (R,vR,vT,phi,t) to (x,vx,A,t,psi) coordinates for SOS integration [or (y,vy,A,t) if surface == 1]
INPUT:
   double * vxvv - (R,vR,vT,z,vz,phi,t)
   int surface - if 1, convert to (x,vx,A,t) instead of (y,vy,A,t)
OUTPUT:
   performed in-place
HISTORY: 2023-03-24 - Written - Bovy (UofT)
 */
void polar_to_sos_galpy(double *vxvv,int surface){
  double R,cp,sp,vR,vT,x,y,vx,vy;
  R  = *vxvv;
  cp = cos ( *(vxvv+3) );
  sp = sin ( *(vxvv+3) );
  vR = *(vxvv+1);
  vT = *(vxvv+2);
  if ( surface == 1 ) { // surface: y=0, vy > 0
    *(vxvv  )= R * cp;
    *(vxvv+1)= vR * cp - vT * sp;
    y= R * sp;
    vy= vR * sp + vT * cp;
    *(vxvv+2)= sqrt ( y * y + vy * vy );
    *(vxvv+3)= *(vxvv+4);
    *(vxvv+4)= atan2( y , vy );
  } else { // surface: x=0, vx > 0
    *(vxvv  )= R * sp;
    *(vxvv+1)= vR * sp + vT * cp;
    x= R * cp;
    vx= vR * cp - vT * sp;
    *(vxvv+2)= sqrt ( x * x + vx * vx );
    *(vxvv+3)= *(vxvv+4);
    *(vxvv+4)= atan2( x , vx );
  }
}
/*
NAME: sos_to_polar_galpy
PURPOSE: convert (y,vy,A,t,psi) or (x,vx,A,t,psi) to (R,vR,vT,phi,t )
INPUT:
   double * vxvv - (x,y,vx,vy,A,t,psi)
   surface - if 1, convert from (y,vy,A,t) instead of (x,vx,A,t)
OUTPUT (as arguments):
   performed in-place
HISTORY: 2023-03-24 - Written - Bovy (UofT)
 */
void sos_to_polar_galpy(double *vxvv,int surface){
  double x,y,vx,vy,phi,cp,sp;
  if ( surface == 1) { // surface: y=0, vy > 0
    x= *(vxvv  );
    vx= *(vxvv+1);
    y= *(vxvv+2) * sin ( *(vxvv+4) );
    vy= *(vxvv+2) * cos ( *(vxvv+4) );
  } else { // surface: x=0, vx > 0
    y= *(vxvv  );
    vy= *(vxvv+1);
    x= *(vxvv+2) * sin ( *(vxvv+4) );
    vx= *(vxvv+2) * cos ( *(vxvv+4) );
  }
  phi= atan2( y , x );
  cp = cos ( phi );
  sp = sin ( phi );
  *(vxvv  )= sqrt ( x * x + y * y );
  *(vxvv+1)=  vx * cp + vy * sp;
  *(vxvv+2)= -vx * sp + vy * cp;
  *(vxvv+4)= *(vxvv+3);
  *(vxvv+3)= phi;
}
