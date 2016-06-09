#include <stdbool.h>
#include <math.h>
#include <galpy_potentials.h>
//General routines for TwoPowerTriaxialPotentials, specific potentials below
//TriaxialNFWPotential
inline void cyl_to_rect(double R, double phi,double *x, double *y){
  *x= R * cos ( phi );
  *y= R * sin ( phi );
}
inline void rotate(double *x, double *y, double *z, double *rot){
  double xp,yp,zp;
  xp= *(rot)   * *x + *(rot+1) * *y + *(rot+2) * *z;
  yp= *(rot+3) * *x + *(rot+4) * *y + *(rot+5) * *z;
  zp= *(rot+6) * *x + *(rot+7) * *y + *(rot+8) * *z;
  *x= xp;
  *y= yp;
  *z= zp;
}
inline void rotate_force(double *Fx, double *Fy, double *Fz, double *rot){
  double Fxp,Fyp,Fzp;
  Fxp= *(rot)   * *Fx + *(rot+3) * *Fy + *(rot+6) * *Fz;
  Fyp= *(rot+1) * *Fx + *(rot+4) * *Fy + *(rot+7) * *Fz;
  Fzp= *(rot+2) * *Fx + *(rot+5) * *Fy + *(rot+8) * *Fz;
  *Fx= Fxp;
  *Fy= Fyp;
  *Fz= Fzp;
}
inline double dens(double m, double a, double alpha, double beta){
  return pow ( a / m, alpha) * pow ( 1. + m / a, alpha - beta);
}
double TwoPowerTriaxialPotentialxforce_xyz_integrand(double s,
						     double x,double y,
						     double z,double a,
						     double alpha, double beta,
						     double b2,double c2){
  double t= 1. / s / s - 1.;
  return dens( sqrt ( x * x / ( 1. + t ) \
		      + y * y / ( b2 + t ) \
		      + z * z / ( c2 + t ) ),a,alpha,beta)\
    * x / ( 1. + t )							\
    / sqrt ( ( 1. + ( b2 - 1. ) * s * s ) * ( 1. + ( c2 - 1. ) * s * s ));
}
double TwoPowerTriaxialPotentialxforce_xyz(double x,double y, double z,
					   double a,
					   double alpha, double beta,
					   double b, double c, 
					   double b2, double c2,
					   int glorder,
					   double * glx, double * glw){
  double out= 0;
  int ii;
  for (ii=0; ii < glorder; ii++) 
    out+= *(glw+ii) \
      * TwoPowerTriaxialPotentialxforce_xyz_integrand(*(glx+ii),x,y,z,
						      a,alpha,beta,b2,c2);
  return -b * c / a / a / a * out; 
}
double TwoPowerTriaxialPotentialyforce_xyz_integrand(double s,
						     double x,double y,
						     double z,double a,
						     double alpha, double beta,
						     double b2,double c2){
  double t= 1. / s / s - 1.;
  return dens( sqrt ( x * x / ( 1. + t ) \
		      + y * y / ( b2 + t ) \
		      + z * z / ( c2 + t ) ),a,alpha,beta)\
    * y / ( b2 + t )							\
    / sqrt ( ( 1. + ( b2 - 1. ) * s * s ) * ( 1. + ( c2 - 1. ) * s * s ));
}
double TwoPowerTriaxialPotentialyforce_xyz(double x,double y, double z,
					   double a,
					   double alpha, double beta,
					   double b, double c, 
					   double b2, double c2,
					   int glorder,
					   double * glx, double * glw){
  double out= 0;
  int ii;
  for (ii=0; ii < glorder; ii++) 
    out+= *(glw+ii) \
      * TwoPowerTriaxialPotentialyforce_xyz_integrand(*(glx+ii),x,y,z,
						      a,alpha,beta,b2,c2);
  return -b * c / a / a / a * out; 
}
double TwoPowerTriaxialPotentialzforce_xyz_integrand(double s,
						     double x,double y,
						     double z,double a,
						     double alpha, double beta,
						     double b2,double c2){
  double t= 1. / s / s - 1.;
  return dens( sqrt ( x * x / ( 1. + t ) \
		      + y * y / ( b2 + t ) \
		      + z * z / ( c2 + t ) ),a,alpha,beta)\
    * z / ( c2 + t )							\
    / sqrt ( ( 1. + ( b2 - 1. ) * s * s ) * ( 1. + ( c2 - 1. ) * s * s ));
}
double TwoPowerTriaxialPotentialzforce_xyz(double x,double y, double z,
					   double a,
					   double alpha, double beta,
					   double b, double c, 
					   double b2, double c2,
					   int glorder,
					   double * glx, double * glw){
  double out= 0;
  int ii;
  for (ii=0; ii < glorder; ii++) 
    out+= *(glw+ii) \
      * TwoPowerTriaxialPotentialzforce_xyz_integrand(*(glx+ii),x,y,z,
						      a,alpha,beta,b2,c2);
  return -b * c / a / a / a * out; 
}
double TriaxialNFWPotentialRforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double b2= *args++;
  double c= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  Fx= TwoPowerTriaxialPotentialxforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					  glorder,glx,glw);
  Fy= TwoPowerTriaxialPotentialyforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					  glorder,glx,glw);
  if ( !aligned ) {
    Fz= TwoPowerTriaxialPotentialzforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					    glorder,glx,glw);
    rotate_force(&Fx,&Fy,&Fz,rot);
  }
  return amp * ( cos ( phi ) * Fx + sin( phi ) * Fy );
}
double TriaxialNFWPotentialphiforce(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double b2= *args++;
  double c= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  Fx= TwoPowerTriaxialPotentialxforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					  glorder,glx,glw);
  Fy= TwoPowerTriaxialPotentialyforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					  glorder,glx,glw);
  if ( !aligned ) {
    Fz= TwoPowerTriaxialPotentialzforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					    glorder,glx,glw);
    rotate_force(&Fx,&Fy,&Fz,rot);
  }
  return amp * R * ( -sin ( phi ) * Fx + cos( phi ) * Fy );
}
double TriaxialNFWPotentialzforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b= *args++;
  double b2= *args++;
  double c= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned ) 
    rotate(&x,&y,&z,rot);
  Fz= TwoPowerTriaxialPotentialzforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					  glorder,glx,glw);
  if ( !aligned ) {
    Fx= TwoPowerTriaxialPotentialxforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					    glorder,glx,glw);
    Fy= fg
      TwoPowerTriaxialPotentialyforce_xyz(x,y,z,a,1,3,b,c,b2,c2,
					    glorder,glx,glw);
    rotate_force(&Fx,&Fy,&Fz,rot);
  }
  return amp * Fz;
}
