#include <stdbool.h>
#include <math.h>
#include <bovy_coords.h>
#include <galpy_potentials.h>
//General routines for EllipsoidalPotentials
double EllipsoidalPotentialEval(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  int ii;
  double s;
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  double * ellipargs= args + 8 + (int) *(args+7); // *(args+7) = num. arguments psi
  double b2= *ellipargs++;
  double c2= *ellipargs++;
  bool aligned= (bool) *ellipargs++;
  double * rot= ellipargs;
  ellipargs+= 9;
  int glorder= (int) *ellipargs++;
  double * glx= ellipargs;
  double * glw= ellipargs + glorder;
  //Calculate potential
  double x, y;
  double out= 0.;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  for (ii=0; ii < glorder; ii++) {
    s= 1. / *(glx+ii) / *(glx+ii) - 1.;
    out+= *(glw+ii) * potentialArgs->psi ( sqrt (  x * x / ( 1. + s )
	 					 + y * y / ( b2 + s )
		 				 + z * z / ( c2 + s ) ),
					   args+8);
  }
  return -0.5 * amp * out;
}
void EllipsoidalPotentialxyzforces_xyz(double (*dens)(double m,
						      double * args),
				       double x,double y, double z,
				       double * Fx, double * Fy,
				       double * Fz,double * args){
  int ii;
  double t;
  double td;
  //Get args
  double * ellipargs= args + 8 + (int) *(args+7); // *(args+7) = num. arguments dens
  double b2= *ellipargs++;
  double c2= *ellipargs++;
  bool aligned= (bool) *ellipargs++;
  double * rot= ellipargs;
  ellipargs+= 9;
  int glorder= (int) *ellipargs++;
  double * glx= ellipargs;
  double * glw= ellipargs + glorder;
  //Setup caching
  *(args + 1)= x;
  *(args + 2)= y;
  *(args + 3)= z;
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  *Fx= 0.;
  *Fy= 0.;
  *Fz= 0.;
  for (ii=0; ii < glorder; ii++) {
    t= 1. / *(glx+ii) / *(glx+ii) - 1.;
    td= *(glw+ii) * dens( sqrt ( x * x / ( 1. + t )	+ y * y / ( b2 + t ) \
				 + z * z / ( c2 + t ) ),args+8);
    *Fx+= td * x / ( 1. + t );
    *Fy+= td * y / ( b2 + t );
    *Fz+= td * z / ( c2 + t );
  }
  if ( !aligned )
    rotate_force(Fx,Fy,Fz,rot);
  *(args + 4)= *Fx;
  *(args + 5)= *Fy;
  *(args + 6)= *Fz;
}
double EllipsoidalPotentialRforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz
  double cached_x= *(args + 1);
  double cached_y= *(args + 2);
  double cached_z= *(args + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    // LCOV_EXCL_START
    Fx= *(args + 4);
    Fy= *(args + 5);
    Fz= *(args + 6);
    // LCOV_EXCL_STOP
  }
  else
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens,
				      x,y,z,&Fx,&Fy,&Fz,args);
  return amp * ( cos ( phi ) * Fx + sin( phi ) * Fy );
}
double EllipsoidalPotentialphitorque(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz
  double cached_x= *(args + 1);
  double cached_y= *(args + 2);
  double cached_z= *(args + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + 4);
    Fy= *(args + 5);
    Fz= *(args + 6);
  }
  else
    // LCOV_EXCL_START
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens,
				      x,y,z,&Fx,&Fy,&Fz,args);
    // LCOV_EXCL_STOP
  return amp * R * ( -sin ( phi ) * Fx + cos( phi ) * Fy );
}
double EllipsoidalPotentialzforce(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  double amp= *args;
  // Get caching args: amp = 0, x,y,z,Fx,Fy,Fz
  double cached_x= *(args + 1);
  double cached_y= *(args + 2);
  double cached_z= *(args + 3);
  //Calculate potential
  double x, y;
  double Fx, Fy, Fz;
  cyl_to_rect(R,phi,&x,&y);
  if ( x == cached_x && y == cached_y && z == cached_z ){
    Fx= *(args + + 4);
    Fy= *(args + + 5);
    Fz= *(args + + 6);
  }
  else
    // LCOV_EXCL_START
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens,
				      x,y,z,&Fx,&Fy,&Fz,args);
    // LCOV_EXCL_STOP
  return amp * Fz;
}

double EllipsoidalPotentialPlanarRforce(double R,double phi,double t,
					struct potentialArg * potentialArgs){
  return EllipsoidalPotentialRforce(R,0.,phi,t,potentialArgs);
}
double EllipsoidalPotentialPlanarphitorque(double R,double phi,double t,
					  struct potentialArg * potentialArgs){
  return EllipsoidalPotentialphitorque(R,0.,phi,t,potentialArgs);
}
double EllipsoidalPotentialDens(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  //Get args
  double * args= potentialArgs->args;
  double amp= *args;
  double * ellipargs= args + 8 + (int) *(args+7); // *(args+7) = num. arguments psi
  double b2= *ellipargs++;
  double c2= *ellipargs++;
  bool aligned= (bool) *ellipargs++;
  double * rot= ellipargs;
  ellipargs+= 9;
  //Calculate density
  double x, y;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  return amp * potentialArgs->mdens ( sqrt (x * x + y * y / b2 + z * z / c2 ),
				     args+8);
}

/*

// Implement the potentials separately
//NFW
static inline double TriaxialNFWPotential_psi(double x){
  return 1. / ( 1. + x );
}
double TriaxialNFWPotentialpotential_xyz_integrand(double s,
						   double x,double y,
						   double z,double a,
						   double b2,double c2){
  double t= 1. / s / s - 1.;
  return TriaxialNFWPotential_psi( sqrt ( x * x / ( 1. + t ) \
					  + y * y / ( b2 + t )	\
					  + z * z / ( c2 + t ) ) / a );
}
//Hernquist
static inline double TriaxialHernquistPotential_psi(double x){
  return 0.5 / ( 1. + x ) / ( 1. + x );
}
double TriaxialHernquistPotentialpotential_xyz_integrand(double s,
						   double x,double y,
						   double z,double a,
						   double b2,double c2){
  double t= 1. / s / s - 1.;
  return TriaxialHernquistPotential_psi( sqrt ( x * x / ( 1. + t ) \
						+ y * y / ( b2 + t )	\
						+ z * z / ( c2 + t ) ) / a );
}
double TriaxialHernquistPotentialEval(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  int ii;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double out= 0.;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  for (ii=0; ii < glorder; ii++)
    out+= *(glw+ii) * a * a						\
      * TriaxialHernquistPotentialpotential_xyz_integrand(*(glx+ii),x,y,z,
							  a,b2,c2);
  return amp * out;
}
//Jaffe
static inline double TriaxialJaffePotential_psi(double x){
  return - 1. / ( 1. + x ) - log ( x / ( 1. + x ) ) ;
}
double TriaxialJaffePotentialpotential_xyz_integrand(double s,
						   double x,double y,
						   double z,double a,
						   double b2,double c2){
  double t= 1. / s / s - 1.;
  return TriaxialJaffePotential_psi( sqrt ( x * x / ( 1. + t )		\
					    + y * y / ( b2 + t )	\
					    + z * z / ( c2 + t ) ) / a );
}
double TriaxialJaffePotentialEval(double R,double z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  int ii;
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args++;
  double b2= *args++;
  double c2= *args++;
  bool aligned= (bool) *args++;
  double * rot= args;
  args+= 9;
  int glorder= (int) *args++;
  double * glx= args;
  double * glw= args + glorder;
  //Calculate potential
  double x, y;
  double out= 0.;
  cyl_to_rect(R,phi,&x,&y);
  if ( !aligned )
    rotate(&x,&y,&z,rot);
  for (ii=0; ii < glorder; ii++)
    out+= *(glw+ii) * a * a						\
      * TriaxialJaffePotentialpotential_xyz_integrand(*(glx+ii),x,y,z,
						      a,b2,c2);
  return amp * out;
}

*/
