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
  double * ellipargs= args + 17 + (int) *(args+16); // *(args+16) = num. arguments psi
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
					   args+17);
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
  double * ellipargs= args + 17 + (int) *(args+16); // *(args+16) = num. arguments dens
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
				 + z * z / ( c2 + t ) ),args+17);
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
  double * ellipargs= args + 17 + (int) *(args+16); // *(args+16) = num. arguments psi
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
				     args+17);
}

// Helper function to compute all second derivatives in xyz coordinates and cache them
void EllipsoidalPotential_2ndderiv_xyz_all(double (*dens)(double m, double * args),
					   double (*densDeriv)(double m, double * args),
					   double x, double y, double z,
					   double b2, double c2,
					   int glorder, double * glx, double * glw,
					   double * args,
					   double * phixx, double * phixy, double * phixz,
					   double * phiyy, double * phiyz, double * phizz) {
  int ii;
  double s, t, m;
  double t1, t2, t3;
  double dens_val, densDeriv_over_m;
  double x_t1, y_t2, z_t3;

  // Initialize output variables directly
  *phixx = 0.;
  *phixy = 0.;
  *phixz = 0.;
  *phiyy = 0.;
  *phiyz = 0.;
  *phizz = 0.;

  // Pre-compute x^2, y^2, z^2 outside the loop
  double x2 = x * x;
  double y2 = y * y;
  double z2 = z * z;

  for (ii = 0; ii < glorder; ii++) {
    s = *(glx + ii);
    t = 1. / s / s - 1.;

    // Pre-compute denominators
    t1 = 1. + t;
    t2 = b2 + t;
    t3 = c2 + t;

    // Calculate m
    m = sqrt(x2 / t1 + y2 / t2 + z2 / t3);

    // Compute dens and densDeriv once per iteration
    // Multiply with glw immediately
    dens_val = *(glw + ii) * dens(m, args+17);
    densDeriv_over_m = *(glw + ii) * densDeriv(m, args+17) / m;

    // Pre-compute x/t1, y/t2, z/t3 for reuse
    x_t1 = x / t1;
    y_t2 = y / t2;
    z_t3 = z / t3;

    // Calculate all 6 unique second derivatives
    // Note: glw already includes the -4*pi*b*c / sqrt(...) factor from Python glue code
    // The negative sign is for forces; for second derivatives we need positive, so we subtract
    // (which negates the negative sign in glw)

    // d^2phi/dx^2
    *phixx -= densDeriv_over_m * x_t1 * x_t1 + dens_val / t1;

    // d^2phi/dxdy
    *phixy -= densDeriv_over_m * x_t1 * y_t2;

    // LCOV_EXCL_START
    // d^2phi/dxdz
    *phixz -= densDeriv_over_m * x_t1 * z_t3;
    // LCOV_EXCL_STOP

    // d^2phi/dy^2
    *phiyy -= densDeriv_over_m * y_t2 * y_t2 + dens_val / t2;

    // LCOV_EXCL_START
    // d^2phi/dydz
    *phiyz -= densDeriv_over_m * y_t2 * z_t3;

    // d^2phi/dz^2
    *phizz -= densDeriv_over_m * z_t3 * z_t3 + dens_val / t3;
    // LCOV_EXCL_STOP
  }

  // Cache the position and results for second derivatives
  *(args + 7) = x;
  *(args + 8) = y;
  *(args + 9) = z;
  *(args + 10) = *phixx;
  *(args + 11) = *phixy;
  *(args + 12) = *phixz;
  *(args + 13) = *phiyy;
  *(args + 14) = *phiyz;
  *(args + 15) = *phizz;
}


double EllipsoidalPotentialPlanarR2deriv(double R, double phi, double t,
					 struct potentialArg * potentialArgs) {
  double * args = potentialArgs->args;
  double amp = *args;
  // Get caching args for second derivatives: x2,y2,z2 at indices 7,8,9
  double cached_x = *(args + 7);
  double cached_y = *(args + 8);
  double cached_z = *(args + 9);
  double * ellipargs = args + 17 + (int) *(args+16);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get second derivatives in xyz coordinates (with caching)
  // Only extract the ones we need for R2deriv: phixx, phixy, phiyy
  double phixx, phixy, phiyy;
  if (x == cached_x && y == cached_y && z == cached_z) {
    // LCOV_EXCL_START
    phixx = *(args + 10);
    phixy = *(args + 11);
    phiyy = *(args + 13);
    // LCOV_EXCL_STOP
  } else {
    double phixz, phiyz, phizz;
    EllipsoidalPotential_2ndderiv_xyz_all(potentialArgs->mdens,
					  potentialArgs->mdensDeriv,
					  x, y, z, b2, c2,
					  glorder, glx, glw, args,
					  &phixx, &phixy, &phixz,
					  &phiyy, &phiyz, &phizz);
  }

  // Transform to cylindrical: d^2phi/dR^2
  double cosphi = cos(phi);
  double sinphi = sin(phi);

  return amp * (cosphi * cosphi * phixx + sinphi * sinphi * phiyy +
		2. * cosphi * sinphi * phixy);
}

double EllipsoidalPotentialPlanarphi2deriv(double R, double phi, double t,
					   struct potentialArg * potentialArgs) {
  double * args = potentialArgs->args;
  double amp = *args;
  // Get caching args for forces: x,y,z at indices 1,2,3
  double cached_x_force = *(args + 1);
  double cached_y_force = *(args + 2);
  double cached_z_force = *(args + 3);
  // Get caching args for second derivatives: x2,y2,z2 at indices 7,8,9
  double cached_x_deriv = *(args + 7);
  double cached_y_deriv = *(args + 8);
  double cached_z_deriv = *(args + 9);
  double * ellipargs = args + 17 + (int) *(args+16);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get forces in xyz coordinates (with caching)
  // Only extract Fx and Fy (needed for this function)
  double Fx, Fy;
  if (x == cached_x_force && y == cached_y_force && z == cached_z_force) {
    Fx = *(args + 4);
    Fy = *(args + 5);
  } else {
    // LCOV_EXCL_START
    double Fz;
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens, x, y, z, &Fx, &Fy, &Fz, args);
    // LCOV_EXCL_STOP
  }

  // Get second derivatives in xyz coordinates (with caching)
  // Only extract the ones we need for phi2deriv: phixx, phixy, phiyy
  double phixx, phixy, phiyy;
  if (x == cached_x_deriv && y == cached_y_deriv && z == cached_z_deriv) {
    phixx = *(args + 10);
    phixy = *(args + 11);
    phiyy = *(args + 13);
  } else {
    // LCOV_EXCL_START
    double phixz, phiyz, phizz;
    EllipsoidalPotential_2ndderiv_xyz_all(potentialArgs->mdens,
					  potentialArgs->mdensDeriv,
					  x, y, z, b2, c2,
					  glorder, glx, glw, args,
					  &phixx, &phixy, &phixz,
					  &phiyy, &phiyz, &phizz);
    // LCOV_EXCL_STOP
  }

  // Transform to cylindrical: d^2phi/dphi^2
  double cosphi = cos(phi);
  double sinphi = sin(phi);

  return amp * (R * R * (sinphi * sinphi * phixx + cosphi * cosphi * phiyy -
			 2. * cosphi * sinphi * phixy) +
		R * (cosphi * Fx + sinphi * Fy));
}

double EllipsoidalPotentialPlanarRphideriv(double R, double phi, double t,
					   struct potentialArg * potentialArgs) {
  double * args = potentialArgs->args;
  double amp = *args;
  // Get caching args for forces: x,y,z at indices 1,2,3
  double cached_x_force = *(args + 1);
  double cached_y_force = *(args + 2);
  double cached_z_force = *(args + 3);
  // Get caching args for second derivatives: x2,y2,z2 at indices 7,8,9
  double cached_x_deriv = *(args + 7);
  double cached_y_deriv = *(args + 8);
  double cached_z_deriv = *(args + 9);
  double * ellipargs = args + 17 + (int) *(args+16);
  double b2 = *ellipargs++;
  double c2 = *ellipargs++;
  bool aligned = (bool) *ellipargs++;
  double * rot = ellipargs;
  ellipargs += 9;
  int glorder = (int) *ellipargs++;
  double * glx = ellipargs;
  double * glw = ellipargs + glorder;

  // Convert to Cartesian (z=0 for planar)
  double x, y, z = 0.;
  cyl_to_rect(R, phi, &x, &y);

  // Get forces in xyz coordinates (with caching)
  // Only extract Fx and Fy (needed for this function)
  double Fx, Fy;
  if (x == cached_x_force && y == cached_y_force && z == cached_z_force) {
    Fx = *(args + 4);
    Fy = *(args + 5);
  } else {
    // LCOV_EXCL_START
    double Fz;
    EllipsoidalPotentialxyzforces_xyz(potentialArgs->mdens, x, y, z, &Fx, &Fy, &Fz, args);
    // LCOV_EXCL_STOP
  }

  // Get second derivatives in xyz coordinates (with caching)
  // Only extract the ones we need for Rphideriv: phixx, phixy, phiyy
  double phixx, phixy, phiyy;
  if (x == cached_x_deriv && y == cached_y_deriv && z == cached_z_deriv) {
    phixx = *(args + 10);
    phixy = *(args + 11);
    phiyy = *(args + 13);
  } else {
    // LCOV_EXCL_START
    double phixz, phiyz, phizz;
    EllipsoidalPotential_2ndderiv_xyz_all(potentialArgs->mdens,
					  potentialArgs->mdensDeriv,
					  x, y, z, b2, c2,
					  glorder, glx, glw, args,
					  &phixx, &phixy, &phixz,
					  &phiyy, &phiyz, &phizz);
    // LCOV_EXCL_STOP
  }

  // Transform to cylindrical: d^2phi/dRdphi
  double cosphi = cos(phi);
  double sinphi = sin(phi);
  double cos2phi = cos(2. * phi);

  return amp * (R * (cosphi * sinphi * (phiyy - phixx) + cos2phi * phixy) +
		sinphi * Fx - cosphi * Fy);
}
