/*
  Wrappers around the C integration code for Full Orbits
*/
#ifdef _WIN32
#include <Python.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <bovy_coords.h>
#include <bovy_symplecticode.h>
#include <leung_dop853.h>
#include <bovy_rk.h>
#include <wez_ias15.h>
#include <integrateFullOrbit.h>
//Potentials
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef ORBITS_CHUNKSIZE
#define ORBITS_CHUNKSIZE 1
#endif
//Macros to export functions in DLL on different OS
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
// Just do nothing?
#define EXPORT
#endif
#ifdef _WIN32
// On Windows, *need* to define this function to allow the package to be imported
#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_libgalpy(void) { // Python 3
  return NULL;
}
#else
PyMODINIT_FUNC initlibgalpy(void) {} // Python 2
#endif
#endif
/*
  Function Declarations
*/
void evalRectForce(double, double *, double *,
		   int, struct potentialArg *);
void evalRectDeriv(double, double *, double *,
			 int, struct potentialArg *);
void evalSOSDeriv(double, double *, double *,
			 int, struct potentialArg *);
void evalRectDeriv_dxdv(double,double *, double *,
			      int, struct potentialArg *);
void initMovingObjectSplines(struct potentialArg *, double ** pot_args);
void initChandrasekharDynamicalFrictionSplines(struct potentialArg *, double ** pot_args);
/*
  Actual functions
*/
void parse_leapFuncArgs_Full(int npot,
			     struct potentialArg * potentialArgs,
			     int ** pot_type,
			     double ** pot_args,
           tfuncs_type_arr * pot_tfuncs){
  int ii,jj,kk;
  int nR, nz, nr;
  double * Rgrid, * zgrid, * potGrid_splinecoeffs;
  init_potentialArgs(npot,potentialArgs);
  for (ii=0; ii < npot; ii++){
    switch ( *(*pot_type)++ ) {
    case 0: //LogarithmicHaloPotential, 4 arguments
      potentialArgs->potentialEval= &LogarithmicHaloPotentialEval;
      potentialArgs->Rforce= &LogarithmicHaloPotentialRforce;
      potentialArgs->zforce= &LogarithmicHaloPotentialzforce;
      potentialArgs->phitorque= &LogarithmicHaloPotentialphitorque;
      potentialArgs->dens= &LogarithmicHaloPotentialDens;
      potentialArgs->R2deriv= &LogarithmicHaloPotentialR2deriv;
      potentialArgs->z2deriv= &LogarithmicHaloPotentialz2deriv;
      potentialArgs->phi2deriv= &LogarithmicHaloPotentialphi2deriv;
      potentialArgs->Rzderiv= &LogarithmicHaloPotentialRzderiv;
      potentialArgs->Rphideriv= &LogarithmicHaloPotentialRphideriv;
      potentialArgs->zphideriv= &LogarithmicHaloPotentialzphideriv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 1: //DehnenBarPotential, 6 arguments
      potentialArgs->Rforce= &DehnenBarPotentialRforce;
      potentialArgs->phitorque= &DehnenBarPotentialphitorque;
      potentialArgs->zforce= &DehnenBarPotentialzforce;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv= &DehnenBarPotentialR2deriv;
      potentialArgs->z2deriv= &DehnenBarPotentialz2deriv;
      potentialArgs->phi2deriv= &DehnenBarPotentialphi2deriv;
      potentialArgs->Rzderiv= &DehnenBarPotentialRzderiv;
      potentialArgs->Rphideriv= &DehnenBarPotentialRphideriv;
      potentialArgs->zphideriv= &DehnenBarPotentialzphideriv;
      potentialArgs->nargs= 6;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->potentialEval= &MiyamotoNagaiPotentialEval;
      potentialArgs->Rforce= &MiyamotoNagaiPotentialRforce;
      potentialArgs->zforce= &MiyamotoNagaiPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &MiyamotoNagaiPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      // Axisymmetric: phi2deriv/Rphideriv/zphideriv are 0 -> left NULL
      // (the NULL-safe aggregators return 0 for them).
      potentialArgs->R2deriv= &MiyamotoNagaiPotentialR2deriv;
      potentialArgs->z2deriv= &MiyamotoNagaiPotentialz2deriv;
      potentialArgs->Rzderiv= &MiyamotoNagaiPotentialRzderiv;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialEval;
      potentialArgs->Rforce= &PowerSphericalPotentialRforce;
      potentialArgs->zforce= &PowerSphericalPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &PowerSphericalPotentialDens;
      potentialArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      potentialArgs->z2deriv= &PowerSphericalPotentialz2deriv;
      potentialArgs->Rzderiv= &PowerSphericalPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 8: //HernquistPotential, 2 arguments
      potentialArgs->potentialEval= &HernquistPotentialEval;
      potentialArgs->Rforce= &HernquistPotentialRforce;
      potentialArgs->zforce= &HernquistPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &HernquistPotentialDens;
      potentialArgs->R2deriv= &HernquistPotentialR2deriv;
      potentialArgs->z2deriv= &HernquistPotentialz2deriv;
      potentialArgs->Rzderiv= &HernquistPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->potentialEval= &NFWPotentialEval;
      potentialArgs->Rforce= &NFWPotentialRforce;
      potentialArgs->zforce= &NFWPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &NFWPotentialDens;
      potentialArgs->R2deriv= &NFWPotentialR2deriv;
      potentialArgs->z2deriv= &NFWPotentialz2deriv;
      potentialArgs->Rzderiv= &NFWPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 10: //JaffePotential, 2 arguments
      potentialArgs->potentialEval= &JaffePotentialEval;
      potentialArgs->Rforce= &JaffePotentialRforce;
      potentialArgs->zforce= &JaffePotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &JaffePotentialDens;
      potentialArgs->R2deriv= &JaffePotentialR2deriv;
      potentialArgs->z2deriv= &JaffePotentialz2deriv;
      potentialArgs->Rzderiv= &JaffePotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      potentialArgs->potentialEval= &DoubleExponentialDiskPotentialEval;
      potentialArgs->Rforce= &DoubleExponentialDiskPotentialRforce;
      potentialArgs->zforce= &DoubleExponentialDiskPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &DoubleExponentialDiskPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      // Axisymmetric: phi2deriv/Rphideriv/zphideriv are 0 -> left NULL
      // (the NULL-safe aggregators return 0 for them). The R2deriv/z2deriv/
      // Rzderiv use the same Ogata/Hankel quadrature (J0/J1 nodes) as the forces.
      potentialArgs->R2deriv= &DoubleExponentialDiskPotentialR2deriv;
      potentialArgs->z2deriv= &DoubleExponentialDiskPotentialz2deriv;
      potentialArgs->Rzderiv= &DoubleExponentialDiskPotentialRzderiv;
      //Look at pot_args to figure out the number of arguments
      potentialArgs->nargs= (int) (5 + 4 * *(*pot_args+4) );
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      potentialArgs->potentialEval= &FlattenedPowerPotentialEval;
      potentialArgs->Rforce= &FlattenedPowerPotentialRforce;
      potentialArgs->zforce= &FlattenedPowerPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &FlattenedPowerPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      // Axisymmetric: phi2deriv/Rphideriv/zphideriv are 0 -> left NULL
      // (the NULL-safe aggregators return 0 for them).
      potentialArgs->R2deriv= &FlattenedPowerPotentialR2deriv;
      potentialArgs->z2deriv= &FlattenedPowerPotentialz2deriv;
      potentialArgs->Rzderiv= &FlattenedPowerPotentialRzderiv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 13: //interpRZPotential, XX arguments
      //Grab the grids and the coefficients
      nR= (int) *(*pot_args)++;
      nz= (int) *(*pot_args)++;
      Rgrid= (double *) malloc ( nR * sizeof ( double ) );
      zgrid= (double *) malloc ( nz * sizeof ( double ) );
      potGrid_splinecoeffs= (double *) malloc ( nR * nz * sizeof ( double ) );
      for (kk=0; kk < nR; kk++)
	*(Rgrid+kk)= *(*pot_args)++;
      for (kk=0; kk < nz; kk++)
	*(zgrid+kk)= *(*pot_args)++;
      for (kk=0; kk < nR; kk++)
	put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz);
      *pot_args+= nR*nz;
      potentialArgs->i2d= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2d,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accx= gsl_interp_accel_alloc ();
      potentialArgs->accy= gsl_interp_accel_alloc ();
      for (kk=0; kk < nR; kk++)
	put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz);
      *pot_args+= nR*nz;
      potentialArgs->i2drforce= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2drforce,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accxrforce= gsl_interp_accel_alloc ();
      potentialArgs->accyrforce= gsl_interp_accel_alloc ();
      for (kk=0; kk < nR; kk++)
	put_row(potGrid_splinecoeffs,kk,*pot_args+kk*nz,nz);
      *pot_args+= nR*nz;
      potentialArgs->i2dzforce= interp_2d_alloc(nR,nz);
      interp_2d_init(potentialArgs->i2dzforce,Rgrid,zgrid,potGrid_splinecoeffs,
		     INTERP_2D_LINEAR); //latter bc we already calculated the coeffs
      potentialArgs->accxzforce= gsl_interp_accel_alloc ();
      potentialArgs->accyzforce= gsl_interp_accel_alloc ();
      potentialArgs->potentialEval= &interpRZPotentialEval;
      potentialArgs->Rforce= &interpRZPotentialRforce;
      potentialArgs->zforce= &interpRZPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      //clean up
      free(Rgrid);
      free(zgrid);
      free(potGrid_splinecoeffs);
      potentialArgs->requiresVelocity= false;
      break;
    case 14: //IsochronePotential, 2 arguments
      potentialArgs->potentialEval= &IsochronePotentialEval;
      potentialArgs->Rforce= &IsochronePotentialRforce;
      potentialArgs->zforce= &IsochronePotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &IsochronePotentialDens;
      potentialArgs->R2deriv= &IsochronePotentialR2deriv;
      potentialArgs->z2deriv= &IsochronePotentialz2deriv;
      potentialArgs->Rzderiv= &IsochronePotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 15: //PowerSphericalwCutoffPotential, 5 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialwCutoffEval;
      potentialArgs->Rforce= &PowerSphericalPotentialwCutoffRforce;
      potentialArgs->zforce= &PowerSphericalPotentialwCutoffzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &PowerSphericalPotentialwCutoffDens;
      potentialArgs->R2deriv= &PowerSphericalPotentialwCutoffR2deriv;
      potentialArgs->z2deriv= &PowerSphericalPotentialwCutoffz2deriv;
      potentialArgs->Rzderiv= &PowerSphericalPotentialwCutoffRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 5;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 16: //KuzminKutuzovStaeckelPotential, 3 arguments
      potentialArgs->potentialEval= &KuzminKutuzovStaeckelPotentialEval;
      potentialArgs->Rforce= &KuzminKutuzovStaeckelPotentialRforce;
      potentialArgs->zforce= &KuzminKutuzovStaeckelPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      // Axisymmetric: phi2deriv/Rphideriv/zphideriv are 0 -> left NULL
      // (the NULL-safe aggregators return 0 for them).
      potentialArgs->R2deriv= &KuzminKutuzovStaeckelPotentialR2deriv;
      potentialArgs->z2deriv= &KuzminKutuzovStaeckelPotentialz2deriv;
      potentialArgs->Rzderiv= &KuzminKutuzovStaeckelPotentialRzderiv;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 17: //PlummerPotential, 2 arguments
      potentialArgs->potentialEval= &PlummerPotentialEval;
      potentialArgs->Rforce= &PlummerPotentialRforce;
      potentialArgs->zforce= &PlummerPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &PlummerPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      // Axisymmetric: phi2deriv/Rphideriv/zphideriv are 0 -> left NULL
      // (the NULL-safe aggregators return 0 for them).
      potentialArgs->R2deriv= &PlummerPotentialR2deriv;
      potentialArgs->z2deriv= &PlummerPotentialz2deriv;
      potentialArgs->Rzderiv= &PlummerPotentialRzderiv;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 18: //PseudoIsothermalPotential, 2 arguments
      potentialArgs->potentialEval= &PseudoIsothermalPotentialEval;
      potentialArgs->Rforce= &PseudoIsothermalPotentialRforce;
      potentialArgs->zforce= &PseudoIsothermalPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &PseudoIsothermalPotentialDens;
      potentialArgs->R2deriv= &PseudoIsothermalPotentialR2deriv;
      potentialArgs->z2deriv= &PseudoIsothermalPotentialz2deriv;
      potentialArgs->Rzderiv= &PseudoIsothermalPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 19: //KuzminDiskPotential, 2 arguments
      potentialArgs->potentialEval= &KuzminDiskPotentialEval;
      potentialArgs->Rforce= &KuzminDiskPotentialRforce;
      potentialArgs->zforce= &KuzminDiskPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      // Axisymmetric: phi2deriv/Rphideriv/zphideriv are 0 -> left NULL
      // (the NULL-safe aggregators return 0 for them).
      potentialArgs->R2deriv= &KuzminDiskPotentialR2deriv;
      potentialArgs->z2deriv= &KuzminDiskPotentialz2deriv;
      potentialArgs->Rzderiv= &KuzminDiskPotentialRzderiv;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 20: //BurkertPotential, 2 arguments
      potentialArgs->potentialEval= &BurkertPotentialEval;
      potentialArgs->Rforce= &BurkertPotentialRforce;
      potentialArgs->zforce= &BurkertPotentialzforce;
      potentialArgs->dens= &BurkertPotentialDens;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->R2deriv= &BurkertPotentialR2deriv;
      potentialArgs->z2deriv= &BurkertPotentialz2deriv;
      potentialArgs->Rzderiv= &BurkertPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 21: //TriaxialHernquistPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      potentialArgs->zphideriv = &EllipsoidalPotentialzphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialHernquistPotentialpsi;
      potentialArgs->mdens= &TriaxialHernquistPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialHernquistPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 22: //TriaxialNFWPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      potentialArgs->zphideriv = &EllipsoidalPotentialzphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialNFWPotentialpsi;
      potentialArgs->mdens= &TriaxialNFWPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialNFWPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 23: //TriaxialJaffePotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      potentialArgs->zphideriv = &EllipsoidalPotentialzphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialJaffePotentialpsi;
      potentialArgs->mdens= &TriaxialJaffePotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialJaffePotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 24: //SCFPotential, many arguments
      potentialArgs->potentialEval= &SCFPotentialEval;
      potentialArgs->Rforce= &SCFPotentialRforce;
      potentialArgs->zforce= &SCFPotentialzforce;
      potentialArgs->phitorque= &SCFPotentialphitorque;
      potentialArgs->dens= &SCFPotentialDens;
      potentialArgs->R2deriv= &SCFPotentialR2deriv;
      potentialArgs->z2deriv= &SCFPotentialz2deriv;
      potentialArgs->Rzderiv= &SCFPotentialRzderiv;
      potentialArgs->phi2deriv= &SCFPotentialphi2deriv;
      potentialArgs->Rphideriv= &SCFPotentialRphideriv;
      potentialArgs->zphideriv= &SCFPotentialzphideriv;
      potentialArgs->nargs= (int) (5 + (1 + *(*pot_args + 1)) * *(*pot_args+2) * *(*pot_args+3)* *(*pot_args+4) + 10);
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 25: //SoftenedNeedleBarPotential, 13 arguments
      potentialArgs->potentialEval= &SoftenedNeedleBarPotentialEval;
      potentialArgs->Rforce= &SoftenedNeedleBarPotentialRforce;
      potentialArgs->zforce= &SoftenedNeedleBarPotentialzforce;
      potentialArgs->phitorque= &SoftenedNeedleBarPotentialphitorque;
      potentialArgs->nargs= 13;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 26: //DiskSCFPotential, nsigma+3 arguments
      potentialArgs->potentialEval= &DiskSCFPotentialEval;
      potentialArgs->Rforce= &DiskSCFPotentialRforce;
      potentialArgs->zforce= &DiskSCFPotentialzforce;
      potentialArgs->dens= &DiskSCFPotentialDens;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->R2deriv= &DiskSCFPotentialR2deriv;
      potentialArgs->z2deriv= &DiskSCFPotentialz2deriv;
      potentialArgs->Rzderiv= &DiskSCFPotentialRzderiv;
      // phi2deriv/Rphideriv/zphideriv are identically zero (axisymmetric) ->
      // left NULL, the 3D Hessian aggregators skip them.
      potentialArgs->nargs= (int) **pot_args + 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 27: // SpiralArmsPotential, 10 arguments + array of Cs
      potentialArgs->Rforce = &SpiralArmsPotentialRforce;
      potentialArgs->zforce = &SpiralArmsPotentialzforce;
      potentialArgs->phitorque = &SpiralArmsPotentialphitorque;
      potentialArgs->R2deriv = &SpiralArmsPotentialR2deriv;
      potentialArgs->z2deriv = &SpiralArmsPotentialz2deriv;
      potentialArgs->phi2deriv = &SpiralArmsPotentialphi2deriv;
      potentialArgs->Rzderiv = &SpiralArmsPotentialRzderiv;
      potentialArgs->Rphideriv = &SpiralArmsPotentialRphideriv;
      potentialArgs->zphideriv = &SpiralArmsPotentialzphideriv;
      potentialArgs->nargs = (int) 10 + **pot_args;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 30: // PerfectEllipsoidPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      potentialArgs->zphideriv = &EllipsoidalPotentialzphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &PerfectEllipsoidPotentialpsi;
      potentialArgs->mdens= &PerfectEllipsoidPotentialmdens;
      potentialArgs->mdensDeriv= &PerfectEllipsoidPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    // 31: KGPotential
    // 32: IsothermalDiskPotential
    case 33: //DehnenCoreSphericalPotential, 2 arguments
      potentialArgs->potentialEval= &DehnenCoreSphericalPotentialEval;
      potentialArgs->Rforce= &DehnenCoreSphericalPotentialRforce;
      potentialArgs->zforce= &DehnenCoreSphericalPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &DehnenCoreSphericalPotentialDens;
      potentialArgs->R2deriv= &DehnenCoreSphericalPotentialR2deriv;
      potentialArgs->z2deriv= &DehnenCoreSphericalPotentialz2deriv;
      potentialArgs->Rzderiv= &DehnenCoreSphericalPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 34: //DehnenSphericalPotential, 3 arguments
      potentialArgs->potentialEval= &DehnenSphericalPotentialEval;
      potentialArgs->Rforce= &DehnenSphericalPotentialRforce;
      potentialArgs->zforce= &DehnenSphericalPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &DehnenSphericalPotentialDens;
      potentialArgs->R2deriv= &DehnenSphericalPotentialR2deriv;
      potentialArgs->z2deriv= &DehnenSphericalPotentialz2deriv;
      potentialArgs->Rzderiv= &DehnenSphericalPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 35: //HomogeneousSpherePotential, 3 arguments
      potentialArgs->potentialEval= &HomogeneousSpherePotentialEval;
      potentialArgs->Rforce= &HomogeneousSpherePotentialRforce;
      potentialArgs->zforce= &HomogeneousSpherePotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &HomogeneousSpherePotentialDens;
      potentialArgs->R2deriv= &HomogeneousSpherePotentialR2deriv;
      potentialArgs->z2deriv= &HomogeneousSpherePotentialz2deriv;
      potentialArgs->Rzderiv= &HomogeneousSpherePotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 36: //interpSphericalPotential, XX arguments
      // Set up 1 spline in potentialArgs
      potentialArgs->nspline1d= 1;
      potentialArgs->spline1d= (gsl_spline **)			\
	malloc ( potentialArgs->nspline1d*sizeof ( gsl_spline *) );
      potentialArgs->acc1d= (gsl_interp_accel **)			\
	malloc ( potentialArgs->nspline1d * sizeof ( gsl_interp_accel * ) );
      // allocate accelerator
      *potentialArgs->acc1d= gsl_interp_accel_alloc();
      // Set up interpolater
      nr= (int) **pot_args;
      *potentialArgs->spline1d= gsl_spline_alloc(gsl_interp_cspline,nr);
      gsl_spline_init(*potentialArgs->spline1d,*pot_args+1,*pot_args+1+nr,nr);
      *pot_args+= 2*nr+1;
      // Bind forces
      potentialArgs->potentialEval= &SphericalPotentialEval;
      potentialArgs->Rforce = &SphericalPotentialRforce;
      potentialArgs->zforce = &SphericalPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &SphericalPotentialDens;
      potentialArgs->R2deriv= &SphericalPotentialR2deriv;
      potentialArgs->z2deriv= &SphericalPotentialz2deriv;
      potentialArgs->Rzderiv= &SphericalPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      // Also assign functions specific to SphericalPotential
      potentialArgs->revaluate= &interpSphericalPotentialrevaluate;
      potentialArgs->rforce= &interpSphericalPotentialrforce;
      potentialArgs->r2deriv= &interpSphericalPotentialr2deriv;
      potentialArgs->rdens= &interpSphericalPotentialrdens;
      potentialArgs->nargs = 6;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 37: // TriaxialGaussianPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      potentialArgs->zphideriv = &EllipsoidalPotentialzphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialGaussianPotentialpsi;
      potentialArgs->mdens= &TriaxialGaussianPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialGaussianPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 38: // PowerTriaxialPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      potentialArgs->zphideriv = &EllipsoidalPotentialzphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &PowerTriaxialPotentialpsi;
      potentialArgs->mdens= &PowerTriaxialPotentialmdens;
      potentialArgs->mdensDeriv= &PowerTriaxialPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 39: //NonInertialFrameForce, 23 arguments (10 caching ones)
      // The time-dependent inputs (a0, x0, v0, Omega, Omegadot) are Python/numba
      // functions called back from C at every step via tfuncs. The cinterp=True
      // variant (case 45) instead precomputes them as GSL splines.
      potentialArgs->RforceVelocity= &NonInertialFrameForceRforce;
      potentialArgs->zforceVelocity= &NonInertialFrameForcezforce;
      potentialArgs->phitorqueVelocity= &NonInertialFrameForcephitorque;
      potentialArgs->nargs= 23;
      potentialArgs->ntfuncs= (int) ( 3 * *(*pot_args + 12) * ( 1 + 2 * *(*pot_args + 11) ) \
                                + ( 6 - 4 * ( *(*pot_args + 13) ) ) * *(*pot_args + 15) );
      potentialArgs->requiresVelocity= true;
      break;
    case 45: //NonInertialFrameForce with cinterp=True (on-the-fly C splines)
      // Same force as case 39, but the time-dependent inputs are evaluated from
      // GSL splines built by initNonInertialFrameForceSplines (see below and
      // _parse_noninertial_frame_force on the Python side) rather than from
      // tfuncs; Omegadot is the spline derivative of Omega. The spline block
      // precedes the 23 case-39 args, plus tmin,tmax (args 23,24) for clamping;
      // hence nargs=25 and ntfuncs=0. The force code branches on spline1d!=NULL.
      potentialArgs->RforceVelocity= &NonInertialFrameForceRforce;
      potentialArgs->zforceVelocity= &NonInertialFrameForcezforce;
      potentialArgs->phitorqueVelocity= &NonInertialFrameForcephitorque;
      potentialArgs->nargs= 25;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= true;
      break;
    case 40: //NullPotential, no arguments (only supported for orbit int)
      potentialArgs->Rforce= &ZeroForce;
      potentialArgs->zforce= &ZeroForce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &ZeroForce;
      potentialArgs->nargs= 0;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 41: //EinastoPotential
      potentialArgs->potentialEval= &SphericalPotentialEval;
      potentialArgs->Rforce = &SphericalPotentialRforce;
      potentialArgs->zforce = &SphericalPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &SphericalPotentialDens;
      potentialArgs->R2deriv= &SphericalPotentialR2deriv;
      potentialArgs->z2deriv= &SphericalPotentialz2deriv;
      potentialArgs->Rzderiv= &SphericalPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      // Also assign functions specific to SphericalPotential
      potentialArgs->revaluate= &EinastoPotentialrevaluate;
      potentialArgs->rforce= &EinastoPotentialrforce;
      potentialArgs->r2deriv= &EinastoPotentialr2deriv;
      potentialArgs->rdens= &EinastoPotentialrdens;
      potentialArgs->nargs = 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 42: //TwoPowerSphericalPotential, 4 arguments
      potentialArgs->potentialEval= &TwoPowerSphericalPotentialEval;
      potentialArgs->Rforce= &TwoPowerSphericalPotentialRforce;
      potentialArgs->zforce= &TwoPowerSphericalPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &TwoPowerSphericalPotentialDens;
      potentialArgs->R2deriv= &TwoPowerSphericalPotentialR2deriv;
      potentialArgs->z2deriv= &TwoPowerSphericalPotentialz2deriv;
      potentialArgs->Rzderiv= &TwoPowerSphericalPotentialRzderiv;
      //spherical: phi2deriv, Rphideriv, zphideriv = 0 (leave NULL)
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 43: //TwoPowerTriaxialPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      potentialArgs->zphideriv = &EllipsoidalPotentialzphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TwoPowerTriaxialPotentialpsi;
      potentialArgs->mdens= &TwoPowerTriaxialPotentialmdens;
      potentialArgs->mdensDeriv= &TwoPowerTriaxialPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 44: //MultipoleExpansionPotential
      potentialArgs->potentialEval= &MultipoleExpansionPotentialEval;
      potentialArgs->Rforce= &MultipoleExpansionPotentialRforce;
      potentialArgs->zforce= &MultipoleExpansionPotentialzforce;
      potentialArgs->phitorque= &MultipoleExpansionPotentialphitorque;
      potentialArgs->dens= &MultipoleExpansionPotentialDens;
      potentialArgs->R2deriv= &MultipoleExpansionPotentialR2deriv;
      potentialArgs->z2deriv= &MultipoleExpansionPotentialz2deriv;
      potentialArgs->Rzderiv= &MultipoleExpansionPotentialRzderiv;
      potentialArgs->phi2deriv= &MultipoleExpansionPotentialphi2deriv;
      potentialArgs->Rphideriv= &MultipoleExpansionPotentialRphideriv;
      potentialArgs->zphideriv= &MultipoleExpansionPotentialzphideriv;
      potentialArgs->nargs= 0; // arguments handled in the initialization code run for this potential
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
//////////////////////////////// WRAPPERS /////////////////////////////////////
    case -1: //DehnenSmoothWrapperPotential
      potentialArgs->potentialEval= &DehnenSmoothWrapperPotentialEval;
      potentialArgs->Rforce= &DehnenSmoothWrapperPotentialRforce;
      potentialArgs->zforce= &DehnenSmoothWrapperPotentialzforce;
      potentialArgs->phitorque= &DehnenSmoothWrapperPotentialphitorque;
      potentialArgs->R2deriv= &DehnenSmoothWrapperPotentialR2deriv;
      potentialArgs->z2deriv= &DehnenSmoothWrapperPotentialz2deriv;
      potentialArgs->Rzderiv= &DehnenSmoothWrapperPotentialRzderiv;
      potentialArgs->phi2deriv= &DehnenSmoothWrapperPotentialphi2deriv;
      potentialArgs->Rphideriv= &DehnenSmoothWrapperPotentialRphideriv;
      potentialArgs->zphideriv= &DehnenSmoothWrapperPotentialzphideriv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -2: //SolidBodyRotationWrapperPotential
      potentialArgs->Rforce= &SolidBodyRotationWrapperPotentialRforce;
      potentialArgs->zforce= &SolidBodyRotationWrapperPotentialzforce;
      potentialArgs->phitorque= &SolidBodyRotationWrapperPotentialphitorque;
      potentialArgs->R2deriv= &SolidBodyRotationWrapperPotentialR2deriv;
      potentialArgs->z2deriv= &SolidBodyRotationWrapperPotentialz2deriv;
      potentialArgs->Rzderiv= &SolidBodyRotationWrapperPotentialRzderiv;
      potentialArgs->phi2deriv= &SolidBodyRotationWrapperPotentialphi2deriv;
      potentialArgs->Rphideriv= &SolidBodyRotationWrapperPotentialRphideriv;
      potentialArgs->zphideriv= &SolidBodyRotationWrapperPotentialzphideriv;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -3: //OblateStaeckelWrapperPotential
      potentialArgs->potentialEval= &OblateStaeckelWrapperPotentialEval;
      potentialArgs->Rforce= &OblateStaeckelWrapperPotentialRforce;
      potentialArgs->zforce= &OblateStaeckelWrapperPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->nargs= (int) 5;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -4: //CorotatingRotationWrapperPotential
      potentialArgs->Rforce= &CorotatingRotationWrapperPotentialRforce;
      potentialArgs->zforce= &CorotatingRotationWrapperPotentialzforce;
      potentialArgs->phitorque= &CorotatingRotationWrapperPotentialphitorque;
      potentialArgs->nargs= 5;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -5: //GaussianAmplitudeWrapperPotential
      potentialArgs->potentialEval= &GaussianAmplitudeWrapperPotentialEval;
      potentialArgs->Rforce= &GaussianAmplitudeWrapperPotentialRforce;
      potentialArgs->zforce= &GaussianAmplitudeWrapperPotentialzforce;
      potentialArgs->phitorque= &GaussianAmplitudeWrapperPotentialphitorque;
      potentialArgs->R2deriv= &GaussianAmplitudeWrapperPotentialR2deriv;
      potentialArgs->z2deriv= &GaussianAmplitudeWrapperPotentialz2deriv;
      potentialArgs->Rzderiv= &GaussianAmplitudeWrapperPotentialRzderiv;
      potentialArgs->phi2deriv= &GaussianAmplitudeWrapperPotentialphi2deriv;
      potentialArgs->Rphideriv= &GaussianAmplitudeWrapperPotentialRphideriv;
      potentialArgs->zphideriv= &GaussianAmplitudeWrapperPotentialzphideriv;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -6: //MovingObjectPotential
      potentialArgs->Rforce= &MovingObjectPotentialRforce;
      potentialArgs->zforce= &MovingObjectPotentialzforce;
      potentialArgs->phitorque= &MovingObjectPotentialphitorque;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -7: //ChandrasekharDynamicalFrictionForce
      potentialArgs->RforceVelocity= &ChandrasekharDynamicalFrictionForceRforce;
      potentialArgs->zforceVelocity= &ChandrasekharDynamicalFrictionForcezforce;
      potentialArgs->phitorqueVelocity= &ChandrasekharDynamicalFrictionForcephitorque;
      // Rectangular dissipative-force Jacobian (dF/dx, dF/dv) for the 3D
      // variational equations (integrate_dxdv with this dissipative force).
      potentialArgs->RectDissipativeForceJacobian= &ChandrasekharDynamicalFrictionForceRectDissipativeForceJacobian;
      potentialArgs->nargs= 16;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= true;
      break;
    case -8: //RotateAndTiltWrapperPotential
      potentialArgs->Rforce= &RotateAndTiltWrapperPotentialRforce;
      potentialArgs->zforce= &RotateAndTiltWrapperPotentialzforce;
      potentialArgs->phitorque= &RotateAndTiltWrapperPotentialphitorque;
      // Full-3D Hessian for the 3D variational equations (integrate_dxdv).
      potentialArgs->R2deriv= &RotateAndTiltWrapperPotentialR2deriv;
      potentialArgs->z2deriv= &RotateAndTiltWrapperPotentialz2deriv;
      potentialArgs->Rzderiv= &RotateAndTiltWrapperPotentialRzderiv;
      potentialArgs->phi2deriv= &RotateAndTiltWrapperPotentialphi2deriv;
      potentialArgs->Rphideriv= &RotateAndTiltWrapperPotentialRphideriv;
      potentialArgs->zphideriv= &RotateAndTiltWrapperPotentialzphideriv;
      potentialArgs->nargs= 31;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -9: //TimeDependentAmplitudeWrapperPotential
      potentialArgs->potentialEval= &TimeDependentAmplitudeWrapperPotentialEval;
      potentialArgs->Rforce= &TimeDependentAmplitudeWrapperPotentialRforce;
      potentialArgs->zforce= &TimeDependentAmplitudeWrapperPotentialzforce;
      potentialArgs->phitorque= &TimeDependentAmplitudeWrapperPotentialphitorque;
      potentialArgs->R2deriv= &TimeDependentAmplitudeWrapperPotentialR2deriv;
      potentialArgs->z2deriv= &TimeDependentAmplitudeWrapperPotentialz2deriv;
      potentialArgs->Rzderiv= &TimeDependentAmplitudeWrapperPotentialRzderiv;
      potentialArgs->phi2deriv= &TimeDependentAmplitudeWrapperPotentialphi2deriv;
      potentialArgs->Rphideriv= &TimeDependentAmplitudeWrapperPotentialRphideriv;
      potentialArgs->zphideriv= &TimeDependentAmplitudeWrapperPotentialzphideriv;
      potentialArgs->nargs= 1;
      potentialArgs->ntfuncs= 1;
      potentialArgs->requiresVelocity= false;
      break;
    case -10: // KuzminLikeWrapperPotential
      potentialArgs->potentialEval= &KuzminLikeWrapperPotentialEval;
      potentialArgs->Rforce= &KuzminLikeWrapperPotentialRforce;
      potentialArgs->zforce= &KuzminLikeWrapperPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -11: //FDMDynamicalFrictionForce
      potentialArgs->RforceVelocity= &FDMDynamicalFrictionForceRforce;
      potentialArgs->zforceVelocity= &FDMDynamicalFrictionForcezforce;
      potentialArgs->phitorqueVelocity= &FDMDynamicalFrictionForcephitorque;
      // Rectangular dissipative-force Jacobian (dF/dx, dF/dv) for the 3D
      // variational equations (integrate_dxdv with this dissipative force).
      potentialArgs->RectDissipativeForceJacobian= &FDMDynamicalFrictionForceRectDissipativeForceJacobian;
      potentialArgs->nargs= 18;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= true;
      break;
    case -12: //CylindricallySeparablePotentialWrapper
      potentialArgs->potentialEval= &CylindricallySeparablePotentialWrapperPotentialEval;
      potentialArgs->Rforce= &CylindricallySeparablePotentialWrapperPotentialRforce;
      potentialArgs->zforce= &CylindricallySeparablePotentialWrapperPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->nargs= (int) 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    }
    int setupMovingObjectSplines = *(*pot_type-1) == -6 ? 1 : 0;
    // Need to set up the same sigma_r spline for both Chandrasekhar and FDM dynamical friction
    int setupChandrasekharDynamicalFrictionSplines = (*(*pot_type-1) == -7 || *(*pot_type-1) == -11) ? 1 : 0;
    int initSCFData = *(*pot_type-1) == 24 ? 1 : 0;
    int initMultipoleExpansionData = *(*pot_type-1) == 44 ? 1 : 0;
    int setupNonInertialFrameForceSplines = *(*pot_type-1) == 45 ? 1 : 0;
    if ( *(*pot_type-1) < 0 ) { // Parse wrapped potential for wrappers
      potentialArgs->nwrapped= (int) *(*pot_args)++;
      potentialArgs->wrappedPotentialArg= \
	(struct potentialArg *) malloc ( potentialArgs->nwrapped	\
					 * sizeof (struct potentialArg) );
      parse_leapFuncArgs_Full(potentialArgs->nwrapped,
			      potentialArgs->wrappedPotentialArg,
			      pot_type,pot_args,pot_tfuncs);
    }
    if (setupMovingObjectSplines)
      initMovingObjectSplines(potentialArgs, pot_args);
    if (setupChandrasekharDynamicalFrictionSplines )
      initChandrasekharDynamicalFrictionSplines(potentialArgs,pot_args);
    if ( setupNonInertialFrameForceSplines )
      initNonInertialFrameForceSplines(potentialArgs,pot_args);
    if ( initMultipoleExpansionData )
      initMultipoleExpansionPotentialArgs(potentialArgs, pot_args);
    // Now load each potential's parameters
    potentialArgs->args= (double *) malloc( potentialArgs->nargs * sizeof(double));
    for (jj=0; jj < potentialArgs->nargs; jj++){
      *(potentialArgs->args)= *(*pot_args)++;
      potentialArgs->args++;
    }
    potentialArgs->args-= potentialArgs->nargs;
    // and load each potential's time functions
    if ( potentialArgs->ntfuncs > 0 ) {
      potentialArgs->tfuncs= (*pot_tfuncs);
      (*pot_tfuncs)+= potentialArgs->ntfuncs;
    }
    // Initialize potential-specific pre-computed data
    if ( initSCFData )
      initSCFPotentialArgs(potentialArgs);
    potentialArgs++;
  }
  potentialArgs-= npot;
}
EXPORT void integrateFullOrbit(int nobj,
			       double *yo,
			       int nt,
			       double *t,
			       int indiv_t,
			       int npot,
			       int * pot_type,
			       double * pot_args,
             tfuncs_type_arr pot_tfuncs,
			       double dt,
			       double rtol,
			       double atol,
			       double *result,
			       int * err,
			       int odeint_type,
             orbint_callback_type cb){
  //Set up the forces, first count
  int ii,jj;
  int dim;
  int max_threads;
  int * thread_pot_type;
  double * thread_pot_args;
  tfuncs_type_arr thread_pot_tfuncs;
  max_threads= ( nobj < omp_get_max_threads() ) ? nobj : omp_get_max_threads();
  // Because potentialArgs may cache, safest to have one / thread
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( max_threads * npot * sizeof (struct potentialArg) );
#pragma omp parallel for schedule(static,1) private(ii,thread_pot_type,thread_pot_args,thread_pot_tfuncs) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++) {
    thread_pot_type= pot_type; // need to make thread-private pointers, bc
    thread_pot_args= pot_args; // these pointers are changed in parse_...
    thread_pot_tfuncs= pot_tfuncs; // ...
    parse_leapFuncArgs_Full(npot,potentialArgs+ii*npot,
			    &thread_pot_type,&thread_pot_args,&thread_pot_tfuncs);
  }
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 6: //DOP853
    odeint_func= &dop853;
    odeint_deriv_func= &evalRectDeriv;
    dim= 6;
    break;
  case 7: //ias15
    odeint_func= &wez_ias15;
    odeint_deriv_func= &evalRectForce;
    dim= 3;
    break;
  }
#pragma omp parallel for schedule(dynamic,ORBITS_CHUNKSIZE) private(ii,jj) num_threads(max_threads)
  for (ii=0; ii < nobj; ii++) {
    cyl_to_rect_galpy(yo+6*ii);
    odeint_func(odeint_deriv_func,dim,yo+6*ii,nt,dt,t+nt*ii*indiv_t,
		npot,potentialArgs+omp_get_thread_num()*npot,rtol,atol,
		result+6*nt*ii,err+ii);
    for (jj=0; jj < nt; jj++)
      rect_to_cyl_galpy(result+6*jj+6*nt*ii);
    if ( cb ) // Callback if not void
      cb();
  }
  //Free allocated memory
#pragma omp parallel for schedule(static,1) private(ii) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++)
    free_potentialArgs(npot,potentialArgs+ii*npot);
  free(potentialArgs);
  //Done!
}
EXPORT void integrateFullOrbit_sos(
    int nobj,
	double *yo,
	int npsi,
	double *psi,
    int indiv_psi,
    int npot,
	int * pot_type,
	double * pot_args,
    tfuncs_type_arr pot_tfuncs,
	double dpsi,
	double rtol,
	double atol,
	double *result,
	int * err,
	int odeint_type,
    orbint_callback_type cb){
  //Set up the forces, first count
  int ii,jj;
  int dim;
  int max_threads;
  int * thread_pot_type;
  double * thread_pot_args;
  tfuncs_type_arr thread_pot_tfuncs;
  max_threads= ( nobj < omp_get_max_threads() ) ? nobj : omp_get_max_threads();
  // Because potentialArgs may cache, safest to have one / thread
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( max_threads * npot * sizeof (struct potentialArg) );
#pragma omp parallel for schedule(static,1) private(ii,thread_pot_type,thread_pot_args,thread_pot_tfuncs) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++) {
    thread_pot_type= pot_type; // need to make thread-private pointers, bc
    thread_pot_args= pot_args; // these pointers are changed in parse_...
    thread_pot_tfuncs= pot_tfuncs; // ...
    parse_leapFuncArgs_Full(npot,potentialArgs+ii*npot,
			    &thread_pot_type,&thread_pot_args,&thread_pot_tfuncs);
  }
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  dim= 7;
  odeint_deriv_func= &evalSOSDeriv;
  switch ( odeint_type ) {
  // case 0: = leapfrog = not supported symplectic method
  case 1: //RK4
    odeint_func= &bovy_rk4;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    break;
  // case 3: = symplec4 = not supported symplectic method
  // case 4: = symplec6 = not supported symplectic method
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    break;
  case 6: //DOP853
    odeint_func= &dop853;
    break;
  }
#pragma omp parallel for schedule(dynamic,ORBITS_CHUNKSIZE) private(ii,jj) num_threads(max_threads)
  for (ii=0; ii < nobj; ii++) {
    cyl_to_sos_galpy(yo+dim*ii);
    odeint_func(odeint_deriv_func,dim,yo+dim*ii,npsi,dpsi,psi+npsi*ii*indiv_psi,
		npot,potentialArgs+omp_get_thread_num()*npot,rtol,atol,
		result+dim*npsi*ii,err+ii);
    for (jj=0; jj < npsi; jj++)
      sos_to_cyl_galpy(result+dim*jj+dim*npsi*ii);
    if ( cb ) // Callback if not void
      cb();
  }
  //Free allocated memory
#pragma omp parallel for schedule(static,1) private(ii) num_threads(max_threads)
  for (ii=0; ii < max_threads; ii++)
    free_potentialArgs(npot,potentialArgs+ii*npot);
  free(potentialArgs);
  //Done!
}
EXPORT void integrateFullOrbit_dxdv(double *yo,
			 int nt,
			 double *t,
			 int npot,
			 int * pot_type,
			 double * pot_args,
       tfuncs_type_arr pot_tfuncs,
			 double dt,
			 double rtol,
			 double atol,
			 double *result,
			 int * err,
			 int odeint_type){
  //Set up the forces, first count
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs_Full(npot,potentialArgs,&pot_type,&pot_args,&pot_tfuncs);
  //Integrate
  void (*odeint_func)(void (*func)(double, double *, double *,
			   int, struct potentialArg *),
		      int,
		      double *,
		      int, double, double *,
		      int, struct potentialArg *,
		      double, double,
		      double *,int *);
  void (*odeint_deriv_func)(double, double *, double *,
			    int,struct potentialArg *);
  // Only the non-symplectic integrators support the 12D variational (dxdv)
  // system; Orbit.integrate_dxdv enforces this upstream
  // (check_integrator(no_symplec=True)), so the symplectic/leapfrog/ias15
  // odeint_types never reach here -- mirroring integratePlanarOrbit_dxdv.
  switch ( odeint_type ) {
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  case 6: //DOP853
    odeint_func= &dop853;
    odeint_deriv_func= &evalRectDeriv_dxdv;
    dim= 12;
    break;
  }
  odeint_func(odeint_deriv_func,dim,yo,nt,dt,t,npot,potentialArgs,
	      rtol,atol,result,err);
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}
void evalRectForce(double t, double *q, double *a,
		   int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phitorque, z;
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs);
  phitorque= calcphitorque(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a++= sinphi*Rforce+1./R*cosphi*phitorque;
  *a= calczforce(R,z,phi,t,nargs,potentialArgs);
}
void evalRectDeriv(double t, double *q, double *a,
		   int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phitorque,z,vR,vT;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //Rest is force
  //q is rectangular so calculate R and phi, vR and vT (for dissipative)
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  vR=  *(q+3) * cosphi + *(q+4) * sinphi;
  vT= -*(q+3) * sinphi + *(q+4) * cosphi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs,1,vR,vT,*(q+5));
  phitorque= calcphitorque(R,z,phi,t,nargs,potentialArgs,1,vR,vT,*(q+5));
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a++= sinphi*Rforce+1./R*cosphi*phitorque;
  *a= calczforce(R,z,phi,t,nargs,potentialArgs,1,vR,vT,*(q+5));;
}

void evalSOSDeriv(double psi, double *q, double *a,
		              int nargs, struct potentialArg * potentialArgs){
  // q= (x,y,vx,vy,A,t,psi); to save operations, we reuse a first for the
  // rectForce then for the actual RHS
  // Note also that we keep track of psi in q+6, not in psi! This is
  // such that we can avoid having to convert psi to psi+psi0
  // q+6 starts as psi0 and then just increments as psi (exactly)
  double sinpsi,cospsi,psidot,x,y,z,R,phi,sinphi,cosphi,vR,vT,vz,Rforce,phitorque;
  sinpsi= sin( *(q+6) );
  cospsi= cos( *(q+6) );
  // Calculate forces, put them in a+3, a+4, a+5
  //q is rectangular so calculate R and phi, vR and vT (for dissipative)
  x= *q;
  y= *(q+1);
  z= *(q+4) * sinpsi;
  R= sqrt(x*x+y*y);
  phi= atan2( y ,x );
  sinphi= y/R;
  cosphi= x/R;
  vR=  *(q+2) * cosphi + *(q+3) * sinphi;
  vT= -*(q+2) * sinphi + *(q+3) * cosphi;
  vz= *(q+4) * cospsi;
  //Calculate the forces
  Rforce= calcRforce(R,z,phi,*(q+5),nargs,potentialArgs,vR,vT,vz);
  phitorque= calcphitorque(R,z,phi,*(q+5),nargs,potentialArgs,vR,vT,vz);
  *(a+3)= cosphi*Rforce-1./R*sinphi*phitorque;
  *(a+4)= sinphi*Rforce+1./R*cosphi*phitorque;
  *(a+5)= calczforce(R,z,phi,*(q+5),nargs,potentialArgs,vR,vT,vz);
  // Now calculate the RHS of the ODE
  psidot= cospsi * cospsi - sinpsi * *(a+5) / ( *(q+4) );
  *(a  )= *(q+2) / psidot;
  *(a+1)= *(q+3) / psidot;
  *(a+2)= *(a+3) / psidot;
  *(a+3)= *(a+4) / psidot;
  *(a+4)= cospsi * ( *(q+4) * sinpsi + *(a+5) ) / psidot;
  *(a+5)= 1./psidot;
  *(a+6)= 1.; // dpsi / dpsi to keep track of psi
}

void initMovingObjectSplines(struct potentialArg * potentialArgs,
			     double ** pot_args){
  gsl_interp_accel *x_accel_ptr = gsl_interp_accel_alloc();
  gsl_interp_accel *y_accel_ptr = gsl_interp_accel_alloc();
  gsl_interp_accel *z_accel_ptr = gsl_interp_accel_alloc();
  int nPts = (int) **pot_args;

  gsl_spline *x_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);
  gsl_spline *y_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);
  gsl_spline *z_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);

  double * t_arr = *pot_args+1;
  double * x_arr = t_arr+1*nPts;
  double * y_arr = t_arr+2*nPts;
  double * z_arr = t_arr+3*nPts;

  double * t= (double *) malloc ( nPts * sizeof (double) );
  double tf = *(t_arr+4*nPts+2);
  double to = *(t_arr+4*nPts+1);

  int ii;
  for (ii=0; ii < nPts; ii++)
    *(t+ii) = (t_arr[ii]-to)/(tf-to);

  gsl_spline_init(x_spline, t, x_arr, nPts);
  gsl_spline_init(y_spline, t, y_arr, nPts);
  gsl_spline_init(z_spline, t, z_arr, nPts);

  potentialArgs->nspline1d= 3;
  potentialArgs->spline1d= (gsl_spline **) malloc ( 3*sizeof ( gsl_spline *) );
  potentialArgs->acc1d= (gsl_interp_accel **) \
    malloc ( 3 * sizeof ( gsl_interp_accel * ) );
  *potentialArgs->spline1d = x_spline;
  *potentialArgs->acc1d = x_accel_ptr;
  *(potentialArgs->spline1d+1)= y_spline;
  *(potentialArgs->acc1d+1)= y_accel_ptr;
  *(potentialArgs->spline1d+2)= z_spline;
  *(potentialArgs->acc1d+2)= z_accel_ptr;

  *pot_args = *pot_args + (int) (1+4*nPts);
  free(t);
}

void initNonInertialFrameForceSplines(struct potentialArg * potentialArgs,
				      double ** pot_args){
  // cinterp NonInertialFrameForce (pot_type 45). The spline block at the front
  // of pot_args is: n_spline, nPts, tgrid[nPts], then n_spline value arrays of
  // length nPts in the order a0(0-2)[, x0(3-5), v0(6-8)][, Omega(9*lin_acc...)]
  // -- matching the tfunc indices used by NonInertialFrameForce.c. (x0/v0 are
  // present only when lin_acc and rot_acc; Omega is present and read only when
  // rot_acc, so its base index 9*lin_acc is 9 when lin_acc -- after a0/x0/v0 --
  // and 0 otherwise.) Splines use the raw (un-normalized) time grid, so
  // gsl_spline_eval_deriv directly gives d/dt (used to obtain Omegadot from the
  // Omega spline).
  int n_spline = (int) **pot_args;
  int nPts = (int) *(*pot_args + 1);
  double * t_arr = *pot_args + 2; // tgrid; value array ii starts at t_arr+(ii+1)*nPts
  int ii;
  potentialArgs->nspline1d= n_spline;
  potentialArgs->spline1d= (gsl_spline **)				\
    malloc ( n_spline * sizeof ( gsl_spline * ) );
  potentialArgs->acc1d= (gsl_interp_accel **)				\
    malloc ( n_spline * sizeof ( gsl_interp_accel * ) );
  for (ii=0; ii < n_spline; ii++) {
    *(potentialArgs->acc1d + ii)= gsl_interp_accel_alloc();
    *(potentialArgs->spline1d + ii)= gsl_spline_alloc(gsl_interp_cspline,nPts);
    gsl_spline_init(*(potentialArgs->spline1d + ii),
		    t_arr,t_arr + (ii + 1) * nPts,nPts);
  }
  *pot_args = *pot_args + (int) ( 2 + ( 1 + n_spline ) * nPts );
}

void initChandrasekharDynamicalFrictionSplines(struct potentialArg * potentialArgs,
					       double ** pot_args){
  gsl_interp_accel *sr_accel_ptr = gsl_interp_accel_alloc();
  int nPts = (int) **pot_args;

  gsl_spline *sr_spline = gsl_spline_alloc(gsl_interp_cspline,nPts);

  double * r_arr = *pot_args+1;
  double * sr_arr = r_arr+1*nPts;

  double * r= (double *) malloc ( nPts * sizeof (double) );
  double ro = *(r_arr+2*nPts+14);
  double rf = *(r_arr+2*nPts+15);

  int ii;
  for (ii=0; ii < nPts; ii++)
    *(r+ii) = (r_arr[ii]-ro)/(rf-ro);

  gsl_spline_init(sr_spline,r,sr_arr,nPts);

  potentialArgs->nspline1d= 1;
  potentialArgs->spline1d= (gsl_spline **) \
    malloc ( potentialArgs->nspline1d*sizeof ( gsl_spline *) );
  potentialArgs->acc1d= (gsl_interp_accel **) \
    malloc ( potentialArgs->nspline1d * sizeof ( gsl_interp_accel * ) );
  *potentialArgs->spline1d = sr_spline;
  *potentialArgs->acc1d = sr_accel_ptr;

  *pot_args = *pot_args + (int) (1+(1+potentialArgs->nspline1d)*nPts);
  free(r);
}

void evalRectDeriv_dxdv(double t, double *q, double *a,
			int nargs, struct potentialArg * potentialArgs){
  // 12D state q = (x,y,z,vx,vy,vz | dx,dy,dz,dvx,dvy,dvz): the orbit plus one
  // phase-space deviation propagated by the variational equation dw'=A w'
  // with the general Jacobian A=[[0,I],[K + dF/dx, dF/dv]]: K is the
  // symmetric Cartesian tidal tensor (-grad grad Phi) of the conservative
  // components; (dF/dx, dF/dv) are the rectangular Jacobian blocks of the
  // velocity-dependent (dissipative) components -- the dissipative dF/dx is
  // NOT symmetric and the velocity block dF/dv is nonzero -- aggregated by
  // the NULL-safe calcRectDissipativeForceJacobian, which returns exact
  // zeros when no component has the Jacobian, reducing the system to the
  // conservative A=[[0,I],[K,0]].
  double sinphi, cosphi, x, y, phi, R, Rforce, phitorque, z, zforce;
  double vR, vT, RforceK, phitorqueK;
  double R2deriv, phi2deriv, Rphideriv, z2deriv, Rzderiv, zphideriv;
  double dFxdx, dFxdy, dFydy, dFxdz, dFydz, dFzdz, dx, dy, dz;
  double jac_x[9], jac_v[9], dvx, dvy, dvz;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //q is rectangular so calculate R and phi, vR and vT (for dissipative)
  x= *q;
  y= *(q+1);
  z= *(q+2);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  vR=  *(q+3) * cosphi + *(q+4) * sinphi;
  vT= -*(q+3) * sinphi + *(q+4) * cosphi;
  //Calculate the forces -> Cartesian accelerations (passing the velocity is
  //a no-op for conservative potentials: requiresVelocity routes to the same
  //velocity-free force functions, so this is bit-identical to before)
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs,1,vR,vT,*(q+5));
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs,1,vR,vT,*(q+5));
  phitorque= calcphitorque(R,z,phi,t,nargs,potentialArgs,1,vR,vT,*(q+5));
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a++= sinphi*Rforce+1./R*cosphi*phitorque;
  *a++= zforce;
  //d(deviation position)/dt = deviation velocity
  *a++= *(q+9);
  *a++= *(q+10);
  *a++= *(q+11);
  // d(deviation velocity)/dt = (K + dF/dx) . (dx,dy,dz) + dF/dv . (dvx,dvy,dvz)
  // K needs the full 3D Hessian (conservative components only: dissipative
  // forces have NULL second-derivative pointers, so the NULL-safe aggregators
  // skip them and their position-Jacobian enters via jac_x instead)
  R2deriv= calcR2deriv(R,z,phi,t,nargs,potentialArgs);
  phi2deriv= calcphi2deriv(R,z,phi,t,nargs,potentialArgs);
  Rphideriv= calcRphideriv(R,z,phi,t,nargs,potentialArgs);
  z2deriv= calcz2deriv(R,z,phi,t,nargs,potentialArgs);
  Rzderiv= calcRzderiv(R,z,phi,t,nargs,potentialArgs);
  zphideriv= calczphideriv(R,z,phi,t,nargs,potentialArgs);
  // The Rforce/phitorque entering the cylindrical->Cartesian conversion of
  // the CONSERVATIVE Hessian K below must be the conservative force only:
  // a dissipative component's dF/dx is already complete in rectangular
  // coordinates (jac_x below), so its force must not enter the curvilinear
  // conversion terms (it would double-count terms that do not apply to it).
  // For purely conservative potentials these aggregators sum exactly the
  // same components in the same order as calcRforce/calcphitorque above, so
  // RforceK/phitorqueK equal Rforce/phitorque bit-for-bit there.
  // conservative force only (include_dissipative=0): the dissipative forces'
  // position-Jacobian is rectangular (calcRectDissipativeForceJacobian), so
  // they must not enter these curvilinear Hessian-conversion terms
  RforceK= calcRforce(R,z,phi,t,nargs,potentialArgs,0,0.,0.,0.);
  phitorqueK= calcphitorque(R,z,phi,t,nargs,potentialArgs,0,0.,0.,0.);
  // In-plane (x,y) block: identical to the verified 2D variational equations
  // (z enters only through the values of the second derivatives above).
  dFxdx= -cosphi*cosphi*R2deriv
    +2.*cosphi*sinphi/R/R*phitorqueK
    +sinphi*sinphi/R*RforceK
    +2.*sinphi*cosphi/R*Rphideriv
    -sinphi*sinphi/R/R*phi2deriv;
  dFxdy= -sinphi*cosphi*R2deriv
    +(sinphi*sinphi-cosphi*cosphi)/R/R*phitorqueK
    -cosphi*sinphi/R*RforceK
    -(cosphi*cosphi-sinphi*sinphi)/R*Rphideriv
    +cosphi*sinphi/R/R*phi2deriv;
  dFydy= -sinphi*sinphi*R2deriv
    -2.*sinphi*cosphi/R/R*phitorqueK
    -2.*sinphi*cosphi/R*Rphideriv
    +cosphi*cosphi/R*RforceK
    -cosphi*cosphi/R/R*phi2deriv;
  // z-coupling (K symmetric: dFzdx=dFxdz, dFzdy=dFydz, dFydx=dFxdy)
  dFxdz= -cosphi*Rzderiv+sinphi/R*zphideriv;
  dFydz= -sinphi*Rzderiv-cosphi/R*zphideriv;
  dFzdz= -z2deriv;
  dx= *(q+6);
  dy= *(q+7);
  dz= *(q+8);
  dvx= *(q+9);
  dvy= *(q+10);
  dvz= *(q+11);
  // Dissipative rectangular Jacobian blocks: exact zeros when no component
  // provides RectDissipativeForceJacobian (purely conservative case)
  calcRectDissipativeForceJacobian(t,q,jac_x,jac_v,nargs,potentialArgs);
  *a++= dFxdx*dx+dFxdy*dy+dFxdz*dz
    + jac_x[0]*dx + jac_x[1]*dy + jac_x[2]*dz
    + jac_v[0]*dvx + jac_v[1]*dvy + jac_v[2]*dvz;
  *a++= dFxdy*dx+dFydy*dy+dFydz*dz
    + jac_x[3]*dx + jac_x[4]*dy + jac_x[5]*dz
    + jac_v[3]*dvx + jac_v[4]*dvy + jac_v[5]*dvz;
  *a  = dFxdz*dx+dFydz*dy+dFzdz*dz
    + jac_x[6]*dx + jac_x[7]*dy + jac_x[8]*dz
    + jac_v[6]*dvx + jac_v[7]*dvy + jac_v[8]*dvz;
}
