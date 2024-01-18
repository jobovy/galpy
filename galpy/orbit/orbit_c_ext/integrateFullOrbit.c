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
      //potentialArgs->R2deriv= &LogarithmicHaloPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 1: //DehnenBarPotential, 6 arguments
      potentialArgs->Rforce= &DehnenBarPotentialRforce;
      potentialArgs->phitorque= &DehnenBarPotentialphitorque;
      potentialArgs->zforce= &DehnenBarPotentialzforce;
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
      //potentialArgs->R2deriv= &MiyamotoNagaiPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
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
      //potentialArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
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
      //potentialArgs->R2deriv= &HernquistPotentialR2deriv;
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
      //potentialArgs->R2deriv= &NFWPotentialR2deriv;
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
      //potentialArgs->R2deriv= &JaffePotentialR2deriv;
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
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 15: //PowerSphericalwCutoffPotential, 3 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialwCutoffEval;
      potentialArgs->Rforce= &PowerSphericalPotentialwCutoffRforce;
      potentialArgs->zforce= &PowerSphericalPotentialwCutoffzforce;
      potentialArgs->phitorque= &ZeroForce;
      potentialArgs->dens= &PowerSphericalPotentialwCutoffDens;
      //potentialArgs->R2deriv= &PowerSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 16: //KuzminKutuzovStaeckelPotential, 3 arguments
      potentialArgs->potentialEval= &KuzminKutuzovStaeckelPotentialEval;
      potentialArgs->Rforce= &KuzminKutuzovStaeckelPotentialRforce;
      potentialArgs->zforce= &KuzminKutuzovStaeckelPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
      //potentialArgs->R2deriv= &KuzminKutuzovStaeckelPotentialR2deriv;
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
      //potentialArgs->R2deriv= &PlummerPotentialR2deriv;
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
      //potentialArgs->R2deriv= &PseudoIsothermalPotentialR2deriv;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 19: //KuzminDiskPotential, 2 arguments
      potentialArgs->potentialEval= &KuzminDiskPotentialEval;
      potentialArgs->Rforce= &KuzminDiskPotentialRforce;
      potentialArgs->zforce= &KuzminDiskPotentialzforce;
      potentialArgs->phitorque= &ZeroForce;
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
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialHernquistPotentialpsi;
      potentialArgs->mdens= &TriaxialHernquistPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialHernquistPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args
					    + (int) (*(*pot_args+7) + 20)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 22: //TriaxialNFWPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialNFWPotentialpsi;
      potentialArgs->mdens= &TriaxialNFWPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialNFWPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args
					    + (int) (*(*pot_args+7) + 20)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 23: //TriaxialJaffePotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialJaffePotentialpsi;
      potentialArgs->mdens= &TriaxialJaffePotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialJaffePotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args
					    + (int) (*(*pot_args+7) + 20)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 24: //SCFPotential, many arguments
      potentialArgs->potentialEval= &SCFPotentialEval;
      potentialArgs->Rforce= &SCFPotentialRforce;
      potentialArgs->zforce= &SCFPotentialzforce;
      potentialArgs->phitorque= &SCFPotentialphitorque;
      potentialArgs->dens= &SCFPotentialDens;
      potentialArgs->nargs= (int) (5 + (1 + *(*pot_args + 1)) * *(*pot_args+2) * *(*pot_args+3)* *(*pot_args+4) + 7);
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
      potentialArgs->nargs= (int) **pot_args + 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 27: // SpiralArmsPotential, 10 arguments + array of Cs
      potentialArgs->Rforce = &SpiralArmsPotentialRforce;
      potentialArgs->zforce = &SpiralArmsPotentialzforce;
      potentialArgs->phitorque = &SpiralArmsPotentialphitorque;
      //potentialArgs->R2deriv = &SpiralArmsPotentialR2deriv;
      //potentialArgs->z2deriv = &SpiralArmsPotentialz2deriv;
      potentialArgs->phi2deriv = &SpiralArmsPotentialphi2deriv;
      //potentialArgs->Rzderiv = &SpiralArmsPotentialRzderiv;
      potentialArgs->Rphideriv = &SpiralArmsPotentialRphideriv;
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
      //potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      //potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      //potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      //potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      //potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &PerfectEllipsoidPotentialpsi;
      potentialArgs->mdens= &PerfectEllipsoidPotentialmdens;
      potentialArgs->mdensDeriv= &PerfectEllipsoidPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args
					    + (int) (*(*pot_args+7) + 20)));
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
      //potentialArgs->R2deriv= &DehnenCoreSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
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
      //potentialArgs->R2deriv= &DehnenSphericalPotentialR2deriv;
      //potentialArgs->planarphi2deriv= &ZeroForce;
      //potentialArgs->planarRphideriv= &ZeroForce;
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
      //potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      //potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      //potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      //potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      //potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialGaussianPotentialpsi;
      potentialArgs->mdens= &TriaxialGaussianPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialGaussianPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args
					    + (int) (*(*pot_args+7) + 20)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 38: // PowerTriaxialPotential, lots of arguments
      potentialArgs->potentialEval= &EllipsoidalPotentialEval;
      potentialArgs->Rforce = &EllipsoidalPotentialRforce;
      potentialArgs->zforce = &EllipsoidalPotentialzforce;
      potentialArgs->phitorque = &EllipsoidalPotentialphitorque;
      potentialArgs->dens= &EllipsoidalPotentialDens;
      //potentialArgs->R2deriv = &EllipsoidalPotentialR2deriv;
      //potentialArgs->z2deriv = &EllipsoidalPotentialz2deriv;
      //potentialArgs->phi2deriv = &EllipsoidalPotentialphi2deriv;
      //potentialArgs->Rzderiv = &EllipsoidalPotentialRzderiv;
      //potentialArgs->Rphideriv = &EllipsoidalPotentialRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &PowerTriaxialPotentialpsi;
      potentialArgs->mdens= &PowerTriaxialPotentialmdens;
      potentialArgs->mdensDeriv= &PowerTriaxialPotentialmdensDeriv;
      potentialArgs->nargs = (int) (21 + *(*pot_args+7) + 2 * *(*pot_args
					    + (int) (*(*pot_args+7) + 20)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 39: //NonInertialFrameForce, 22 arguments (10 caching ones)
      potentialArgs->RforceVelocity= &NonInertialFrameForceRforce;
      potentialArgs->zforceVelocity= &NonInertialFrameForcezforce;
      potentialArgs->phitorqueVelocity= &NonInertialFrameForcephitorque;
      potentialArgs->nargs= 23;
      potentialArgs->ntfuncs= (int) ( 3 * *(*pot_args + 12) * ( 1 + 2 * *(*pot_args + 11) ) \
                                + ( 6 - 4 * ( *(*pot_args + 13) ) ) * *(*pot_args + 15) );
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
//////////////////////////////// WRAPPERS /////////////////////////////////////
    case -1: //DehnenSmoothWrapperPotential
      potentialArgs->potentialEval= &DehnenSmoothWrapperPotentialEval;
      potentialArgs->Rforce= &DehnenSmoothWrapperPotentialRforce;
      potentialArgs->zforce= &DehnenSmoothWrapperPotentialzforce;
      potentialArgs->phitorque= &DehnenSmoothWrapperPotentialphitorque;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -2: //SolidBodyRotationWrapperPotential
      potentialArgs->Rforce= &SolidBodyRotationWrapperPotentialRforce;
      potentialArgs->zforce= &SolidBodyRotationWrapperPotentialzforce;
      potentialArgs->phitorque= &SolidBodyRotationWrapperPotentialphitorque;
      potentialArgs->nargs= 3;
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
      potentialArgs->nargs= 16;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= true;
      break;
    case -8: //RotateAndTiltWrapperPotential
      potentialArgs->Rforce= &RotateAndTiltWrapperPotentialRforce;
      potentialArgs->zforce= &RotateAndTiltWrapperPotentialzforce;
      potentialArgs->phitorque= &RotateAndTiltWrapperPotentialphitorque;
      potentialArgs->nargs= 21;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -9: //TimeDependentAmplitudeWrapperPotential
      potentialArgs->potentialEval= &TimeDependentAmplitudeWrapperPotentialEval;
      potentialArgs->Rforce= &TimeDependentAmplitudeWrapperPotentialRforce;
      potentialArgs->zforce= &TimeDependentAmplitudeWrapperPotentialzforce;
      potentialArgs->phitorque= &TimeDependentAmplitudeWrapperPotentialphitorque;
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
    }
    int setupMovingObjectSplines = *(*pot_type-1) == -6 ? 1 : 0;
    int setupChandrasekharDynamicalFrictionSplines = *(*pot_type-1) == -7 ? 1 : 0;
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
    if (setupChandrasekharDynamicalFrictionSplines)
      initChandrasekharDynamicalFrictionSplines(potentialArgs,pot_args);
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
    potentialArgs++;
  }
  potentialArgs-= npot;
}
EXPORT void integrateFullOrbit(int nobj,
			       double *yo,
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
  }
#pragma omp parallel for schedule(dynamic,ORBITS_CHUNKSIZE) private(ii,jj) num_threads(max_threads)
  for (ii=0; ii < nobj; ii++) {
    cyl_to_rect_galpy(yo+6*ii);
    odeint_func(odeint_deriv_func,dim,yo+6*ii,nt,dt,t,
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
// LCOV_EXCL_START
void integrateOrbit_dxdv(double *yo,
			 int nt,
			 double *t,
			 int npot,
			 int * pot_type,
			 double * pot_args,
       tfuncs_type_arr pot_tfuncs,
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
  switch ( odeint_type ) {
  case 0: //leapfrog
    odeint_func= &leapfrog;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
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
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalRectForce;
    dim= 6;
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
  odeint_func(odeint_deriv_func,dim,yo,nt,-9999.99,t,npot,potentialArgs,
	      rtol,atol,result,err);
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}
// LCOV_EXCL_STOP
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
  Rforce= calcRforce(R,z,phi,t,nargs,potentialArgs,vR,vT,*(q+5));
  phitorque= calcphitorque(R,z,phi,t,nargs,potentialArgs,vR,vT,*(q+5));
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a++= sinphi*Rforce+1./R*cosphi*phitorque;
  *a= calczforce(R,z,phi,t,nargs,potentialArgs,vR,vT,*(q+5));;
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

// LCOV_EXCL_START
void evalRectDeriv_dxdv(double t, double *q, double *a,
			int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phitorque,z,zforce;
  double R2deriv, phi2deriv, Rphideriv, dFxdx, dFxdy, dFydx, dFydy;
  //first three derivatives are just the velocities
  *a++= *(q+3);
  *a++= *(q+4);
  *a++= *(q+5);
  //Rest is force
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
  zforce= calczforce(R,z,phi,t,nargs,potentialArgs);
  phitorque= calcphitorque(R,z,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a++= sinphi*Rforce+1./R*cosphi*phitorque;
  *a++= zforce;
  //dx derivatives are just dv
  *a++= *(q+9);
  *a++= *(q+10);
  *a++= *(q+11);
  //for the dv derivatives we need also R2deriv, phi2deriv, and Rphideriv
  R2deriv= calcR2deriv(R,z,phi,t,nargs,potentialArgs);
  phi2deriv= calcphi2deriv(R,z,phi,t,nargs,potentialArgs);
  Rphideriv= calcRphideriv(R,z,phi,t,nargs,potentialArgs);
  //..and dFxdx, dFxdy, dFydx, dFydy
  dFxdx= -cosphi*cosphi*R2deriv
    +2.*cosphi*sinphi/R/R*phitorque
    +sinphi*sinphi/R*Rforce
    +2.*sinphi*cosphi/R*Rphideriv
    -sinphi*sinphi/R/R*phi2deriv;
  dFxdy= -sinphi*cosphi*R2deriv
    +(sinphi*sinphi-cosphi*cosphi)/R/R*phitorque
    -cosphi*sinphi/R*Rforce
    -(cosphi*cosphi-sinphi*sinphi)/R*Rphideriv
    +cosphi*sinphi/R/R*phi2deriv;
  dFydx= -cosphi*sinphi*R2deriv
    +(sinphi*sinphi-cosphi*cosphi)/R/R*phitorque
    +(sinphi*sinphi-cosphi*cosphi)/R*Rphideriv
    -sinphi*cosphi/R*Rforce
    +sinphi*cosphi/R/R*phi2deriv;
  dFydy= -sinphi*sinphi*R2deriv
    -2.*sinphi*cosphi/R/R*phitorque
    -2.*sinphi*cosphi/R*Rphideriv
    +cosphi*cosphi/R*Rforce
    -cosphi*cosphi/R/R*phi2deriv;
  *a++= dFxdx * *(q+4) + dFxdy * *(q+5);
  *a++= dFydx * *(q+4) + dFydy * *(q+5);
  *a= 0; //BOVY: PUT IN Z2DERIVS
}

// LCOV_EXCL_STOP
