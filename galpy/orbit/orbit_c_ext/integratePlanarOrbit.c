/*
  Wrappers around the C integration code for planar Orbits
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <bovy_coords.h>
#include <bovy_symplecticode.h>
#include <bovy_rk.h>
#include <wez_ias15.h>
#include <leung_dop853.h>
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
/*
  Function Declarations
*/
void evalPlanarRectForce(double, double *, double *,
			 int, struct potentialArg *);
void evalPlanarRectDeriv(double, double *, double *,
			 int, struct potentialArg *);
void evalPlanarSOSDerivx(double, double *, double *,
			 int, struct potentialArg *);
void evalPlanarSOSDerivy(double, double *, double *,
			 int, struct potentialArg *);
void evalPlanarRectDeriv_dxdv(double, double *, double *,
			      int, struct potentialArg *);
// Augmented force+Hessian evaluator and planar symplectic variational (dxdv)
// steppers: carry nde phase-space deviation columns via the closed-form
// drift/kick tangent maps (see integratePlanarOrbit_dxdv). Planar (2D) analogs
// of evalRectForce_dxdv/leapfrog_dxdv/... in integrateFullOrbit.c.
void evalPlanarRectForce_dxdv(double, double *, double *,
			      int, struct potentialArg *, int);
void leapfrog_dxdv_planar(int, double *, int, double, double *,
			  int, struct potentialArg *, double, double,
			  double *, int *);
void symplec4_dxdv_planar(int, double *, int, double, double *,
			  int, struct potentialArg *, double, double,
			  double *, int *);
void symplec6_dxdv_planar(int, double *, int, double, double *,
			  int, struct potentialArg *, double, double,
			  double *, int *);
void initPlanarMovingObjectSplines(struct potentialArg *, double ** pot_args);
/*
  Actual functions
*/
void parse_leapFuncArgs(int npot,struct potentialArg * potentialArgs,
			int ** pot_type,
			double ** pot_args,
      tfuncs_type_arr * pot_tfuncs){
  int ii,jj;
  int nr;
  init_potentialArgs(npot,potentialArgs);
  for (ii=0; ii < npot; ii++){
    switch ( *(*pot_type)++ ) {
    case 0: //LogarithmicHaloPotential, 4 arguments
      potentialArgs->potentialEval= &LogarithmicHaloPotentialEval;
      potentialArgs->planarRforce= &LogarithmicHaloPotentialPlanarRforce;
      potentialArgs->planarphitorque= &LogarithmicHaloPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &LogarithmicHaloPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &LogarithmicHaloPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &LogarithmicHaloPotentialPlanarRphideriv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 1: //DehnenBarPotential, 6 arguments
      potentialArgs->planarRforce= &DehnenBarPotentialPlanarRforce;
      potentialArgs->planarphitorque= &DehnenBarPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &DehnenBarPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &DehnenBarPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &DehnenBarPotentialPlanarRphideriv;
      potentialArgs->nargs= 6;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 2: //TransientLogSpiralPotential, 8 arguments
      potentialArgs->planarRforce= &TransientLogSpiralPotentialRforce;
      potentialArgs->planarphitorque= &TransientLogSpiralPotentialphitorque;
      potentialArgs->planarR2deriv= &TransientLogSpiralPotentialR2deriv;
      potentialArgs->planarphi2deriv= &TransientLogSpiralPotentialphi2deriv;
      potentialArgs->planarRphideriv= &TransientLogSpiralPotentialRphideriv;
      potentialArgs->nargs= 8;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 3: //SteadyLogSpiralPotential, 8 arguments
      potentialArgs->planarRforce= &SteadyLogSpiralPotentialRforce;
      potentialArgs->planarphitorque= &SteadyLogSpiralPotentialphitorque;
      potentialArgs->planarR2deriv= &SteadyLogSpiralPotentialR2deriv;
      potentialArgs->planarphi2deriv= &SteadyLogSpiralPotentialphi2deriv;
      potentialArgs->planarRphideriv= &SteadyLogSpiralPotentialRphideriv;
      potentialArgs->nargs= 8;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 4: //EllipticalDiskPotential, 6 arguments
      potentialArgs->planarRforce= &EllipticalDiskPotentialRforce;
      potentialArgs->planarphitorque= &EllipticalDiskPotentialphitorque;
      potentialArgs->planarR2deriv= &EllipticalDiskPotentialR2deriv;
      potentialArgs->planarphi2deriv= &EllipticalDiskPotentialphi2deriv;
      potentialArgs->planarRphideriv= &EllipticalDiskPotentialRphideriv;
      potentialArgs->nargs= 6;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 5: //MiyamotoNagaiPotential, 3 arguments
      potentialArgs->potentialEval= &MiyamotoNagaiPotentialEval;
      potentialArgs->planarRforce= &MiyamotoNagaiPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &MiyamotoNagaiPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 6: //LopsidedDiskPotential, 4 arguments
      potentialArgs->planarRforce= &LopsidedDiskPotentialRforce;
      potentialArgs->planarphitorque= &LopsidedDiskPotentialphitorque;
      potentialArgs->planarR2deriv= &LopsidedDiskPotentialR2deriv;
      potentialArgs->planarphi2deriv= &LopsidedDiskPotentialphi2deriv;
      potentialArgs->planarRphideriv= &LopsidedDiskPotentialRphideriv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 7: //PowerSphericalPotential, 2 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialEval;
      potentialArgs->planarRforce= &PowerSphericalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &PowerSphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 8: //HernquistPotential, 2 arguments
      potentialArgs->potentialEval= &HernquistPotentialEval;
      potentialArgs->planarRforce= &HernquistPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &HernquistPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 9: //NFWPotential, 2 arguments
      potentialArgs->potentialEval= &NFWPotentialEval;
      potentialArgs->planarRforce= &NFWPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &NFWPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 10: //JaffePotential, 2 arguments
      potentialArgs->potentialEval= &JaffePotentialEval;
      potentialArgs->planarRforce= &JaffePotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &JaffePotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 11: //DoubleExponentialDiskPotential, XX arguments
      potentialArgs->potentialEval= &DoubleExponentialDiskPotentialEval;
      potentialArgs->planarRforce= &DoubleExponentialDiskPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &DoubleExponentialDiskPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      //Look at pot_args to figure out the number of arguments
      potentialArgs->nargs= (int) (5 + 4 * *(*pot_args+4) );
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 12: //FlattenedPowerPotential, 4 arguments
      potentialArgs->potentialEval= &FlattenedPowerPotentialEval;
      potentialArgs->planarRforce= &FlattenedPowerPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &FlattenedPowerPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 14: //IsochronePotential, 2 arguments
      potentialArgs->potentialEval= &IsochronePotentialEval;
      potentialArgs->planarRforce= &IsochronePotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &IsochronePotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 15: //PowerSphericalPotentialwCutoff, 3 arguments
      potentialArgs->potentialEval= &PowerSphericalPotentialwCutoffEval;
      potentialArgs->planarRforce= &PowerSphericalPotentialwCutoffPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &PowerSphericalPotentialwCutoffPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 16: //KuzminKutuzovStaeckelPotential, 3 arguments
      potentialArgs->potentialEval= &KuzminKutuzovStaeckelPotentialEval;
      potentialArgs->planarRforce= &KuzminKutuzovStaeckelPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &KuzminKutuzovStaeckelPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 17: //PlummerPotential, 2 arguments
      potentialArgs->potentialEval= &PlummerPotentialEval;
      potentialArgs->planarRforce= &PlummerPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &PlummerPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 18: //PseudoIsothermalPotential, 2 arguments
      potentialArgs->potentialEval= &PseudoIsothermalPotentialEval;
      potentialArgs->planarRforce= &PseudoIsothermalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &PseudoIsothermalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 19: //KuzminDiskPotential, 2 arguments
      potentialArgs->potentialEval= &KuzminDiskPotentialEval;
      potentialArgs->planarRforce= &KuzminDiskPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &KuzminDiskPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 20: //BurkertPotential, 2 arguments
      potentialArgs->potentialEval= &BurkertPotentialEval;
      potentialArgs->planarRforce= &BurkertPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &BurkertPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 21: // TriaxialHernquistPotential, lots of arguments
      potentialArgs->planarRforce = &EllipsoidalPotentialPlanarRforce;
      potentialArgs->planarphitorque = &EllipsoidalPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &EllipsoidalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &EllipsoidalPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &EllipsoidalPotentialPlanarRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialHernquistPotentialpsi;
      potentialArgs->mdens= &TriaxialHernquistPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialHernquistPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 22: // TriaxialNFWPotential, lots of arguments
      potentialArgs->planarRforce = &EllipsoidalPotentialPlanarRforce;
      potentialArgs->planarphitorque = &EllipsoidalPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &EllipsoidalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &EllipsoidalPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &EllipsoidalPotentialPlanarRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialNFWPotentialpsi;
      potentialArgs->mdens= &TriaxialNFWPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialNFWPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 23: // TriaxialJaffePotential, lots of arguments
      potentialArgs->planarRforce = &EllipsoidalPotentialPlanarRforce;
      potentialArgs->planarphitorque = &EllipsoidalPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &EllipsoidalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &EllipsoidalPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &EllipsoidalPotentialPlanarRphideriv;
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
      potentialArgs->planarRforce= &SCFPotentialPlanarRforce;
      potentialArgs->planarphitorque= &SCFPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &SCFPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &SCFPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &SCFPotentialPlanarRphideriv;
      // Layout header: a, isNonAxi, N, L, M, Nt (6 doubles), then either the
      // static coefficient arrays (Nt==0) or tgrid + time-PPoly blocks (Nt>0),
      // then 11 cache slots (type + 4 coords[R,Z,phi,t] + 6 values).
      potentialArgs->nargs= (int) ( *(*pot_args+5) == 0
        ? 6 + (1 + *(*pot_args+1)) * *(*pot_args+2) * *(*pot_args+3) * *(*pot_args+4) + 11
        : 6 + *(*pot_args+5) + (1 + *(*pot_args+1)) * *(*pot_args+2) * *(*pot_args+3) * *(*pot_args+4) * 4 * ( *(*pot_args+5) - 1 ) + 11 );
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 25: //SoftenedNeedleBarPotential, 13 arguments
      potentialArgs->potentialEval= &SoftenedNeedleBarPotentialEval;
      potentialArgs->planarRforce= &SoftenedNeedleBarPotentialPlanarRforce;
      potentialArgs->planarphitorque= &SoftenedNeedleBarPotentialPlanarphitorque;
      // Planar Hessian for the 2D variational equations (integrate_dxdv).
      potentialArgs->planarR2deriv= &SoftenedNeedleBarPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &SoftenedNeedleBarPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &SoftenedNeedleBarPotentialPlanarRphideriv;
      potentialArgs->nargs= 13;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 26: //DiskSCFPotential, nsigma+3 arguments
      potentialArgs->potentialEval= &DiskSCFPotentialEval;
      potentialArgs->planarRforce= &DiskSCFPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &DiskSCFPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= (int) **pot_args + 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 27: // SpiralArmsPotential, 10 arguments + array of Cs
      potentialArgs->planarRforce = &SpiralArmsPotentialPlanarRforce;
      potentialArgs->planarphitorque = &SpiralArmsPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &SpiralArmsPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &SpiralArmsPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &SpiralArmsPotentialPlanarRphideriv;
      potentialArgs->nargs = (int) 10 + **pot_args;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 28: //CosmphiDiskPotential, 9 arguments
      potentialArgs->planarRforce= &CosmphiDiskPotentialRforce;
      potentialArgs->planarphitorque= &CosmphiDiskPotentialphitorque;
      potentialArgs->planarR2deriv= &CosmphiDiskPotentialR2deriv;
      potentialArgs->planarphi2deriv= &CosmphiDiskPotentialphi2deriv;
      potentialArgs->planarRphideriv= &CosmphiDiskPotentialRphideriv;
      potentialArgs->nargs= 9;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 29: //HenonHeilesPotential, 1 argument
      potentialArgs->planarRforce= &HenonHeilesPotentialRforce;
      potentialArgs->planarphitorque= &HenonHeilesPotentialphitorque;
      potentialArgs->planarR2deriv= &HenonHeilesPotentialR2deriv;
      potentialArgs->planarphi2deriv= &HenonHeilesPotentialphi2deriv;
      potentialArgs->planarRphideriv= &HenonHeilesPotentialRphideriv;
      potentialArgs->nargs= 1;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 30: // PerfectEllipsoidPotential, lots of arguments
      potentialArgs->planarRforce = &EllipsoidalPotentialPlanarRforce;
      potentialArgs->planarphitorque = &EllipsoidalPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &EllipsoidalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &EllipsoidalPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &EllipsoidalPotentialPlanarRphideriv;
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
    case 33: //DehnenCoreSphericalpotential
      potentialArgs->potentialEval= &DehnenCoreSphericalPotentialEval;
      potentialArgs->planarRforce= &DehnenCoreSphericalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &DehnenCoreSphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 2;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 34: //DehnenSphericalpotential
      potentialArgs->potentialEval= &DehnenSphericalPotentialEval;
      potentialArgs->planarRforce= &DehnenSphericalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &DehnenSphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 35: //HomogeneousSpherePotential, 3 arguments
      potentialArgs->potentialEval= &HomogeneousSpherePotentialEval;
      potentialArgs->planarRforce= &HomogeneousSpherePotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &HomogeneousSpherePotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
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
      potentialArgs->planarRforce = &SphericalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &SphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      // Also assign functions specific to SphericalPotential
      potentialArgs->revaluate= &interpSphericalPotentialrevaluate;
      potentialArgs->rforce= &interpSphericalPotentialrforce;
      potentialArgs->r2deriv= &interpSphericalPotentialr2deriv;
      potentialArgs->nargs = 6;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 37: // TriaxialGaussianPotential, lots of arguments
      potentialArgs->planarRforce = &EllipsoidalPotentialPlanarRforce;
      potentialArgs->planarphitorque = &EllipsoidalPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &EllipsoidalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &EllipsoidalPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &EllipsoidalPotentialPlanarRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &TriaxialGaussianPotentialpsi;
      potentialArgs->mdens= &TriaxialGaussianPotentialmdens;
      potentialArgs->mdensDeriv= &TriaxialGaussianPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 38: // PowerTriaxialPotential, lots of arguments
      potentialArgs->planarRforce = &EllipsoidalPotentialPlanarRforce;
      potentialArgs->planarphitorque = &EllipsoidalPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &EllipsoidalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &EllipsoidalPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &EllipsoidalPotentialPlanarRphideriv;
      // Also assign functions specific to EllipsoidalPotential
      potentialArgs->psi= &PowerTriaxialPotentialpsi;
      potentialArgs->mdens= &PowerTriaxialPotentialmdens;
      potentialArgs->mdensDeriv= &PowerTriaxialPotentialmdensDeriv;
      potentialArgs->nargs = (int) (30 + *(*pot_args+16) + 2 * *(*pot_args + (int) (*(*pot_args+16) + 29)));
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 39: //NonInertialFrameForce, 23 arguments (10 caching ones)
      // Time-dependent inputs passed as tfuncs (called from C every step); the
      // cinterp=True variant (case 45) precomputes them as GSL splines instead.
      potentialArgs->planarRforceVelocity= &NonInertialFrameForcePlanarRforce;
      potentialArgs->planarphitorqueVelocity= &NonInertialFrameForcePlanarphitorque;
      potentialArgs->nargs= 23;
      potentialArgs->ntfuncs= (int) ( 3 * *(*pot_args + 12) * ( 1 + 2 * *(*pot_args + 11) ) \
                                + ( 6 - 4 * ( *(*pot_args + 13) ) ) * *(*pot_args + 15) );
      potentialArgs->requiresVelocity= true;
      break;
    case 45: //NonInertialFrameForce with cinterp=True (on-the-fly C splines)
      // Same force as case 39, but time-dependent inputs are evaluated from GSL
      // splines built by initNonInertialFrameForceSplines (Omegadot = spline
      // derivative of Omega) rather than tfuncs; the spline block precedes the
      // 23 case-39 args, plus tmin,tmax (args 23,24). nargs=25, ntfuncs=0. The
      // shared force code branches on spline1d!=NULL.
      potentialArgs->planarRforceVelocity= &NonInertialFrameForcePlanarRforce;
      potentialArgs->planarphitorqueVelocity= &NonInertialFrameForcePlanarphitorque;
      potentialArgs->nargs= 25;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= true;
      break;
    case 40: //NullPotential, no arguments (only supported for orbit int)
      potentialArgs->potentialEval= &ZeroForce;
      potentialArgs->planarRforce= &ZeroPlanarForce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &ZeroPlanarForce;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 0;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 41: //EinastoPotential
      potentialArgs->potentialEval= &SphericalPotentialEval;
      potentialArgs->planarRforce = &SphericalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &SphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      // Also assign functions specific to SphericalPotential
      potentialArgs->revaluate= &EinastoPotentialrevaluate;
      potentialArgs->rforce= &EinastoPotentialrforce;
      potentialArgs->r2deriv= &EinastoPotentialr2deriv;
      potentialArgs->nargs = 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 42: //TwoPowerSphericalPotential, 4 arguments
      potentialArgs->potentialEval= &TwoPowerSphericalPotentialEval;
      potentialArgs->planarRforce= &TwoPowerSphericalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &TwoPowerSphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 43: // TwoPowerTriaxialPotential, lots of arguments
      potentialArgs->planarRforce = &EllipsoidalPotentialPlanarRforce;
      potentialArgs->planarphitorque = &EllipsoidalPotentialPlanarphitorque;
      potentialArgs->planarR2deriv = &EllipsoidalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv = &EllipsoidalPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv = &EllipsoidalPotentialPlanarRphideriv;
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
      potentialArgs->planarRforce= &MultipoleExpansionPotentialPlanarRforce;
      potentialArgs->planarphitorque= &MultipoleExpansionPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &MultipoleExpansionPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &MultipoleExpansionPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &MultipoleExpansionPotentialPlanarRphideriv;
      potentialArgs->nargs= 0; // arguments handled in the initialization code run for this potential
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case 46: //ExpTruncNFWPotential
      potentialArgs->potentialEval= &SphericalPotentialEval;
      potentialArgs->planarRforce = &SphericalPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &SphericalPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      // Also assign functions specific to SphericalPotential
      potentialArgs->revaluate= &ExpTruncNFWPotentialrevaluate;
      potentialArgs->rforce= &ExpTruncNFWPotentialrforce;
      potentialArgs->r2deriv= &ExpTruncNFWPotentialr2deriv;
      potentialArgs->nargs = 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
//////////////////////////////// WRAPPERS /////////////////////////////////////
    case -1: //DehnenSmoothWrapperPotential
      potentialArgs->potentialEval= &DehnenSmoothWrapperPotentialEval;
      potentialArgs->planarRforce= &DehnenSmoothWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &DehnenSmoothWrapperPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &DehnenSmoothWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &DehnenSmoothWrapperPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &DehnenSmoothWrapperPotentialPlanarRphideriv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -2: //SolidBodyRotationWrapperPotential
      potentialArgs->planarRforce= &SolidBodyRotationWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &SolidBodyRotationWrapperPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &SolidBodyRotationWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &SolidBodyRotationWrapperPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &SolidBodyRotationWrapperPotentialPlanarRphideriv;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -3: //OblateStaeckelWrapperPotential
      potentialArgs->potentialEval= &OblateStaeckelWrapperPotentialEval;
      potentialArgs->planarRforce= &OblateStaeckelWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      // Planar 2nd derivatives for the planar variational equations
      // (integrate_dxdv); axisymmetric, so the phi-derivatives vanish.
      potentialArgs->planarR2deriv= &OblateStaeckelWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= (int) 5;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -4: //CorotatingRotationWrapperPotential
      potentialArgs->planarRforce= &CorotatingRotationWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &CorotatingRotationWrapperPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &CorotatingRotationWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &CorotatingRotationWrapperPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &CorotatingRotationWrapperPotentialPlanarRphideriv;
      potentialArgs->nargs= 5;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -5: //GaussianAmplitudeWrapperPotential
      potentialArgs->planarRforce= &GaussianAmplitudeWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &GaussianAmplitudeWrapperPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &GaussianAmplitudeWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &GaussianAmplitudeWrapperPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &GaussianAmplitudeWrapperPotentialPlanarRphideriv;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    case -6: //MovingObjectPotential
      potentialArgs->planarRforce= &MovingObjectPotentialPlanarRforce;
      potentialArgs->planarphitorque= &MovingObjectPotentialPlanarphitorque;
      // Planar 2D Hessian for the planar variational equations
      // (integrate_dxdv): the kernel's planar Hessian at the shifted point
      // x-x_obj(t); nonaxisymmetric (the object is off-center). Only used
      // when the kernel's planar Hessian is in C (gated by hasC_dxdv =
      // _check_c(kernel, dxdv=True) on the Python side).
      potentialArgs->planarR2deriv= &MovingObjectPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &MovingObjectPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &MovingObjectPotentialPlanarRphideriv;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    //ChandrasekharDynamicalFrictionForce omitted, bc no planar version
    //RotateAndTiltWrapperPotential omitted, bc no planar version
    case -9: //TimeDependentAmplitudeWrapperPotential
      potentialArgs->potentialEval= &TimeDependentAmplitudeWrapperPotentialEval;
      potentialArgs->planarRforce= &TimeDependentAmplitudeWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &TimeDependentAmplitudeWrapperPotentialPlanarphitorque;
      potentialArgs->planarR2deriv= &TimeDependentAmplitudeWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &TimeDependentAmplitudeWrapperPotentialPlanarphi2deriv;
      potentialArgs->planarRphideriv= &TimeDependentAmplitudeWrapperPotentialPlanarRphideriv;
      potentialArgs->nargs= 4;
      potentialArgs->ntfuncs= 1;
      potentialArgs->requiresVelocity= false;
      break;
    case -10: //KuzminLikeWrapperPotential
      potentialArgs->potentialEval= &KuzminLikeWrapperPotentialEval;
      potentialArgs->planarRforce= &KuzminLikeWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      potentialArgs->planarR2deriv= &KuzminLikeWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;
    // FDMDynamicalFrictionForce omitted, bc no planar version
    case -12: //CylindricallySeparablePotentialWrapper
      potentialArgs->potentialEval= &CylindricallySeparablePotentialWrapperPotentialEval;
      potentialArgs->planarRforce= &CylindricallySeparablePotentialWrapperPotentialPlanarRforce;
      potentialArgs->planarphitorque= &ZeroPlanarForce;
      // Planar 2nd derivatives for the planar variational equations
      // (integrate_dxdv); axisymmetric, so the phi-derivatives vanish.
      potentialArgs->planarR2deriv= &CylindricallySeparablePotentialWrapperPotentialPlanarR2deriv;
      potentialArgs->planarphi2deriv= &ZeroPlanarForce;
      potentialArgs->planarRphideriv= &ZeroPlanarForce;
      potentialArgs->nargs= (int) 3;
      potentialArgs->ntfuncs= 0;
      potentialArgs->requiresVelocity= false;
      break;

    }
    int setupSplines = *(*pot_type-1) == -6 ? 1 : 0;
    int initSCFData = *(*pot_type-1) == 24 ? 1 : 0;
    int initMultipoleExpansionData = *(*pot_type-1) == 44 ? 1 : 0;
    int setupNonInertialFrameForceSplines = *(*pot_type-1) == 45 ? 1 : 0;
    if ( *(*pot_type-1) < 0) { // Parse wrapped potential for wrappers
      potentialArgs->nwrapped= (int) *(*pot_args)++;
      potentialArgs->wrappedPotentialArg= \
	(struct potentialArg *) malloc ( potentialArgs->nwrapped	\
					 * sizeof (struct potentialArg) );
      parse_leapFuncArgs(potentialArgs->nwrapped,
			 potentialArgs->wrappedPotentialArg,
			 pot_type,pot_args,pot_tfuncs);
    }
    if (setupSplines) initPlanarMovingObjectSplines(potentialArgs, pot_args);
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
EXPORT void integratePlanarOrbit(int nobj,
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
    parse_leapFuncArgs(npot,potentialArgs+ii*npot,
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
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  case 1: //RK4
    odeint_func= &bovy_rk4;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  case 3: //symplec4
    odeint_func= &symplec4;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  case 4: //symplec6
    odeint_func= &symplec6;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  case 6: //DOP853
    odeint_func= &dop853;
    odeint_deriv_func= &evalPlanarRectDeriv;
    dim= 4;
    break;
  case 7: //ias15
    odeint_func= &wez_ias15;
    odeint_deriv_func= &evalPlanarRectForce;
    dim= 2;
    break;
  }
#pragma omp parallel for schedule(dynamic,ORBITS_CHUNKSIZE) private(ii,jj) num_threads(max_threads)
  for (ii=0; ii < nobj; ii++) {
    polar_to_rect_galpy(yo+4*ii);
    odeint_func(odeint_deriv_func,dim,yo+4*ii,nt,dt,t+nt*ii*indiv_t,
		npot,potentialArgs+omp_get_thread_num()*npot,rtol,atol,
		result+4*nt*ii,err+ii);
    for (jj= 0; jj < nt; jj++)
      rect_to_polar_galpy(result+4*jj+4*nt*ii);
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
EXPORT void integratePlanarOrbit_sos(
    int nobj,
	double *yo,
	int npsi,
	double *psi,
    int indiv_psi,
    int surface,
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
    parse_leapFuncArgs(npot,potentialArgs+ii*npot,
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
  dim= 5;
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
  switch ( surface ) {
    case 0: // x=0
      odeint_deriv_func= &evalPlanarSOSDerivx;
      break;
    case 1: // y=0
      odeint_deriv_func= &evalPlanarSOSDerivy;
      break;
  }

#pragma omp parallel for schedule(dynamic,ORBITS_CHUNKSIZE) private(ii,jj) num_threads(max_threads)
  for (ii=0; ii < nobj; ii++) {
    polar_to_sos_galpy(yo+dim*ii,surface);
    odeint_func(odeint_deriv_func,dim,yo+dim*ii,npsi,dpsi,psi+npsi*ii*indiv_psi,
		npot,potentialArgs+omp_get_thread_num()*npot,rtol,atol,
		result+dim*npsi*ii,err+ii);
    for (jj=0; jj < npsi; jj++)
      sos_to_polar_galpy(result+dim*jj+dim*npsi*ii,surface);
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
EXPORT void integratePlanarOrbit_dxdv(double *yo,
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
  int dim;
  struct potentialArg * potentialArgs= (struct potentialArg *) malloc ( npot * sizeof (struct potentialArg) );
  parse_leapFuncArgs(npot,potentialArgs,&pot_type,&pot_args,&pot_tfuncs);
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
  // Non-symplectic (RK) integrators propagate the 8D deviation via the planar
  // variational RHS evalPlanarRectDeriv_dxdv (dim=8); the symplectic ones
  // (leapfrog=0/symplec4=3/symplec6=4) instead carry the deviation through the
  // closed-form drift/kick tangent maps of the *_dxdv_planar steppers (nde=1
  // deviation column). ias15 has no dxdv path and is blocked upstream by
  // Orbit.integrate_dxdv (check_integrator).
  odeint_func= NULL;
  odeint_deriv_func= &evalPlanarRectDeriv_dxdv;
  dim= 8;
  switch ( odeint_type ) {
  case 1: //RK4
    odeint_func= &bovy_rk4;
    break;
  case 2: //RK6
    odeint_func= &bovy_rk6;
    break;
  case 5: //DOPR54
    odeint_func= &bovy_dopr54;
    break;
  case 6: //DOP853
    odeint_func= &dop853;
    break;
  }
  switch ( odeint_type ) {
  case 0: //leapfrog
    leapfrog_dxdv_planar(1,yo,nt,dt,t,npot,potentialArgs,rtol,atol,result,err);
    break;
  case 3: //symplec4
    symplec4_dxdv_planar(1,yo,nt,dt,t,npot,potentialArgs,rtol,atol,result,err);
    break;
  case 4: //symplec6
    symplec6_dxdv_planar(1,yo,nt,dt,t,npot,potentialArgs,rtol,atol,result,err);
    break;
  default: //RK method selected above
    odeint_func(odeint_deriv_func,dim,yo,nt,dt,t,npot,potentialArgs,rtol,atol,
		result,err);
  }
  //Free allocated memory
  free_potentialArgs(npot,potentialArgs);
  free(potentialArgs);
  //Done!
}

void evalPlanarRectForce(double t, double *q, double *a,
			 int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phitorque;
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,potentialArgs);
  phitorque= calcPlanarphitorque(R,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a--= sinphi*Rforce+1./R*cosphi*phitorque;
}
void evalPlanarRectDeriv(double t, double *q, double *a,
			 int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phitorque,vR,vT;
  //first two derivatives are just the velocities
  *a++= *(q+2);
  *a++= *(q+3);
  //Rest is force
  //q is rectangular so calculate R and phi, vR and vT (for dissipative)
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  vR=  *(q+2) * cosphi + *(q+3) * sinphi;
  vT= -*(q+2) * sinphi + *(q+3) * cosphi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,potentialArgs,vR,vT);
  phitorque= calcPlanarphitorque(R,phi,t,nargs,potentialArgs,vR,vT);
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a= sinphi*Rforce+1./R*cosphi*phitorque;
}

void evalPlanarSOSDerivx(double psi, double *q, double *a,
		                 int nargs, struct potentialArg * potentialArgs){
  // q= (y,vy,A,t,psi); to save operations, we reuse a first for the
  // rectForce then for the actual RHS
  // Note also that we keep track of psi in q+4, not in psi! This is
  // such that we can avoid having to convert psi to psi+psi0
  // q+4 starts as psi0 and then just increments as psi (exactly)
  double sinpsi,cospsi,psidot,x,y,R,phi,sinphi,cosphi,Rforce,phitorque,vR,vT;
  sinpsi= sin( *(q+4) );
  cospsi= cos( *(q+4) );
  // Calculate forces, put them in a+2, a+3
  //q is rectangular so calculate R and phi
  x= *(q+2) * sinpsi;
  y= *(q  );
  R= sqrt(x*x+y*y);
  phi= atan2( y ,x );
  sinphi= y/R;
  cosphi= x/R;
  vR=  *(q+2) * cospsi * cosphi + *(q+1) * sinphi;
  vT= -*(q+2) * cospsi * sinphi + *(q+1) * cosphi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,*(q+3),nargs,potentialArgs,vR,vT);
  phitorque= calcPlanarphitorque(R,phi,*(q+3),nargs,potentialArgs,vR,vT);
  *(a+2)= cosphi*Rforce-1./R*sinphi*phitorque;
  *(a+3)= sinphi*Rforce+1./R*cosphi*phitorque;
  // Now calculate the RHS of the ODE
  psidot= cospsi * cospsi - sinpsi * *(a+2) / ( *(q+2) );
  *(a  )= *(q+1) / psidot;
  *(a+1)= *(a+3) / psidot;
  *(a+2)= cospsi * ( *(q+2) * sinpsi + *(a+2) ) / psidot;
  *(a+3)= 1./psidot;
  *(a+4)= 1.; // dpsi / dpsi to keep track of psi
}

void evalPlanarSOSDerivy(double psi, double *q, double *a,
		                 int nargs, struct potentialArg * potentialArgs){
  // q= (x,vx,A,t,psi); to save operations, we reuse a first for the
  // rectForce then for the actual RHS
  // Note also that we keep track of psi in q+4, not in psi! This is
  // such that we can avoid having to convert psi to psi+psi0
  // q+4 starts as psi0 and then just increments as psi (exactly)
  double sinpsi,cospsi,psidot,x,y,R,phi,sinphi,cosphi,Rforce,phitorque,vR,vT;
  sinpsi= sin( *(q+4) );
  cospsi= cos( *(q+4) );
  // Calculate forces, put them in a+2, a+3
  //q is rectangular so calculate R and phi
  x= *(q  );
  y= *(q+2) * sinpsi;
  R= sqrt(x*x+y*y);
  phi= atan2( y ,x );
  sinphi= y/R;
  cosphi= x/R;
  vR=  *(q+1 ) * cosphi + *(q+2) * cospsi * sinphi;
  vT= -*(q+1 ) * sinphi + *(q+2) * cospsi * cosphi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,*(q+3),nargs,potentialArgs,vR,vT);
  phitorque= calcPlanarphitorque(R,phi,*(q+3),nargs,potentialArgs,vR,vT);
  *(a+2)= cosphi*Rforce-1./R*sinphi*phitorque;
  *(a+3)= sinphi*Rforce+1./R*cosphi*phitorque;
  // Now calculate the RHS of the ODE
  psidot= cospsi * cospsi - sinpsi * *(a+3) / ( *(q+2) );
  *(a  )= *(q+1) / psidot;
  *(a+1)= *(a+2) / psidot;
  *(a+2)= cospsi * ( *(q+2) * sinpsi + *(a+3) ) / psidot;
  *(a+3)= 1./psidot;
  *(a+4)= 1.; // dpsi / dpsi to keep track of psi
}

void evalPlanarRectDeriv_dxdv(double t, double *q, double *a,
			      int nargs, struct potentialArg * potentialArgs){
  double sinphi, cosphi, x, y, phi,R,Rforce,phitorque;
  double R2deriv, phi2deriv, Rphideriv, dFxdx, dFxdy, dFydx, dFydy;
  //first two derivatives are just the velocities
  *a++= *(q+2);
  *a++= *(q+3);
  //Rest is force
  //q is rectangular so calculate R and phi
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Calculate the forces
  Rforce= calcPlanarRforce(R,phi,t,nargs,potentialArgs);
  phitorque= calcPlanarphitorque(R,phi,t,nargs,potentialArgs);
  *a++= cosphi*Rforce-1./R*sinphi*phitorque;
  *a++= sinphi*Rforce+1./R*cosphi*phitorque;
  //dx derivatives are just dv
  *a++= *(q+6);
  *a++= *(q+7);
  //for the dv derivatives we need also R2deriv, phi2deriv, and Rphideriv
  R2deriv= calcPlanarR2deriv(R,phi,t,nargs,potentialArgs);
  phi2deriv= calcPlanarphi2deriv(R,phi,t,nargs,potentialArgs);
  Rphideriv= calcPlanarRphideriv(R,phi,t,nargs,potentialArgs);
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
  *a= dFydx * *(q+4) + dFydy * *(q+5);
}

void initPlanarMovingObjectSplines(struct potentialArg * potentialArgs, double ** pot_args){
  gsl_interp_accel *x_accel_ptr = gsl_interp_accel_alloc();
  gsl_interp_accel *y_accel_ptr = gsl_interp_accel_alloc();
  int nPts = (int) **pot_args;

  gsl_spline *x_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);
  gsl_spline *y_spline = gsl_spline_alloc(gsl_interp_cspline, nPts);

  double * t_arr = *pot_args+1;
  double * x_arr = t_arr+1*nPts;
  double * y_arr = t_arr+2*nPts;

  double * t= (double *) malloc ( nPts * sizeof (double) );
  double tf = *(t_arr+3*nPts+2);
  double to = *(t_arr+3*nPts+1);

  int ii;
  for (ii=0; ii < nPts; ii++)
    *(t+ii) = (t_arr[ii]-to)/(tf-to);

  gsl_spline_init(x_spline, t, x_arr, nPts);
  gsl_spline_init(y_spline, t, y_arr, nPts);

  potentialArgs->nspline1d= 2;
  potentialArgs->spline1d= (gsl_spline **) malloc ( 2*sizeof ( gsl_spline *) );
  potentialArgs->acc1d= (gsl_interp_accel **) \
    malloc ( 2 * sizeof ( gsl_interp_accel * ) );
  *potentialArgs->spline1d = x_spline;
  *potentialArgs->acc1d = x_accel_ptr;
  *(potentialArgs->spline1d+1)= y_spline;
  *(potentialArgs->acc1d+1)= y_accel_ptr;

  *pot_args = *pot_args+ (int) (1+3*nPts);
  free(t);
}

/*
  Planar symplectic variational (state-transition) integration.

  Planar (2D) analog of leapfrog_dxdv/symplec4_dxdv/symplec6_dxdv in
  integrateFullOrbit.c: mirrors leapfrog/symplec4/symplec6 in
  galpy/util/bovy_symplecticode.c (same DKD ordering, coefficients, and
  interior micro-step drift merges) but additionally propagates nde phase-space
  deviation columns through the exact, closed-form tangent maps of each drift
  and kick:
       drift M_D = [[I2, h I2],[0, I2]]   kick M_K = [[I2, 0],[h K2, I2]]
  with K2 the symmetric conservative planar Cartesian Hessian (-grad grad Phi)
  assembled once per kick from the base position by evalPlanarRectForce_dxdv
  (reusing the evalPlanarRectDeriv_dxdv K2 block). Each elementary map is exactly
  symplectic for a conservative (symmetric-K2) system, so the per-step product is
  symplectic to machine precision. Only nde=1 is wired here (8-wide dxdv). The
  split arrays qo/po hold the base 2-vector in block 0 and deviation column j in
  block j (ndim=2*(nde+1)); the yo/result buffers use the interleaved (pos2,vel2)
  blocks (4*(nde+1) wide). Dissipative (velocity-dependent) forces never reach
  this path: galpy reroutes symplectic+dissipative to a non-symplectic integrator
  upstream, so the kick is always the conservative, explicit map above.
*/
// Augmented drift qn = q + dt p over all ndim entries (base + deviations).
static inline void leapq_aug_planar(int dim, double *q, double *p, double dt,
				    double *qn){
  int ii;
  for (ii=0; ii < dim; ii++) (*qn++)= (*q++) + dt * (*p++);
}
// Augmented kick pn = p + dt a over all ndim entries; a holds the base
// acceleration in block 0 and K2.dq_j in block j (evalPlanarRectForce_dxdv).
static inline void leapp_aug_planar(int dim, double *p, double dt, double *a,
				    double *pn){
  int ii;
  for (ii=0; ii < dim; ii++) (*pn++)= (*p++) + dt * (*a++);
}
// Repack split qo/po into the interleaved (base pos2,vel2, then per-column
// dpos2,dvel2) output layout the Python caller consumes.
static inline void save_qp_aug_planar(int nde, double *qo, double *po,
				      double *result){
  int bb, kk;
  for (bb=0; bb <= nde; bb++) {
    for (kk=0; kk < 2; kk++) *result++= *(qo+2*bb+kk);
    for (kk=0; kk < 2; kk++) *result++= *(po+2*bb+kk);
  }
}
// Fill a with the base acceleration (block 0, byte-identical to
// evalPlanarRectForce) and, per deviation column j, the kick tangent K2.dq_j
// (block j). K2 is the conservative planar Cartesian Hessian assembled exactly
// as in evalPlanarRectDeriv_dxdv (symmetric: dFxdy=dFydx).
void evalPlanarRectForce_dxdv(double t, double *q, double *a,
			      int nargs, struct potentialArg * potentialArgs,
			      int nde){
  double sinphi, cosphi, x, y, phi, R, Rforce, phitorque;
  double R2deriv, phi2deriv, Rphideriv, dFxdx, dFxdy, dFydy, dx, dy;
  int jj;
  //q is rectangular so calculate R and phi (base block only)
  x= *q;
  y= *(q+1);
  R= sqrt(x*x+y*y);
  phi= acos(x/R);
  sinphi= y/R;
  cosphi= x/R;
  if ( y < 0. ) phi= 2.*M_PI-phi;
  //Base acceleration: identical calls/order to evalPlanarRectForce (bit-
  //identical base trajectory vs a plain leapfrog/symplec4/symplec6 run)
  Rforce= calcPlanarRforce(R,phi,t,nargs,potentialArgs);
  phitorque= calcPlanarphitorque(R,phi,t,nargs,potentialArgs);
  *a    = cosphi*Rforce-1./R*sinphi*phitorque;
  *(a+1)= sinphi*Rforce+1./R*cosphi*phitorque;
  if ( nde == 0 ) return;
  //Conservative planar Cartesian Hessian K2 (same aggregators as
  //evalPlanarRectDeriv_dxdv, which likewise uses the conservative force)
  R2deriv= calcPlanarR2deriv(R,phi,t,nargs,potentialArgs);
  phi2deriv= calcPlanarphi2deriv(R,phi,t,nargs,potentialArgs);
  Rphideriv= calcPlanarRphideriv(R,phi,t,nargs,potentialArgs);
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
  dFydy= -sinphi*sinphi*R2deriv
    -2.*sinphi*cosphi/R/R*phitorque
    -2.*sinphi*cosphi/R*Rphideriv
    +cosphi*cosphi/R*Rforce
    -cosphi*cosphi/R/R*phi2deriv;
  //Kick tangent per deviation column: dv_j += h K2.dq_j (K2 symmetric)
  for (jj=1; jj <= nde; jj++) {
    dx= *(q+2*jj);
    dy= *(q+2*jj+1);
    *(a+2*jj  )= dFxdx*dx+dFxdy*dy;
    *(a+2*jj+1)= dFxdy*dx+dFydy*dy;
  }
}

// Augmented DKD leapfrog (planar; mirrors leapfrog in bovy_symplecticode.c).
void leapfrog_dxdv_planar(int nde,
			  double * yo,
			  int nt, double dt, double *t,
			  int nargs, struct potentialArg * potentialArgs,
			  double rtol, double atol,
			  double *result, int * err){
  int ndim= 2*(nde+1);
  double *qo= (double *) malloc ( ndim * sizeof(double) );
  double *po= (double *) malloc ( ndim * sizeof(double) );
  double *q12= (double *) malloc ( ndim * sizeof(double) );
  double *p12= (double *) malloc ( ndim * sizeof(double) );
  double *a= (double *) malloc ( ndim * sizeof(double) );
  int ii, jj, kk, bb;
  //unpack interleaved (pos2,vel2) blocks into split qo/po
  for (bb=0; bb <= nde; bb++) {
    for (kk=0; kk < 2; kk++) {
      *(qo+2*bb+kk)= *(yo+4*bb+kk);
      *(po+2*bb+kk)= *(yo+4*bb+2+kk);
    }
  }
  save_qp_aug_planar(nde,qo,po,result);
  result+= 2 * ndim;
  *err= 0;
  //Estimate stepsize from the BASE orbit only (dim=2): same dt a plain
  //leapfrog run would pick, so the base trajectory is bit-identical
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= leapfrog_estimate_step(&evalPlanarRectForce,2,qo,po,init_dt,t,nargs,
			       potentialArgs,rtol,atol);
  }
  long ndt= (long) (init_dt/dt);
  double to= *t;
#ifndef _WIN32
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler= handle_sigint;
  sigaction(SIGINT,&action,NULL);
#else
  if (SetConsoleCtrlHandler(CtrlHandler, TRUE)){}
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0;
#ifdef USING_COVERAGE
      __gcov_dump();
// LCOV_EXCL_START
      __gcov_reset();
#endif
      break;
// LCOV_EXCL_STOP
    }
    //drift half
    leapq_aug_planar(ndim,qo,po,dt/2.,q12);
    //now drift full for a while
    for (jj=0; jj < (ndt-1); jj++){
      //kick (K at half-step position and midpoint time)
      evalPlanarRectForce_dxdv(to+dt/2.,q12,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,po,dt,a,p12);
      //drift
      leapq_aug_planar(ndim,q12,p12,dt,qo);
      //reset
      to= to+dt;
      for (kk=0; kk < ndim; kk++) {
	*(q12+kk)= *(qo+kk);
	*(po+kk)= *(p12+kk);
      }
    }
    //end with one last kick and drift
    evalPlanarRectForce_dxdv(to+dt/2.,q12,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,po,dt,a,po);
    leapq_aug_planar(ndim,q12,po,dt/2.,qo);
    to= to+dt;
    save_qp_aug_planar(nde,qo,po,result);
    result+= 2 * ndim;
  }
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  free(qo);
  free(po);
  free(q12);
  free(p12);
  free(a);
}

// Augmented 4th-order Forest-Ruth symplec4 (planar; mirrors symplec4).
void symplec4_dxdv_planar(int nde,
			  double * yo,
			  int nt, double dt, double *t,
			  int nargs, struct potentialArg * potentialArgs,
			  double rtol, double atol,
			  double *result, int * err){
  //coefficients (verbatim from bovy_symplecticode.c)
  double c1= 0.6756035959798289;
  double c4= c1;
  double c2= -0.1756035959798288;
  double c3= c2;
  double d1= 1.3512071919596578;
  double d3= d1;
  double d2= -1.7024143839193153; //d4=0
  int ndim= 2*(nde+1);
  double *qo= (double *) malloc ( ndim * sizeof(double) );
  double *po= (double *) malloc ( ndim * sizeof(double) );
  double *q12= (double *) malloc ( ndim * sizeof(double) );
  double *p12= (double *) malloc ( ndim * sizeof(double) );
  double *a= (double *) malloc ( ndim * sizeof(double) );
  int ii, jj, kk, bb;
  for (bb=0; bb <= nde; bb++) {
    for (kk=0; kk < 2; kk++) {
      *(qo+2*bb+kk)= *(yo+4*bb+kk);
      *(po+2*bb+kk)= *(yo+4*bb+2+kk);
    }
  }
  save_qp_aug_planar(nde,qo,po,result);
  result+= 2 * ndim;
  *err= 0;
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= symplec4_estimate_step(&evalPlanarRectForce,2,qo,po,init_dt,t,nargs,
			       potentialArgs,rtol,atol);
  }
  long ndt= (long) (init_dt/dt);
  double to= *t;
#ifndef _WIN32
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler= handle_sigint;
  sigaction(SIGINT,&action,NULL);
#else
  if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {}
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0;
#ifdef USING_COVERAGE
      __gcov_dump();
// LCOV_EXCL_START
      __gcov_reset();
#endif
      break;
// LCOV_EXCL_STOP
    }
    //drift for c1*dt
    leapq_aug_planar(ndim,qo,po,c1*dt,q12);
    to+= c1*dt;
    //steps ignoring q4/p4 when output is not wanted
    for (jj=0; jj < (ndt-1); jj++){
      //kick for d1*dt
      evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,po,d1*dt,a,p12);
      //drift for c2*dt
      leapq_aug_planar(ndim,q12,p12,c2*dt,qo);
      //kick for d2*dt
      to+= c2*dt;
      evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,p12,d2*dt,a,po);
      //drift for c3*dt
      leapq_aug_planar(ndim,qo,po,c3*dt,q12);
      to+= c3*dt;
      //kick for d3*dt
      evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,po,d3*dt,a,p12);
      //drift for (c4+c1)*dt
      leapq_aug_planar(ndim,q12,p12,(c4+c1)*dt,qo);
      to+= (c4+c1)*dt;
      //reset
      for (kk=0; kk < ndim; kk++) {
	*(q12+kk)= *(qo+kk);
	*(po+kk)= *(p12+kk);
      }
    }
    //steps not ignoring q4/p4 when output is wanted
    //kick for d1*dt
    evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,po,d1*dt,a,p12);
    //drift for c2*dt
    leapq_aug_planar(ndim,q12,p12,c2*dt,qo);
    //kick for d2*dt
    to+= c2*dt;
    evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,p12,d2*dt,a,po);
    //drift for c3*dt
    leapq_aug_planar(ndim,qo,po,c3*dt,q12);
    to+= c3*dt;
    //kick for d3*dt
    evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,po,d3*dt,a,p12);
    //drift for c4*dt
    leapq_aug_planar(ndim,q12,p12,c4*dt,qo);
    to+= c4*dt;
    //p4=p3
    for (kk=0; kk < ndim; kk++) *(po+kk)= *(p12+kk);
    save_qp_aug_planar(nde,qo,po,result);
    result+= 2 * ndim;
  }
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  free(qo);
  free(po);
  free(q12);
  free(p12);
  free(a);
}

// Augmented 6th-order Yoshida symplec6 (planar; mirrors symplec6).
void symplec6_dxdv_planar(int nde,
			  double * yo,
			  int nt, double dt, double *t,
			  int nargs, struct potentialArg * potentialArgs,
			  double rtol, double atol,
			  double *result, int * err){
  //coefficients (verbatim from bovy_symplecticode.c)
  double c1= 0.392256805238780;
  double c8= c1;
  double c2= 0.510043411918458;
  double c7= c2;
  double c3= -0.471053385409758;
  double c6= c3;
  double c4= 0.687531682525198e-1;
  double c5= c4;
  double d1= 0.784513610477560;
  double d7= d1;
  double d2= 0.235573213359357;
  double d6= d2;
  double d3= -0.117767998417887e1;
  double d5= d3;
  double d4= 0.131518632068391e1; //d8=0
  int ndim= 2*(nde+1);
  double *qo= (double *) malloc ( ndim * sizeof(double) );
  double *po= (double *) malloc ( ndim * sizeof(double) );
  double *q12= (double *) malloc ( ndim * sizeof(double) );
  double *p12= (double *) malloc ( ndim * sizeof(double) );
  double *a= (double *) malloc ( ndim * sizeof(double) );
  int ii, jj, kk, bb;
  for (bb=0; bb <= nde; bb++) {
    for (kk=0; kk < 2; kk++) {
      *(qo+2*bb+kk)= *(yo+4*bb+kk);
      *(po+2*bb+kk)= *(yo+4*bb+2+kk);
    }
  }
  save_qp_aug_planar(nde,qo,po,result);
  result+= 2 * ndim;
  *err= 0;
  double init_dt= (*(t+1))-(*t);
  if ( dt == -9999.99 ) {
    dt= symplec6_estimate_step(&evalPlanarRectForce,2,qo,po,init_dt,t,nargs,
			       potentialArgs,rtol,atol);
  }
  long ndt= (long) (init_dt/dt);
  double to= *t;
#ifndef _WIN32
  struct sigaction action;
  memset(&action, 0, sizeof(struct sigaction));
  action.sa_handler= handle_sigint;
  sigaction(SIGINT,&action,NULL);
#else
  if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {}
#endif
  for (ii=0; ii < (nt-1); ii++){
    if ( interrupted ) {
      *err= -10;
      interrupted= 0;
#ifdef USING_COVERAGE
      __gcov_dump();
// LCOV_EXCL_START
      __gcov_reset();
#endif
      break;
// LCOV_EXCL_STOP
    }
    //drift for c1*dt
    leapq_aug_planar(ndim,qo,po,c1*dt,q12);
    to+= c1*dt;
    //steps ignoring q8/p8 when output is not wanted
    for (jj=0; jj < (ndt-1); jj++){
      //kick for d1*dt
      evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,po,d1*dt,a,p12);
      //drift for c2*dt
      leapq_aug_planar(ndim,q12,p12,c2*dt,qo);
      to+= c2*dt;
      //kick for d2*dt
      evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,p12,d2*dt,a,po);
      //drift for c3*dt
      leapq_aug_planar(ndim,qo,po,c3*dt,q12);
      to+= c3*dt;
      //kick for d3*dt
      evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,po,d3*dt,a,p12);
      //drift for c4*dt
      leapq_aug_planar(ndim,q12,p12,c4*dt,qo);
      //kick for d4*dt
      to+= c4*dt;
      evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,p12,d4*dt,a,po);
      //drift for c5*dt
      leapq_aug_planar(ndim,qo,po,c5*dt,q12);
      to+= c5*dt;
      //kick for d5*dt
      evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,po,d5*dt,a,p12);
      //drift for c6*dt
      leapq_aug_planar(ndim,q12,p12,c6*dt,qo);
      //kick for d6*dt
      to+= c6*dt;
      evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,p12,d6*dt,a,po);
      //drift for c7*dt
      leapq_aug_planar(ndim,qo,po,c7*dt,q12);
      to+= c7*dt;
      //kick for d7*dt
      evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
      leapp_aug_planar(ndim,po,d7*dt,a,p12);
      //drift for (c8+c1)*dt
      leapq_aug_planar(ndim,q12,p12,(c8+c1)*dt,qo);
      to+= (c8+c1)*dt;
      //reset
      for (kk=0; kk < ndim; kk++) {
	*(q12+kk)= *(qo+kk);
	*(po+kk)= *(p12+kk);
      }
    }
    //steps not ignoring q8/p8 when output is wanted
    //kick for d1*dt
    evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,po,d1*dt,a,p12);
    //drift for c2*dt
    leapq_aug_planar(ndim,q12,p12,c2*dt,qo);
    to+= c2*dt;
    //kick for d2*dt
    evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,p12,d2*dt,a,po);
    //drift for c3*dt
    leapq_aug_planar(ndim,qo,po,c3*dt,q12);
    to+= c3*dt;
    //kick for d3*dt
    evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,po,d3*dt,a,p12);
    //drift for c4*dt
    leapq_aug_planar(ndim,q12,p12,c4*dt,qo);
    to+= c4*dt;
    //kick for d4*dt
    evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,p12,d4*dt,a,po);
    //drift for c5*dt
    leapq_aug_planar(ndim,qo,po,c5*dt,q12);
    to+= c5*dt;
    //kick for d5*dt
    evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,po,d5*dt,a,p12);
    //drift for c6*dt
    leapq_aug_planar(ndim,q12,p12,c6*dt,qo);
    //kick for d6*dt
    to+= c6*dt;
    evalPlanarRectForce_dxdv(to,qo,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,p12,d6*dt,a,po);
    //drift for c7*dt
    leapq_aug_planar(ndim,qo,po,c7*dt,q12);
    to+= c7*dt;
    //kick for d7*dt
    evalPlanarRectForce_dxdv(to,q12,a,nargs,potentialArgs,nde);
    leapp_aug_planar(ndim,po,d7*dt,a,p12);
    //drift for c8*dt
    leapq_aug_planar(ndim,q12,p12,c8*dt,qo);
    to+= c8*dt;
    //p8=p7
    for (kk=0; kk < ndim; kk++) *(po+kk)= *(p12+kk);
    save_qp_aug_planar(nde,qo,po,result);
    result+= 2 * ndim;
  }
#ifndef _WIN32
  action.sa_handler= SIG_DFL;
  sigaction(SIGINT,&action,NULL);
#endif
  free(qo);
  free(po);
  free(q12);
  free(p12);
  free(a);
}
