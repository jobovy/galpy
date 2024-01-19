/* C implementations of galpy potentials */
/*
  Structure declarations
*/
#ifndef __GALPY_POTENTIALS_H__
#define __GALPY_POTENTIALS_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <interp_2d.h>
#ifndef M_1_PI
#define M_1_PI 0.31830988618379069122
#endif
typedef double (**tfuncs_type_arr)(double t); // array of functions of time
struct potentialArg{
  double (*potentialEval)(double R, double Z, double phi, double t,
			  struct potentialArg *);
  double (*Rforce)(double R, double Z, double phi, double t,
		   struct potentialArg *);
  double (*zforce)(double R, double Z, double phi, double t,
		   struct potentialArg *);
  double (*phitorque)(double R, double Z, double phi, double t,
		     struct potentialArg *);
  double (*planarRforce)(double R,double phi, double t,
			 struct potentialArg *);
  double (*planarphitorque)(double R,double phi, double t,
			   struct potentialArg *);
  double (*R2deriv)(double R,double Z,double phi, double t,
		    struct potentialArg *);
  double (*phi2deriv)(double R,double Z,double phi, double t,
		      struct potentialArg *);
  double (*Rphideriv)(double R,double Z,double phi, double t,
		      struct potentialArg *);
  double (*planarR2deriv)(double R,double phi, double t,
			  struct potentialArg *);
  double (*planarphi2deriv)(double R,double phi, double t,
			    struct potentialArg *);
  double (*planarRphideriv)(double R,double phi, double t,
			    struct potentialArg *);
  double (*linearForce)(double x, double t,
			 struct potentialArg *);
  double (*dens)(double R, double Z, double phi, double t,
		 struct potentialArg *);
  // For forces that require velocity input (e.g., dynam fric)
  bool requiresVelocity;
  double (*RforceVelocity)(double R, double Z, double phi, double t,
			    struct potentialArg *,double,double,double);
  double (*zforceVelocity)(double R, double Z, double phi, double t,
			   struct potentialArg *,double,double,double);
  double (*phitorqueVelocity)(double R, double Z, double phi, double t,
			     struct potentialArg *,double,double,double);
  double (*planarRforceVelocity)(double R,double phi, double t,
			 struct potentialArg *,double,double);
  double (*planarphitorqueVelocity)(double R,double phi, double t,
			   struct potentialArg *,double,double);

  int nargs;
  double * args;
  // To allow 1D interpolation for an arbitrary number of splines
  int nspline1d;
  gsl_interp_accel ** acc1d;
  gsl_spline ** spline1d;
  // 2D interpolation
  interp_2d * i2d;
  gsl_interp_accel * accx;
  gsl_interp_accel * accy;
  interp_2d * i2drforce;
  gsl_interp_accel * accxrforce;
  gsl_interp_accel * accyrforce;
  interp_2d * i2dzforce;
  gsl_interp_accel * accxzforce;
  gsl_interp_accel * accyzforce;
  // To allow an arbitrary number of functions of time
  int ntfuncs;
  tfuncs_type_arr tfuncs; // see typedef above
  // Wrappers
  int nwrapped;
  struct potentialArg * wrappedPotentialArg;
  // For EllipsoidalPotentials
  double (*psi)(double m,double * args);
  double (*mdens)(double m,double * args);
  double (*mdensDeriv)(double m,double * args);
  // For SphericalPotentials
  double (*revaluate)(double r,double t,struct potentialArg *);
  double (*rforce)(double r,double t,struct potentialArg *);
  double (*r2deriv)(double r,double t,struct potentialArg *);
  double (*rdens)(double r,double t,struct potentialArg *);
};
/*
  Function declarations
*/
//Dealing with potentialArg
void init_potentialArgs(int,struct potentialArg *);
void free_potentialArgs(int,struct potentialArg *);
//Potential and force evaluation
double evaluatePotentials(double,double,int, struct potentialArg *);
// Hack to allow optional velocity for dissipative forces
// https://stackoverflow.com/a/52610204/10195320
// Reason to use ##__VA_ARGS__ is that when no optional velocity is supplied,
// there is a comma left in the argument list and ## absorbs that for gcc/icc
// MSVC supposedly does this automatically for regular __VA_ARGS__, at least
// for now, but I can't get this to work locally or on AppVeyor
// https://docs.microsoft.com/en-us/cpp/preprocessor/preprocessor-experimental-overview?view=vs-2019#comma-elision-in-variadic-macros
// Therefore, I use an alternative where all arguments are variadic, such
// that there should always be many. Final subtlety is that we have to define
// the EXPAND macro to expand the __VA_ARGS__ into multiple arguments,
// otherwise it's treated as a single argument in the next function call
// (e.g., CALCRFORCE would get R = __VA_ARGS = R,Z,phi,...)
#ifdef _MSC_VER
#define EXPAND(x) x
#define calcRforce(...)   EXPAND(CALCRFORCE(__VA_ARGS__,0.,0.,0.))
#define calczforce(...)   EXPAND(CALCZFORCE(__VA_ARGS__,0.,0.,0.))
#define calcphitorque(...) EXPAND(CALCPHITORQUE(__VA_ARGS__,0.,0.,0.))
#else
#define calcRforce(R,Z,phi,t,nargs,potentialArgs,...) CALCRFORCE(R,Z,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.,0.)
#define calczforce(R,Z,phi,t,nargs,potentialArgs,...) CALCZFORCE(R,Z,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.,0.)
#define calcphitorque(R,Z,phi,t,nargs,potentialArgs,...) CALCPHITORQUE(R,Z,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.,0.)
#endif
#define CALCRFORCE(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ,...) calcRforce(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ)
#define CALCZFORCE(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ,...) calczforce(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ)
#define CALCPHITORQUE(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ,...) calcphitorque(R,Z,phi,t,nargs,potentialArgs,vR,vT,vZ)
double (calcRforce)(double,double,double,double,int,struct potentialArg *,
		    double,double,double);
double (calczforce)(double,double,double,double,int,struct potentialArg *,
		      double,double,double);
double (calcphitorque)(double, double,double, double,
		      int, struct potentialArg *,
		      double,double,double);
// end hack
double calcR2deriv(double, double, double,double,
			 int, struct potentialArg *);
double calcphi2deriv(double, double, double,double,
			   int, struct potentialArg *);
double calcRphideriv(double, double, double,double,
			   int, struct potentialArg *);
// Same hack as for Rforce etc. above to allow optional velocity for dissipative forces
#ifdef _MSC_VER
#define calcPlanarRforce(...)   EXPAND(CALCPLANARRFORCE(__VA_ARGS__,0.,0.))
#define calcPlanarphitorque(...)   EXPAND(CALCPLANARPHITORQUE(__VA_ARGS__,0.,0.))
#else
#define calcPlanarRforce(R,phi,t,nargs,potentialArgs,...) CALCPLANARRFORCE(R,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.)
#define calcPlanarphitorque(R,phi,t,nargs,potentialArgs,...) CALCPLANARPHITORQUE(R,phi,t,nargs,potentialArgs,##__VA_ARGS__,0.,0.)
#endif
#define CALCPLANARRFORCE(R,phi,t,nargs,potentialArgs,vR,vT,...) calcPlanarRforce(R,phi,t,nargs,potentialArgs,vR,vT)
#define CALCPLANARPHITORQUE(R,phi,t,nargs,potentialArgs,vR,vT,...) calcPlanarphitorque(R,phi,t,nargs,potentialArgs,vR,vT)
double (calcPlanarRforce)(double, double, double,
			int, struct potentialArg *,double,double);
double (calcPlanarphitorque)(double, double, double,
			int, struct potentialArg *,double,double);
// end hack
double calcPlanarR2deriv(double, double, double,
			 int, struct potentialArg *);
double calcPlanarphi2deriv(double, double, double,
			   int, struct potentialArg *);
double calcPlanarRphideriv(double, double, double,
			   int, struct potentialArg *);
double calcLinearForce(double, double, int, struct potentialArg *);
double calcDensity(double, double, double,double, int, struct potentialArg *);
void rotate(double *, double *, double *, double *);
void rotate_force(double *, double *, double *,double *);
//ZeroForce
double ZeroPlanarForce(double,double,double,
		       struct potentialArg *);
double ZeroForce(double,double,double,double,
		 struct potentialArg *);
//verticalPotential
double verticalPotentialLinearForce(double,double,struct potentialArg *);
//LogarithmicHaloPotential
double LogarithmicHaloPotentialEval(double ,double , double, double,
				    struct potentialArg *);
double LogarithmicHaloPotentialRforce(double ,double , double, double,
				    struct potentialArg *);
double LogarithmicHaloPotentialPlanarRforce(double ,double, double,
				    struct potentialArg *);
double LogarithmicHaloPotentialzforce(double,double,double,double,
				    struct potentialArg *);
double LogarithmicHaloPotentialphitorque(double,double,double,double,
					struct potentialArg *);
double LogarithmicHaloPotentialPlanarphitorque(double,double,double,
					      struct potentialArg *);
double LogarithmicHaloPotentialPlanarR2deriv(double ,double, double,
				    struct potentialArg *);
double LogarithmicHaloPotentialPlanarphi2deriv(double ,double, double,
					       struct potentialArg *);
double LogarithmicHaloPotentialPlanarRphideriv(double ,double, double,
					       struct potentialArg *);
double LogarithmicHaloPotentialDens(double ,double , double, double,
				    struct potentialArg *);
//DehnenBarPotential
double DehnenBarPotentialRforce(double,double,double,double,
				struct potentialArg *);
double DehnenBarPotentialphitorque(double,double,double,double,
				  struct potentialArg *);
double DehnenBarPotentialzforce(double,double,double,double,
				struct potentialArg *);
double DehnenBarPotentialPlanarRforce(double,double,double,
				      struct potentialArg *);
double DehnenBarPotentialPlanarphitorque(double,double,double,
					struct potentialArg *);
double DehnenBarPotentialPlanarR2deriv(double,double,double,
				       struct potentialArg *);
double DehnenBarPotentialPlanarphi2deriv(double,double,double,
					 struct potentialArg *);
double DehnenBarPotentialPlanarRphideriv(double,double,double,
					 struct potentialArg *);
//TransientLogSpiralPotential
double TransientLogSpiralPotentialRforce(double,double,double,
		       struct potentialArg *);
double TransientLogSpiralPotentialphitorque(double,double,double,
		       struct potentialArg *);
//SteadyLogSpiralPotential
double SteadyLogSpiralPotentialRforce(double,double,double,
		       struct potentialArg *);
double SteadyLogSpiralPotentialphitorque(double,double,double,
		       struct potentialArg *);
//EllipticalDiskPotential
double EllipticalDiskPotentialRforce(double,double,double,
		       struct potentialArg *);
double EllipticalDiskPotentialphitorque(double,double,double,
		       struct potentialArg *);
double EllipticalDiskPotentialR2deriv(double,double,double,
		       struct potentialArg *);
double EllipticalDiskPotentialphi2deriv(double,double,double,
		       struct potentialArg *);
double EllipticalDiskPotentialRphideriv(double,double,double,
		       struct potentialArg *);
//Miyamoto-Nagai Potential
double MiyamotoNagaiPotentialEval(double ,double , double, double,
				  struct potentialArg *);
double MiyamotoNagaiPotentialRforce(double ,double , double, double,
				    struct potentialArg *);
double MiyamotoNagaiPotentialPlanarRforce(double ,double, double,
					  struct potentialArg *);
double MiyamotoNagaiPotentialzforce(double,double,double,double,
				    struct potentialArg *);
double MiyamotoNagaiPotentialPlanarR2deriv(double ,double, double,
					   struct potentialArg *);
double MiyamotoNagaiPotentialDens(double ,double , double, double,
				  struct potentialArg *);
//LopsidedDiskPotential
double LopsidedDiskPotentialRforce(double,double,double,
					   struct potentialArg *);
double LopsidedDiskPotentialphitorque(double,double,double,
					   struct potentialArg *);
double LopsidedDiskPotentialR2deriv(double,double,double,
					   struct potentialArg *);
double LopsidedDiskPotentialphi2deriv(double,double,double,
					   struct potentialArg *);
double LopsidedDiskPotentialRphideriv(double,double,double,
					   struct potentialArg *);
//PowerSphericalPotential
double PowerSphericalPotentialEval(double ,double , double, double,
				   struct potentialArg *);
double PowerSphericalPotentialRforce(double ,double , double, double,
					   struct potentialArg *);
double PowerSphericalPotentialPlanarRforce(double ,double, double,
					   struct potentialArg *);
double PowerSphericalPotentialzforce(double,double,double,double,
				     struct potentialArg *);
double PowerSphericalPotentialPlanarR2deriv(double ,double, double,
					    struct potentialArg *);
double PowerSphericalPotentialDens(double ,double , double, double,
				   struct potentialArg *);
//HernquistPotential
double HernquistPotentialEval(double ,double , double, double,
			      struct potentialArg *);
double HernquistPotentialRforce(double ,double , double, double,
				struct potentialArg *);
double HernquistPotentialPlanarRforce(double ,double, double,
				      struct potentialArg *);
double HernquistPotentialzforce(double,double,double,double,
				struct potentialArg *);
double HernquistPotentialPlanarR2deriv(double ,double, double,
				       struct potentialArg *);
double HernquistPotentialDens(double ,double , double, double,
			      struct potentialArg *);
//NFWPotential
double NFWPotentialEval(double ,double , double, double,
			struct potentialArg *);
double NFWPotentialRforce(double ,double , double, double,
			  struct potentialArg *);
double NFWPotentialPlanarRforce(double ,double, double,
				struct potentialArg *);
double NFWPotentialzforce(double,double,double,double,
			  struct potentialArg *);
double NFWPotentialPlanarR2deriv(double ,double, double,
				 struct potentialArg *);
double NFWPotentialDens(double ,double , double, double,
			 struct potentialArg *);
//JaffePotential
double JaffePotentialEval(double ,double , double, double,
			  struct potentialArg *);
double JaffePotentialRforce(double ,double , double, double,
			    struct potentialArg *);
double JaffePotentialPlanarRforce(double ,double, double,
				  struct potentialArg *);
double JaffePotentialzforce(double,double,double,double,
			    struct potentialArg *);
double JaffePotentialPlanarR2deriv(double ,double, double,
				   struct potentialArg *);
double JaffePotentialDens(double ,double , double, double,
			  struct potentialArg *);
//DoubleExponentialDiskPotential
double DoubleExponentialDiskPotentialEval(double ,double , double, double,
					  struct potentialArg *);
double DoubleExponentialDiskPotentialRforce(double,double, double,double,
					    struct potentialArg *);
double DoubleExponentialDiskPotentialPlanarRforce(double,double,double,
						  struct potentialArg *);
double DoubleExponentialDiskPotentialzforce(double,double, double,double,
					    struct potentialArg *);
double DoubleExponentialDiskPotentialDens(double ,double , double, double,
					  struct potentialArg *);
//FlattenedPowerPotential
double FlattenedPowerPotentialEval(double,double,double,double,
				   struct potentialArg *);
double FlattenedPowerPotentialRforce(double,double,double,double,
				     struct potentialArg *);
double FlattenedPowerPotentialPlanarRforce(double,double,double,
					   struct potentialArg *);
double FlattenedPowerPotentialzforce(double,double,double,double,
				     struct potentialArg *);
double FlattenedPowerPotentialPlanarR2deriv(double,double,double,
					    struct potentialArg *);
double FlattenedPowerPotentialDens(double,double,double,double,
				   struct potentialArg *);
//interpRZPotential
double interpRZPotentialEval(double ,double , double, double,
			     struct potentialArg *);
double interpRZPotentialRforce(double ,double , double, double,
			       struct potentialArg *);
double interpRZPotentialzforce(double ,double , double, double,
			       struct potentialArg *);
//IsochronePotential
double IsochronePotentialEval(double ,double , double, double,
			      struct potentialArg *);
double IsochronePotentialRforce(double ,double , double, double,
				struct potentialArg *);
double IsochronePotentialPlanarRforce(double ,double, double,
				      struct potentialArg *);
double IsochronePotentialzforce(double,double,double,double,
				struct potentialArg *);
double IsochronePotentialPlanarR2deriv(double ,double, double,
				       struct potentialArg *);
double IsochronePotentialDens(double ,double , double, double,
			      struct potentialArg *);
//PowerSphericalPotentialwCutoff
double PowerSphericalPotentialwCutoffEval(double ,double , double, double,
					  struct potentialArg *);
double PowerSphericalPotentialwCutoffRforce(double ,double , double, double,
					    struct potentialArg *);
double PowerSphericalPotentialwCutoffPlanarRforce(double ,double, double,
						  struct potentialArg *);
double PowerSphericalPotentialwCutoffzforce(double,double,double,double,
					    struct potentialArg *);
double PowerSphericalPotentialwCutoffPlanarR2deriv(double ,double, double,
						   struct potentialArg *);
double PowerSphericalPotentialwCutoffDens(double ,double , double, double,
					  struct potentialArg *);
//KuzminKutuzovStaeckelPotential
double KuzminKutuzovStaeckelPotentialEval(double,double,double,double,
                        struct potentialArg *);
double KuzminKutuzovStaeckelPotentialRforce(double,double,double,double,
                        struct potentialArg *);
double KuzminKutuzovStaeckelPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double KuzminKutuzovStaeckelPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double KuzminKutuzovStaeckelPotentialPlanarR2deriv(double,double,double,
					    struct potentialArg *);

//KuzminDiskPotential
double KuzminDiskPotentialEval(double,double,double,double,
                        struct potentialArg *);
double KuzminDiskPotentialRforce(double,double,double,double,
                        struct potentialArg *);
double KuzminDiskPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double KuzminDiskPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double KuzminDiskPotentialPlanarR2deriv(double,double,double,
					    struct potentialArg *);
//PlummerPotential
double PlummerPotentialEval(double,double,double,double,
                        struct potentialArg *);
double PlummerPotentialRforce(double,double,double,double,
                        struct potentialArg *);
double PlummerPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double PlummerPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double PlummerPotentialPlanarR2deriv(double,double,double,
					    struct potentialArg *);
double PlummerPotentialDens(double,double,double,double,
			    struct potentialArg *);
//PseudoIsothermalPotential
double PseudoIsothermalPotentialEval(double,double,double,double,
				     struct potentialArg *);
double PseudoIsothermalPotentialRforce(double,double,double,double,
				       struct potentialArg *);
double PseudoIsothermalPotentialPlanarRforce(double,double,double,
					     struct potentialArg *);
double PseudoIsothermalPotentialzforce(double,double,double,double,
				       struct potentialArg *);
double PseudoIsothermalPotentialPlanarR2deriv(double,double,double,
					      struct potentialArg *);
double PseudoIsothermalPotentialDens(double,double,double,double,
				     struct potentialArg *);
//BurkertPotential
double BurkertPotentialEval(double,double,double,double,
				     struct potentialArg *);
double BurkertPotentialRforce(double,double,double,double,
				       struct potentialArg *);
double BurkertPotentialPlanarRforce(double,double,double,
					     struct potentialArg *);
double BurkertPotentialzforce(double,double,double,double,
				       struct potentialArg *);
double BurkertPotentialPlanarR2deriv(double,double,double,
					      struct potentialArg *);
double BurkertPotentialDens(double,double,double,double,
			    struct potentialArg *);
//EllipsoidalPotential
double EllipsoidalPotentialEval(double,double,double,double,
				     struct potentialArg *);
double EllipsoidalPotentialRforce(double,double,double,double,
				  struct potentialArg *);
double EllipsoidalPotentialPlanarRforce(double,double,double,
					struct potentialArg *);
double EllipsoidalPotentialphitorque(double,double,double,double,
				    struct potentialArg *);
double EllipsoidalPotentialPlanarphitorque(double,double,double,
					  struct potentialArg *);
double EllipsoidalPotentialzforce(double,double,double,double,
				  struct potentialArg *);
double EllipsoidalPotentialDens(double,double,double,double,
				struct potentialArg *);
//TriaxialHernquistPotential: uses EllipsoidalPotential, only need psi, dens, densDeriv
double TriaxialHernquistPotentialpsi(double,double *);
double TriaxialHernquistPotentialmdens(double,double *);
double TriaxialHernquistPotentialmdensDeriv(double,double *);
//TriaxialJaffePotential: uses EllipsoidalPotential, only need psi, dens, densDeriv
double TriaxialJaffePotentialpsi(double,double *);
double TriaxialJaffePotentialmdens(double,double *);
double TriaxialJaffePotentialmdensDeriv(double,double *);
//TriaxialNFWPotential: uses EllipsoidalPotential, only need psi, dens, densDeriv
double TriaxialNFWPotentialpsi(double,double *);
double TriaxialNFWPotentialmdens(double,double *);
double TriaxialNFWPotentialmdensDeriv(double,double *);
//SCFPotential
double SCFPotentialEval(double,double,double,double,
				     struct potentialArg *);
double SCFPotentialRforce(double,double,double,double,
                        struct potentialArg *);
double SCFPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double SCFPotentialphitorque(double,double,double,double,
				        struct potentialArg *);

double SCFPotentialPlanarRforce(double,double,double,
                        struct potentialArg *);
double SCFPotentialPlanarphitorque(double,double,double,
				        struct potentialArg *);
double SCFPotentialPlanarR2deriv(double,double,double,
				        struct potentialArg *);
double SCFPotentialPlanarphi2deriv(double,double,double,
				        struct potentialArg *);
double SCFPotentialPlanarRphideriv(double,double,double,
				        struct potentialArg *);
double SCFPotentialDens(double,double,double,double,
			struct potentialArg *);
//SoftenedNeedleBarPotential
double SoftenedNeedleBarPotentialEval(double,double,double,double,
				      struct potentialArg *);
double SoftenedNeedleBarPotentialRforce(double,double,double,double,
					struct potentialArg *);
double SoftenedNeedleBarPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double SoftenedNeedleBarPotentialphitorque(double,double,double,double,
					  struct potentialArg *);
double SoftenedNeedleBarPotentialPlanarRforce(double,double,double,
					      struct potentialArg *);
double SoftenedNeedleBarPotentialPlanarphitorque(double,double,double,
					  struct potentialArg *);
//DiskSCFPotential
double DiskSCFPotentialEval(double,double,double,double,
				      struct potentialArg *);
double DiskSCFPotentialRforce(double,double,double,double,
					struct potentialArg *);
double DiskSCFPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double DiskSCFPotentialPlanarRforce(double,double,double,
					      struct potentialArg *);
double DiskSCFPotentialDens(double,double,double,double,
			    struct potentialArg *);

// SpiralArmsPotential
double SpiralArmsPotentialEval(double, double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialRforce(double, double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialzforce(double, double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialphitorque(double, double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialR2deriv(double R, double z, double phi, double t,
                            struct potentialArg* potentialArgs);
double SpiralArmsPotentialz2deriv(double R, double z, double phi, double t,
                            struct potentialArg* potentialArgs);
double SpiralArmsPotentialphi2deriv(double R, double z, double phi, double t,
                            struct potentialArg* potentialArgs);
double SpiralArmsPotentialRzderiv(double R, double z, double phi, double t,
                            struct potentialArg* potentialArgs);
double SpiralArmsPotentialRphideriv(double R, double z, double phi, double t,
                            struct potentialArg* potentialArgs);
double SpiralArmsPotentialPlanarRforce(double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialPlanarphitorque(double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialPlanarR2deriv(double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialPlanarphi2deriv(double, double, double,
                            struct potentialArg*);
double SpiralArmsPotentialPlanarRphideriv(double, double, double,
                            struct potentialArg*);
//CosmphiDiskPotential
double CosmphiDiskPotentialRforce(double,double,double,
					   struct potentialArg *);
double CosmphiDiskPotentialphitorque(double,double,double,
					   struct potentialArg *);
double CosmphiDiskPotentialR2deriv(double,double,double,
					   struct potentialArg *);
double CosmphiDiskPotentialphi2deriv(double,double,double,
					   struct potentialArg *);
double CosmphiDiskPotentialRphideriv(double,double,double,
					   struct potentialArg *);
//HenonHeilesPotential
double HenonHeilesPotentialRforce(double,double,double,
				  struct potentialArg *);
double HenonHeilesPotentialphitorque(double,double,double,
				    struct potentialArg *);
double HenonHeilesPotentialR2deriv(double,double,double,
				   struct potentialArg *);
double HenonHeilesPotentialphi2deriv(double,double,double,
				     struct potentialArg *);
double HenonHeilesPotentialRphideriv(double,double,double,
				     struct potentialArg *);
//PerfectEllipsoid: uses EllipsoidalPotential, only need psi, dens, densDeriv
double PerfectEllipsoidPotentialpsi(double,double *);
double PerfectEllipsoidPotentialmdens(double,double *);
double PerfectEllipsoidPotentialmdensDeriv(double,double *);

//KGPotential
double KGPotentialLinearForce(double,double,struct potentialArg *);

//IsothermalDiskPotential
double IsothermalDiskPotentialLinearForce(double,double,struct potentialArg *);

//DehnenSphericalPotential
double DehnenSphericalPotentialEval(double ,double , double, double,
			      struct potentialArg *);
double DehnenSphericalPotentialRforce(double ,double , double, double,
				struct potentialArg *);
double DehnenSphericalPotentialPlanarRforce(double ,double, double,
				      struct potentialArg *);
double DehnenSphericalPotentialzforce(double,double,double,double,
				struct potentialArg *);
double DehnenSphericalPotentialPlanarR2deriv(double ,double, double,
				       struct potentialArg *);
double DehnenSphericalPotentialDens(double ,double , double, double,
			      struct potentialArg *);
//DehnenCoreSphericalPotential
double DehnenCoreSphericalPotentialEval(double ,double , double, double,
			      struct potentialArg *);
double DehnenCoreSphericalPotentialRforce(double ,double , double, double,
				struct potentialArg *);
double DehnenCoreSphericalPotentialPlanarRforce(double ,double, double,
				      struct potentialArg *);
double DehnenCoreSphericalPotentialzforce(double,double,double,double,
				struct potentialArg *);
double DehnenCoreSphericalPotentialPlanarR2deriv(double ,double, double,
				       struct potentialArg *);
double DehnenCoreSphericalPotentialDens(double ,double , double, double,
					struct potentialArg *);

//HomogeneousSpherePotential
double HomogeneousSpherePotentialEval(double ,double , double, double,
				      struct potentialArg *);
double HomogeneousSpherePotentialRforce(double ,double , double, double,
					struct potentialArg *);
double HomogeneousSpherePotentialPlanarRforce(double ,double, double,
					      struct potentialArg *);
double HomogeneousSpherePotentialzforce(double,double,double,double,
					struct potentialArg *);
double HomogeneousSpherePotentialPlanarR2deriv(double ,double, double,
					       struct potentialArg *);
double HomogeneousSpherePotentialDens(double ,double , double, double,
				      struct potentialArg *);
//SphericalPotential
double SphericalPotentialEval(double,double,double,double,
			      struct potentialArg *);
double SphericalPotentialRforce(double,double,double,double,
				struct potentialArg *);
double SphericalPotentialPlanarRforce(double,double,double,
				      struct potentialArg *);
double SphericalPotentialzforce(double,double,double,double,
				struct potentialArg *);
double SphericalPotentialPlanarR2deriv(double ,double, double,
				       struct potentialArg *);
double SphericalPotentialDens(double,double,double,double,
			      struct potentialArg *);
//interpSphericalPotential: uses SphericalPotential, only need revaluate, rforce, r2deriv
double interpSphericalPotentialrevaluate(double,double,struct potentialArg *);
double interpSphericalPotentialrforce(double,double,struct potentialArg *);
double interpSphericalPotentialr2deriv(double,double,struct potentialArg *);
double interpSphericalPotentialrdens(double,double,struct potentialArg *);

//TriaxialGaussian: uses EllipsoidalPotential, only need psi, dens, densDeriv
double TriaxialGaussianPotentialpsi(double,double *);
double TriaxialGaussianPotentialmdens(double,double *);
double TriaxialGaussianPotentialmdensDeriv(double,double *);
//PowerTriaxial: uses EllipsoidalPotential, only need psi, dens, densDeriv
double PowerTriaxialPotentialpsi(double,double *);
double PowerTriaxialPotentialmdens(double,double *);
double PowerTriaxialPotentialmdensDeriv(double,double *);
//NonInertialFrameForce, takes vR,vT,vZ
double NonInertialFrameForceRforce(double,double,double,double,
						 		   struct potentialArg *,
						 		   double,double,double);
double NonInertialFrameForcePlanarRforce(double,double,double,
						 		         struct potentialArg *,
						 		         double,double);
double NonInertialFrameForcephitorque(double,double,double,double,
						   			 struct potentialArg *,
						   			 double,double,double);
double NonInertialFrameForcePlanarphitorque(double,double,double,
						   			        struct potentialArg *,
						   			        double,double);
double NonInertialFrameForcezforce(double,double,double,double,
						 		   struct potentialArg *,
						 		   double,double,double);

//////////////////////////////// WRAPPERS /////////////////////////////////////
//DehnenSmoothWrapperPotential
double DehnenSmoothWrapperPotentialEval(double,double,double,double,
				      struct potentialArg *);
double DehnenSmoothWrapperPotentialRforce(double,double,double,double,
					struct potentialArg *);
double DehnenSmoothWrapperPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double DehnenSmoothWrapperPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double DehnenSmoothWrapperPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double DehnenSmoothWrapperPotentialPlanarphitorque(double,double,double,
						  struct potentialArg *);
double DehnenSmoothWrapperPotentialPlanarR2deriv(double,double,double,
						 struct potentialArg *);
double DehnenSmoothWrapperPotentialPlanarphi2deriv(double,double,double,
						   struct potentialArg *);
double DehnenSmoothWrapperPotentialPlanarRphideriv(double,double,double,
						   struct potentialArg *);
//SolidBodyRotationWrapperPotential
double SolidBodyRotationWrapperPotentialRforce(double,double,double,double,
					struct potentialArg *);
double SolidBodyRotationWrapperPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double SolidBodyRotationWrapperPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double SolidBodyRotationWrapperPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double SolidBodyRotationWrapperPotentialPlanarphitorque(double,double,double,
						  struct potentialArg *);
double SolidBodyRotationWrapperPotentialPlanarR2deriv(double,double,double,
						 struct potentialArg *);
double SolidBodyRotationWrapperPotentialPlanarphi2deriv(double,double,double,
						   struct potentialArg *);
double SolidBodyRotationWrapperPotentialPlanarRphideriv(double,double,double,
						   struct potentialArg *);
//CorotatingRotationWrapperPotential
double CorotatingRotationWrapperPotentialRforce(double,double,double,double,
					struct potentialArg *);
double CorotatingRotationWrapperPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double CorotatingRotationWrapperPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double CorotatingRotationWrapperPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double CorotatingRotationWrapperPotentialPlanarphitorque(double,double,double,
						  struct potentialArg *);
double CorotatingRotationWrapperPotentialPlanarR2deriv(double,double,double,
						 struct potentialArg *);
double CorotatingRotationWrapperPotentialPlanarphi2deriv(double,double,double,
						   struct potentialArg *);
double CorotatingRotationWrapperPotentialPlanarRphideriv(double,double,double,
						   struct potentialArg *);
//GaussianAmplitudeWrapperPotential
double GaussianAmplitudeWrapperPotentialEval(double,double,double,double,
				      struct potentialArg *);
double GaussianAmplitudeWrapperPotentialRforce(double,double,double,double,
					struct potentialArg *);
double GaussianAmplitudeWrapperPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double GaussianAmplitudeWrapperPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double GaussianAmplitudeWrapperPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double GaussianAmplitudeWrapperPotentialPlanarphitorque(double,double,double,
						  struct potentialArg *);
double GaussianAmplitudeWrapperPotentialPlanarR2deriv(double,double,double,
						 struct potentialArg *);
double GaussianAmplitudeWrapperPotentialPlanarphi2deriv(double,double,double,
						   struct potentialArg *);
double GaussianAmplitudeWrapperPotentialPlanarRphideriv(double,double,double,
						   struct potentialArg *);
//MovingObjectPotential
double MovingObjectPotentialRforce(double,double,double,double,
					struct potentialArg *);
double MovingObjectPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double MovingObjectPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double MovingObjectPotentialPlanarRforce(double,double,double,
					struct potentialArg *);
double MovingObjectPotentialPlanarphitorque(double,double,double,
					    struct potentialArg *);
//RotateAndTiltWrapperPotential
double RotateAndTiltWrapperPotentialRforce(double,double,double,double,
					struct potentialArg *);
double RotateAndTiltWrapperPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double RotateAndTiltWrapperPotentialzforce(double,double,double,double,
				        struct potentialArg *);
//ChandrasekharDynamicalFrictionForce, takes vR,vT,vZ
double ChandrasekharDynamicalFrictionForceRforce(double,double,double,double,
						 struct potentialArg *,
						 double,double,double);
double ChandrasekharDynamicalFrictionForcephitorque(double,double,double,double,
						   struct potentialArg *,
						   double,double,double);
double ChandrasekharDynamicalFrictionForcezforce(double,double,double,double,
						 struct potentialArg *,
						 double,double,double);
//TimeDependentAmplitudeWrapperPotential
double TimeDependentAmplitudeWrapperPotentialEval(double,double,double,double,
				      struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialRforce(double,double,double,double,
					struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialphitorque(double,double,double,double,
					    struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialPlanarphitorque(double,double,double,
						  struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialPlanarR2deriv(double,double,double,
						 struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialPlanarphi2deriv(double,double,double,
						   struct potentialArg *);
double TimeDependentAmplitudeWrapperPotentialPlanarRphideriv(double,double,double,
						   struct potentialArg *);
//KuzminLikeWrapperPotential
double KuzminLikeWrapperPotentialEval(double,double,double,double,
				      struct potentialArg *);
double KuzminLikeWrapperPotentialRforce(double,double,double,double,
					struct potentialArg *);
double KuzminLikeWrapperPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double KuzminLikeWrapperPotentialPlanarRforce(double,double,double,
						struct potentialArg *);
double KuzminLikeWrapperPotentialPlanarR2deriv(double,double,double,
						struct potentialArg *);

#ifdef __cplusplus
}
#endif
#endif /* galpy_potentials.h */
