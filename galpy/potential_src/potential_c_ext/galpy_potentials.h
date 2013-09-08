/* C implementations of galpy potentials */
/*
  Structure declarations
*/
#ifndef __GALPY_POTENTIALS_H__
#define __GALPY_POTENTIALS_H__
#include <interp_2d.h>
struct potentialArg{
  double (*potentialEval)(double R, double Z, double phi, double t,
			  struct potentialArg *);
  double (*Rforce)(double R, double Z, double phi, double t,
		   struct potentialArg *);
  double (*zforce)(double R, double Z, double phi, double t,
		   struct potentialArg *);
  double (*phiforce)(double R, double Z, double phi, double t,
		     struct potentialArg *);
  double (*planarRforce)(double R,double phi, double t,
			 struct potentialArg *);
  double (*planarphiforce)(double R,double phi, double t,
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
  int nargs;
  double * args;
  interp_2d * i2d;
  gsl_interp_accel * acc;
  interp_2d * i2drforce;
  gsl_interp_accel * accrforce;
  interp_2d * i2dzforce;
  gsl_interp_accel * acczforce;
};
/*
  Function declarations
*/
//ZeroForce
double ZeroPlanarForce(double,double,double,
		       struct potentialArg *);
double ZeroForce(double,double,double,double,
		 struct potentialArg *);
//LogarithmicHaloPotential
double LogarithmicHaloPotentialEval(double ,double , double, double,
				    struct potentialArg *);
double LogarithmicHaloPotentialRforce(double ,double , double, double,
				    struct potentialArg *);
double LogarithmicHaloPotentialPlanarRforce(double ,double, double,
				    struct potentialArg *);
double LogarithmicHaloPotentialzforce(double,double,double,double,
				    struct potentialArg *);
double LogarithmicHaloPotentialPlanarR2deriv(double ,double, double,
				    struct potentialArg *);
//DehnenBarPotential
double DehnenBarPotentialRforce(double,double,double,
				struct potentialArg *);
double DehnenBarPotentialphiforce(double,double,double,
		       struct potentialArg *);
double DehnenBarPotentialR2deriv(double,double,double,
		       struct potentialArg *);
double DehnenBarPotentialphi2deriv(double,double,double,
		       struct potentialArg *);
double DehnenBarPotentialRphideriv(double,double,double,
		       struct potentialArg *);
//TransientLogSpiralPotential
double TransientLogSpiralPotentialRforce(double,double,double,
		       struct potentialArg *);
double TransientLogSpiralPotentialphiforce(double,double,double,
		       struct potentialArg *);
//SteadyLogSpiralPotential
double SteadyLogSpiralPotentialRforce(double,double,double,
		       struct potentialArg *);
double SteadyLogSpiralPotentialphiforce(double,double,double,
		       struct potentialArg *);
//EllipticalDiskPotential
double EllipticalDiskPotentialRforce(double,double,double,
		       struct potentialArg *);
double EllipticalDiskPotentialphiforce(double,double,double,
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
//LopsidedDiskPotential
double LopsidedDiskPotentialRforce(double,double,double,
					   struct potentialArg *);
double LopsidedDiskPotentialphiforce(double,double,double,
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
//DoubleExponentialDiskPotential
double DoubleExponentialDiskPotentialEval(double ,double , double, double,
					  struct potentialArg *);
double DoubleExponentialDiskPotentialRforce(double,double, double,double,
					    struct potentialArg *);
double DoubleExponentialDiskPotentialPlanarRforce(double,double,double,
						  struct potentialArg *);
double DoubleExponentialDiskPotentialzforce(double,double, double,double,
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
#endif /* galpy_potentials.h */
