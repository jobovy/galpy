/* C implementations of galpy potentials */
/*
  Structure declarations
*/
#ifndef __GALPY_POTENTIALS_H__
#define __GALPY_POTENTIALS_H__
#ifdef __cplusplus
extern "C" {
#endif
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
  gsl_interp_accel * accx;
  gsl_interp_accel * accy;
  interp_2d * i2drforce;
  gsl_interp_accel * accxrforce;
  gsl_interp_accel * accyrforce;
  interp_2d * i2dzforce;
  gsl_interp_accel * accxzforce;
  gsl_interp_accel * accyzforce;
};
/*
  Function declarations
*/
//Utility
void cyl_to_rect(double,double,double *,double *);
//Potential and force evaluation
double evaluatePotentials(double,double,int, struct potentialArg *);
double calcRforce(double,double,double,double,int,struct potentialArg *);
double calczforce(double,double,double,double,int,struct potentialArg *);
double calcPhiforce(double, double,double, double, 
			int, struct potentialArg *);
double calcR2deriv(double, double, double,double, 
			 int, struct potentialArg *);
double calcphi2deriv(double, double, double,double, 
			   int, struct potentialArg *);
double calcRphideriv(double, double, double,double, 
			   int, struct potentialArg *);
double calcPlanarRforce(double, double, double, 
			int, struct potentialArg *);
double calcPlanarphiforce(double, double, double, 
			int, struct potentialArg *);
double calcPlanarR2deriv(double, double, double, 
			 int, struct potentialArg *);
double calcPlanarphi2deriv(double, double, double, 
			   int, struct potentialArg *);
double calcPlanarRphideriv(double, double, double, 
			   int, struct potentialArg *);
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
double EllipticalDiskSmooth(double,double, double);

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
//TriaxialHernquistPotential
double TriaxialHernquistPotentialEval(double,double,double,double,
				struct potentialArg *);
double TriaxialHernquistPotentialRforce(double,double,double,double,
					struct potentialArg *);
double TriaxialHernquistPotentialPlanarRforce(double,double,double,
					      struct potentialArg *);
double TriaxialHernquistPotentialphiforce(double,double,double,double,
					  struct potentialArg *);
double TriaxialHernquistPotentialPlanarphiforce(double,double,double,
						struct potentialArg *);
double TriaxialHernquistPotentialzforce(double,double,double,double,
					struct potentialArg *);
//TriaxialNFWPotential
double TriaxialNFWPotentialEval(double,double,double,double,
				struct potentialArg *);
double TriaxialNFWPotentialRforce(double,double,double,double,
				  struct potentialArg *);
double TriaxialNFWPotentialPlanarRforce(double,double,double,
					struct potentialArg *);
double TriaxialNFWPotentialphiforce(double,double,double,double,
				    struct potentialArg *);
double TriaxialNFWPotentialPlanarphiforce(double,double,double,
					  struct potentialArg *);
double TriaxialNFWPotentialzforce(double,double,double,double,
				  struct potentialArg *);
//TriaxialJaffePotential
double TriaxialJaffePotentialEval(double,double,double,double,
				struct potentialArg *);
double TriaxialJaffePotentialRforce(double,double,double,double,
				    struct potentialArg *);
double TriaxialJaffePotentialPlanarRforce(double,double,double,
					  struct potentialArg *);
double TriaxialJaffePotentialphiforce(double,double,double,double,
				      struct potentialArg *);
double TriaxialJaffePotentialPlanarphiforce(double,double,double,
					    struct potentialArg *);
double TriaxialJaffePotentialzforce(double,double,double,double,
				    struct potentialArg *);					      
//SCFPotential
double SCFPotentialEval(double,double,double,double,
				     struct potentialArg *);
double SCFPotentialRforce(double,double,double,double,
                        struct potentialArg *);
double SCFPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double SCFPotentialphiforce(double,double,double,double,
				        struct potentialArg *);
				        
double SCFPotentialPlanarRforce(double,double,double,
                        struct potentialArg *);
double SCFPotentialPlanarphiforce(double,double,double,
				        struct potentialArg *);
double SCFPotentialPlanarR2deriv(double,double,double,
				        struct potentialArg *);
double SCFPotentialPlanarphi2deriv(double,double,double,
				        struct potentialArg *);
double SCFPotentialPlanarRphideriv(double,double,double,
				        struct potentialArg *);
//SoftenedNeedleBarPotential
double SoftenedNeedleBarPotentialEval(double,double,double,double,
				      struct potentialArg *);
double SoftenedNeedleBarPotentialRforce(double,double,double,double,
					struct potentialArg *);
double SoftenedNeedleBarPotentialzforce(double,double,double,double,
				        struct potentialArg *);
double SoftenedNeedleBarPotentialphiforce(double,double,double,double,
					  struct potentialArg *);
double SoftenedNeedleBarPotentialPlanarRforce(double,double,double,
					      struct potentialArg *);
double SoftenedNeedleBarPotentialPlanarphiforce(double,double,double,
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
#ifdef __cplusplus
}
#endif
// Wilkinson Evans potential 
double WilkinsonEvansPotentialEval(double,double, double,double,             
                                  struct potentialArg * );
double WilkinsonEvansPotentialRforce(double,double, double,double,                       
                                    struct potentialArg * );
double WilkinsonEvansPotentialzforce(double,double,double,double,
                                    struct potentialArg * );   

#endif /* galpy_potentials.h */
