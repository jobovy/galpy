/* C implementations of galpy potentials */
/*
  Function declarations
*/
//ZeroForce
double ZeroPlanarForce(double, double,double,int, double *);
double ZeroForce(double,double,double,double,int, double *);
//LogarithmicHaloPotential
double LogarithmicHaloPotentialEval(double ,double , double, double,
				    int , double *);
double LogarithmicHaloPotentialRforce(double ,double , double, double,
				      int , double *);
double LogarithmicHaloPotentialPlanarRforce(double ,double, double,
					    int , double *);
double LogarithmicHaloPotentialzforce(double,double,double,double,
				      int, double *);
double LogarithmicHaloPotentialPlanarR2deriv(double ,double, double,
					     int , double *);
//DehnenBarPotential
double DehnenBarPotentialRforce(double,double,double,int,double *);
double DehnenBarPotentialphiforce(double,double,double,int,double *);
double DehnenBarPotentialR2deriv(double,double,double,int,double *);
double DehnenBarPotentialphi2deriv(double,double,double,int,double *);
double DehnenBarPotentialRphideriv(double,double,double,int,double *);
//TransientLogSpiralPotential
double TransientLogSpiralPotentialRforce(double,double,double,int,double *);
double TransientLogSpiralPotentialphiforce(double,double,double,int,double *);
//SteadyLogSpiralPotential
double SteadyLogSpiralPotentialRforce(double,double,double,int,double *);
double SteadyLogSpiralPotentialphiforce(double,double,double,int,double *);
//EllipticalDiskPotential
double EllipticalDiskPotentialRforce(double,double,double,int,double *);
double EllipticalDiskPotentialphiforce(double,double,double,int,double *);
double EllipticalDiskPotentialR2deriv(double,double,double,int,double *);
double EllipticalDiskPotentialphi2deriv(double,double,double,int,double *);
double EllipticalDiskPotentialRphideriv(double,double,double,int,double *);
//Miyamoto-Nagai Potential
double MiyamotoNagaiPotentialEval(double ,double , double, double,
				  int , double *);
double MiyamotoNagaiPotentialRforce(double ,double , double, double,
				    int , double *);
double MiyamotoNagaiPotentialPlanarRforce(double ,double, double,
					  int , double *);
double MiyamotoNagaiPotentialzforce(double,double,double,double,
				    int, double *);
double MiyamotoNagaiPotentialPlanarR2deriv(double ,double, double,
					   int , double *);
//LopsidedDiskPotential
double LopsidedDiskPotentialRforce(double,double,double,int,double *);
double LopsidedDiskPotentialphiforce(double,double,double,int,double *);
double LopsidedDiskPotentialR2deriv(double,double,double,int,double *);
double LopsidedDiskPotentialphi2deriv(double,double,double,int,double *);
double LopsidedDiskPotentialRphideriv(double,double,double,int,double *);
//PowerSphericalPotential
double PowerSphericalPotentialEval(double ,double , double, double,
				   int , double *);
double PowerSphericalPotentialRforce(double ,double , double, double,
				     int , double *);
double PowerSphericalPotentialPlanarRforce(double ,double, double,
					   int , double *);
double PowerSphericalPotentialzforce(double,double,double,double,
				     int, double *);
double PowerSphericalPotentialPlanarR2deriv(double ,double, double,
					    int , double *);
//HernquistPotential
double HernquistPotentialEval(double ,double , double, double,
			      int , double *);
double HernquistPotentialRforce(double ,double , double, double,
				     int , double *);
double HernquistPotentialPlanarRforce(double ,double, double,
					   int , double *);
double HernquistPotentialzforce(double,double,double,double,
				     int, double *);
double HernquistPotentialPlanarR2deriv(double ,double, double,
					    int , double *);
//NFWPotential
double NFWPotentialEval(double ,double , double, double,
			int , double *);
double NFWPotentialRforce(double ,double , double, double,
				     int , double *);
double NFWPotentialPlanarRforce(double ,double, double,
					   int , double *);
double NFWPotentialzforce(double,double,double,double,
				     int, double *);
double NFWPotentialPlanarR2deriv(double ,double, double,
					    int , double *);
//JaffePotential
double JaffePotentialEval(double ,double , double, double,
			    int , double *);
double JaffePotentialRforce(double ,double , double, double,
				     int , double *);
double JaffePotentialPlanarRforce(double ,double, double,
					   int , double *);
double JaffePotentialzforce(double,double,double,double,
				     int, double *);
double JaffePotentialPlanarR2deriv(double ,double, double,
					    int , double *);
//DoubleExponentialDiskPotential
double DoubleExponentialDiskPotentialEval(double ,double , double, double,
					  int , double *);
double DoubleExponentialDiskPotentialRforce(double,double, double,double,
					    int, double *);
double DoubleExponentialDiskPotentialPlanarRforce(double,double,double,
						  int,double *);
double DoubleExponentialDiskPotentialzforce(double,double, double,double,
					    int, double *);
//FlattenedPowerPotential
double FlattenedPowerPotentialEval(double,double,double,double,
				   int, double *);
