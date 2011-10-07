/* C implementations of galpy potentials */
/*
  Function declarations
*/
//ZeroForce
double ZeroPlanarForce(double, double,double,int, double *);
double ZeroForce(double,double,double,double,int, double *);
//LogarithmicHaloPotential
double LogarithmicHaloPotentialRforce(double ,double , double, double,
				      int , double *);
double LogarithmicHaloPotentialPlanarRforce(double ,double, double,
					    int , double *);
double LogarithmicHaloPotentialzforce(double,double,double,double,
				      int, double *);
//DehnenBarPotential
double DehnenBarPotentialRforce(double,double,double,int,double *);
double DehnenBarPotentialphiforce(double,double,double,int,double *);
//TransientLogSpiralPotential
double TransientLogSpiralPotentialRforce(double,double,double,int,double *);
double TransientLogSpiralPotentialphiforce(double,double,double,int,double *);
