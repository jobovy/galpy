/* C implementations of galpy potentials */
/*
  Function declarations
*/
//ZeroForce
double ZeroPlanarForce(double, double,int, double *);
double ZeroForce(double,double,double,int, double *);
//LogarithmicHaloPotential
double LogarithmicHaloPotentialRforce(double ,double , double ,
				      int , double *);
double LogarithmicHaloPotentialPlanarRforce(double ,double ,
					    int , double *);
double LogarithmicHaloPotentialzforce(double,double,double,
				      int, double *);
