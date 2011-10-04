/* C implementations of galpy potentials */
/*
  Function declarations
*/
//ZeroForce
double zeroPlanarForce(double, double,int, double);
double zeroForce(double,int, double);
//LogarithmicHaloPotential
double LogarithmicHaloPotentialRforce(double ,double , double ,
				      int , double *, 
				      double * );
double LogarithmicHaloPotentialPlanarRforce(double ,double ,
					    int , double *, 
					    double * );
double LogarithmicHaloPotentialzforce(double,double,double,
				      int, double, 
				      double *);
