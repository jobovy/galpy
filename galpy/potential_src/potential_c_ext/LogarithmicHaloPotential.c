//LogarithmicHaloPotential
//2 arguments: q and c2
double LogarithmicHaloPotentialRforce(double R,double Z, double phi,
				      int nargs, double *args, 
				      double * result){
  //Get args
  double q= *args;
  args++;
  double c= *args;
  //Calculate Rforce
  double zq= z/q;
  *result= -R/(R*R+zq*zq+c);
}
double LogarithmicHaloPotentialzforce(double R,double z,double phi,
				      int nargs, double *args, 
				      double * result){
  //Get args
  double q= *args;
  args++;
  double c= *args;
  //Calculate zforce
  double zq= z/q;
  *result= -z/q/q/(R*R+zq*zq+c);
}
