#include <math.h>
#include <galpy_potentials.h>
//Kuzmin Disk potential
//2 arguments: amp, a
double KuzminDiskPotentialEval(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Potential
  return - amp * pow(R*R+pow(a+fabs(z),2.),-0.5);
}
double KuzminDiskPotentialRforce(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate Rforce
  return - amp * R * pow(R*R+pow(a+fabs(z),2.),-1.5);
}

double KuzminDiskPotentialzforce(double R,double z,double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a= *args;
  //Calculate zforce
  double zsign= (z > 0 ) - (z < 0); //Gets the sign of z
  return -zsign* amp * pow(R*R+pow(a+fabs(z),2.),-1.5) * (a + fabs(z));
}

