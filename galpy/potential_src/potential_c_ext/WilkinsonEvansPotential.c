#include <math.h>
#include <stdio.h>
#include "galpy_potentials.h"
//Wilkinson Evans potential
//4 arguments: Mh, ah
double WilkinsonEvansPotentialEval(double R,double z, double phi,
				  double t,
				  struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double Mh= *args++;
  double ah= *args;
  //Calculate potential
  
  return -Mh/ah*log((sqrt(R*R+z*z+ah*ah)+ah)/sqrt(R*R+z*z)); 

//  return
//  return - amp * pow(R*R+pow(a+pow(z*z+b*b,0.5),2.),-0.5);
}

double WilkinsonEvansPotentialRforce(double R,double z, double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double Mh= *args++;
  double ah= *args;
  //Calculate Rforce
  return Mh*sqrt(z*z+R*R)*(R/(sqrt(z*z+R*R)*sqrt(z*z+R*R+ah*ah))-R*pow(z*z+R*R, ((-3.0)/2.0))*(sqrt(z*z+R*R+ah*ah)+ah))/(ah*(sqrt(z*z+R*R+ah*ah)+ah));

}

double WilkinsonEvansPotentialzforce(double R,double z,double phi,
				    double t,
				    struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double Mh= *args++;
  double ah= *args;

  return Mh*sqrt(z*z+R*R)*(z/(sqrt(z*z+R*R)*sqrt(z*z+R*R+ah*ah))-z*pow(z*z+R*R, ((-3.0)/2.0))*(sqrt(z*z+R*R+ah*ah)+ah))/(ah*(sqrt(z*z+R*R+ah*ah)+ah));

}
