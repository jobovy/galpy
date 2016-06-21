#include <math.h>
#include <galpy_potentials.h>
#include <stdio.h>

//SCF Disk potential
//4 arguments: amp, Acos, Asin, a


double SCFPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double Acos[N][L][M];
  double Asin[N][L][M];
  
  // Get Acos
  for (int n=0; n <N;n= n+1){
     for (int l=0; l <L;l= l+1){
      for (int m=0; m <M;m= m+1){
      Acos[n][l][m] = *args++;
  }
  } 
  }
  // Get Asin
  for (int n=0; n <N;n= n+1){
     for (int l=0; l <L;l= l+1){
      for (int m=0; m <M;m= m+1){
      Asin[n][l][m] = *args++;
  }
  } 
  }

 return 1.;
}

double SCFPotentialzforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double Acos[N][L][M];
  double Asin[N][L][M];
  
  // Get Acos
  for (int n=0; n <N;n= n+1){
     for (int l=0; l <L;l= l+1){
      for (int m=0; m <M;m= m+1){
      Acos[n][l][m] = *args++;
  }
  } 
  }
  // Get Asin
  for (int n=0; n <N;n= n+1){
     for (int l=0; l <L;l= l+1){
      for (int m=0; m <M;m= m+1){
      Asin[n][l][m] = *args++;
  }
  } 
  }
  
  //Calculate zforce
  return 1.;
}

double SCFPotentialphiforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double amp= *args++;
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double Acos[N][L][M];
  double Asin[N][L][M];
  
  // Get Acos
  for (int n=0; n <N;n= n+1){
     for (int l=0; l <L;l= l+1){
      for (int m=0; m <M;m= m+1){
      Acos[n][l][m] = *args++;
  }
  } 
  }
  // Get Asin
  for (int n=0; n <N;n= n+1){
     for (int l=0; l <L;l= l+1){
      for (int m=0; m <M;m= m+1){
      Asin[n][l][m] = *args++;
  }
  } 
  }
  //Calculate phiforce
  
  return 1.;
}
