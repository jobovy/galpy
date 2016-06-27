#include <math.h>
#include <galpy_potentials.h>
#include <stdio.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_legendre.h>

//SCF Disk potential
//4 arguments: amp, Acos, Asin, a


inline void cyl_to_spher(double R, double Z,double *r, double *theta){
  *r = sqrt(R*R + Z*Z);
  *theta = atan2(R, Z);
}

inline void calculateXi(double r, double a, double *xi){
*xi = (r - a)/(r + a);
}

inline void compute_phiTilde(double r, double a, int N, int L, double* C, double * phiTilde){
    double xi;
    calculateXi(r, a, &xi);
    double rterms = -1./(r + a);

    for (int l = 0; l < L; l++){

        if (l != 0)
            rterms *= r*a/((a + r)*(a + r));

        for (int n = 0; n < N; n++){
            *(phiTilde + l*N + n)  = rterms*(*(C + n + l*N));
        }
    }

}

inline void compute_dphiTilde(double r, double a, int N, int L, double * C, double * dC, double * phiTilde){
    double xi;
    calculateXi(r, a, &xi);
    double rterm1 = -1./((a + r)*(a + r));
    double rterm2 = -1.;
    double rterm3 = -(1. - xi)*(1. - xi) / (2*a*(a + r));

    for (int l = 0; l < L; l++){
        if (l != 0){
            double rterm = (a*r)/((a + r)*(a + r)); 
            rterm1 *= rterm;
            rterm2 = (l*(a + r)/r - (2*l + 1));
            rterm3 *= rterm;
            }
        for (int n = 0; n < N; n++){
           *( phiTilde + l*N + n) = rterm1*rterm2* (*(C + l*N + n)) +
            rterm3*(*(dC + l*N + n));
        }
    }

}

inline void compute_C(double xi, int N, int L, double * C_array){
  for (int l = 0; l < L; l++){
        gsl_sf_gegenpoly_array(N - 1, 3./2 + 2*l, xi, C_array + l*N);
    }
}


inline void compute_dC(double xi, int N, int L, double * dC_array){
    for (int l = 0; l < L; l++){
        *(dC_array +l*N) = 0;
        if (N != 1)
            gsl_sf_gegenpoly_array(N - 2, 5./2 + 2*l, xi, dC_array + l*N + 1);
        for (int n = 0; n<N; n++){
        *(dC_array +l*N + n) *= 2*(2*l + 3./2);
        }
    }

}

inline void compute_P(double x, int L, double * P_array, double *dP_array){
  for (int l = 0; l < L; l++){
        gsl_sf_legendre_Plm_deriv_array(L - 1, l, x, P_array + l*L + l, dP_array + l*L + l);
    }
}

inline double computeForce(double a, int N, int L, int M, 
                        double *Acos, double *Asin,
                        double dr_dx, double dtheta_dx, double dphi_dx,
                        double r, double theta, double phi){

                                
                                
                                double xi;
                                calculateXi(r, a, &xi);
    
                                //Compute the gegenbauer polynomials and its derivative. 
                                double C[N*L];
                                double dC[N*L];

                                compute_C(xi, N, L, &C);
                                compute_dC(xi, N, L, &dC);
                                
                                //Compute phiTilde and its derivative
                                double phiTilde[L*N];
                                compute_phiTilde(r, a, N, L, &C, &phiTilde);
                                
                                double dphiTilde[L*N];
                                compute_dphiTilde(r, a, N, L, &C, &dC, &dphiTilde);
                                
                                //Compute Associated Legendre Polynomials
                                double P[L*L];
                                double dP[L*L];
                                
                                compute_P(cos(theta), L, &P, &dP);
                                
                                double F_r = 0;
                                double F_theta = 0;
                                double F_phi = 0;
                                
                                for (int l = 0; l < L; l++){
                                    for (int m = 0; m<=l;m++){
                                        double mCos = cos(m*phi);
                                        double mSin = sin(m*phi);
                                        for (int n = 0;n < N; n++){
                                            
                                        
                                        double Acos_val = *(Acos +m + M*l + M*L*n);
                                        double Asin_val = *(Asin +m + M*l + M*L*n);
                                            F_r -= (Acos_val*mCos + Asin_val*mSin)*
                                                            P[m*L + l]*dphiTilde[l*N + n];
                                                            
                                            F_theta -= (Acos_val*mCos + Asin_val*mSin)*
                                                            dP[m*L + l]*phiTilde[l*N + n]*(-sin(theta));
                                                            
                                            F_phi -= m*(Asin_val*mCos - Acos_val*mSin)*
                                                            P[m*L + l]*phiTilde[l*N + n];
                                          
                                            
                                        }                                    
                                    }                                
                                
                            }
                              
                            F_r *= dr_dx;
                            F_theta *= dtheta_dx;
                            F_phi *= dphi_dx;
                            double force = (F_r + F_theta + F_phi)*sqrt(4*M_PI);
                             
                             return force;
                           
                        }

double SCFPotentialRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double* Acos = args;
  double* Asin = args + N*L*M;

  //convert R,Z to r, theta
  double r;
  double theta;
  cyl_to_spher(R, Z,&r, &theta);

  // The derivatives 
  double dr_dR = R/r; 
  double dtheta_dR = Z/(r*r);
  double dphi_dR = 0;
  
 return computeForce(a, N, L, M, Acos, Asin, dr_dR,dtheta_dR, dphi_dR, r, theta, phi);
}

double SCFPotentialzforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double* Acos = args;
  double* Asin = args + N*L*M;
  double r;
  double theta;
  cyl_to_spher(R, Z,&r, &theta);
  
  double dr_dz = Z/r; 
  double dtheta_dz = -R/(r*r); 
  double dphi_dz = 0;
  
 return computeForce(a, N, L, M, Acos, Asin, dr_dz,dtheta_dz, dphi_dz, r, theta, phi);
}

double SCFPotentialphiforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  double * args= potentialArgs->args;
  //Get args
  double a = *args++;
  int N = *args++;
  int L = *args++;
  int M = *args++;
  double* Acos = args;
  double* Asin = args + N*L*M;
  double r;
  double theta;
  cyl_to_spher(R, Z, &r, &theta);
  
  double dr_dphi = 0; 
  double dtheta_dphi = 0; 
  double dphi_dphi = 1;
 return computeForce(a, N, L, M, Acos, Asin, dr_dphi,dtheta_dphi, dphi_dphi, r, theta, phi);
}
