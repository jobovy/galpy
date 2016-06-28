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

inline void compute_dphiTilde(double r, double a, int N, int L, double * C, double * dC, double * dphiTilde){
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
           *( dphiTilde + l*N + n) = rterm1*rterm2* (*(C + l*N + n)) +
            rterm3*(*(dC + l*N + n));
        }
    }

}

inline void compute_d2phiTilde(double r, double a, int N, int L, double * C, double * dC,double * d2C, double * d2phiTilde){
    double xi;
    calculateXi(r, a, &xi);
    double rterm = 1./(r*r) / ((a + r)*(a + r)*(a + r));

    for (int l = 0; l < L; l++){
        if (l != 0){
            rterm *= (a*r)/((a + r)*(a + r));
            }
        for (int n = 0; n < N; n++){
           *( d2phiTilde + l*N + n) = rterm*(*(C + l*N + n)*(a*a * (l - 1)*l - 2*a*l*(l + 2)*r + (l+1)*(l+2)*r*r) +
            r*(a + r)*((*(dC + l*N + n))*(2*(a*l - (l + 1)*r)*(2*a)/((a + r)*(a + r)) - 4*r/((a + r)*(a + r))) +
             4*a*a*r/((a + r)*(a + r)*(a + r))*(*(d2C + l*N + n)))) ;
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


inline void compute_d2C(double xi, int N, int L, double * d2C_array){
    for (int l = 0; l < L; l++){
        *(d2C_array +l*N) = 0;
        if (N !=1)
            *(d2C_array +l*N + 1) = 0;
        if (N > 2)
            gsl_sf_gegenpoly_array(N - 3, 7./2 + 2*l, xi, d2C_array + l*N + 2);
        for (int n = 0; n<N; n++){
        *(d2C_array +l*N + n) *= 4*(2*l + 3./2)*(2*l + 5./2);
        }
    }

}

inline void compute_P(double x, int L, double * P_array, double *dP_array){
  for (int l = 0; l < L; l++){
        int shift = l*L + l;
        gsl_sf_legendre_Plm_deriv_array(L - 1, l, x, P_array + shift, dP_array + shift);
    }
}


inline void computePlanarderiv(double a, int N, int L, int M,
                        double *Acos, double *Asin,
                        double r, double theta, double phi,
                        double *Farray){
                                printf("In Function");
                                double xi;
                                calculateXi(r, a, &xi);
    
                                //Compute the gegenbauer polynomials and its derivative. 
                                double C[N*L];
                                double dC[N*L];
                                double d2C[N*L];

                                compute_C(xi, N, L, &C);
                                compute_dC(xi, N, L, &dC);
                                compute_d2C(xi, N, L, &d2C);
                                printf("Success C");
                                
                                //Compute phiTilde and its derivative
                                double phiTilde[L*N];
                                compute_phiTilde(r, a, N, L, &C, &phiTilde);
                                
                                double dphiTilde[L*N];
                                compute_dphiTilde(r, a, N, L, &C, &dC, &dphiTilde);
                                
                                double d2phiTilde[L*N];
                                compute_d2phiTilde(r, a, N, L, &C, &dC, &d2C, &d2phiTilde);
                                
                                
                                //Compute Associated Legendre Polynomials
                                double P[L*L];
                                double dP[L*L];
                                double d2P[L*L]; //TODO
                                
                                compute_P(cos(theta), L, &P, &dP);
                                
                                double F_r = 0;
                                double F_theta = 0;
                                double F_phi = 0;
                                double F_rr = 0;
                                double F_thetatheta = 0;
                                double F_phiphi = 0;
                                double F_rtheta = 0;
                                double F_rphi = 0;
                                double F_thetaphi = 0;
                                
                                for (int l = 0; l < L; l++){
                                    for (int m = 0; m<=l;m++){
                                        double mCos = cos(m*phi);
                                        double mSin = sin(m*phi);
                                        for (int n = 0;n < N; n++){
                                            
                                        
                                        double Acos_val = *(Acos +m + M*l + M*L*n);
                                        double Asin_val = *(Asin +m + M*l + M*L*n);
                                            F_r -= (Acos_val*mCos + Asin_val*mSin)*
                                                            P[m*L + l]*dphiTilde[l*N + n];
                                                            
                                            F_theta += (Acos_val*mCos + Asin_val*mSin)*
                                                            dP[m*L + l]*phiTilde[l*N + n]*(sin(theta));
                                                            
                                            F_phi -= m*(Asin_val*mCos - Acos_val*mSin)*
                                                            P[m*L + l]*phiTilde[l*N + n];
                                                            
                                            F_rr -= (Acos_val*mCos + Asin_val*mSin)*
                                                            P[m*L + l]*d2phiTilde[l*N + n];
                                            
                                            F_thetatheta -= (Acos_val*mCos + Asin_val*mSin)*
                                                            (dP[m*L + l]*phiTilde[l*N + n]*(-cos(theta)) + 
                                                            d2P[m*L + l]*phiTilde[l*N + n]*(sin(theta)*sin(theta)));
                                            //F_phiphi
                                            F_phiphi += m*m*(Acos_val*mCos + Asin_val*mSin)*
                                                            P[m*L + l]*phiTilde[l*N + n];
                                            
                                            //F_rtheta
                                            F_rtheta += (Acos_val*mCos + Asin_val*mSin)*
                                                            dP[m*L + l]*dphiTilde[l*N + n]*(sin(theta));
                                            
                                            //TODO F_rphi
                                            F_rphi -= m*(Asin_val*mCos - Acos_val*mSin)*
                                                            P[m*L + l]*dphiTilde[l*N + n];
                                            
                                            //F_thetaphi
                                            F_thetaphi -= m*(Asin_val*mCos - Acos_val*mSin)*
                                                            dP[m*L + l]*phiTilde[l*N + n]*(-sin(theta));
                                          
                                            
                                        }                                    
                                    }     
                            }
                              
                            *(Farray + 0) = F_r*sqrt(4*M_PI);
                            *(Farray + 1) = F_theta*sqrt(4*M_PI);
                            *(Farray + 2) = F_phi*sqrt(4*M_PI);
                            
                            *(Farray + 3) = F_rr*sqrt(4*M_PI);
                            *(Farray + 4) = F_thetatheta*sqrt(4*M_PI);
                            *(Farray + 5) = F_phiphi*sqrt(4*M_PI);
                            
                            *(Farray + 6) = F_rtheta*sqrt(4*M_PI);
                            *(Farray + 7) = F_rphi*sqrt(4*M_PI);
                            *(Farray + 8) = F_thetaphi*sqrt(4*M_PI);

                           
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
                       
double SCFPotentialEval(double R,double Z, double phi,
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
  double xi;
  calculateXi(r, a, &xi);
    
  //Compute the gegenbauer polynomials and its derivative. 
  double C[N*L];

  compute_C(xi, N, L, &C);
                            
  //Compute phiTilde and its derivative
  double phiTilde[L*N];
  compute_phiTilde(r, a, N, L, &C, &phiTilde);
  //Compute Associated Legendre Polynomials
  double P[L*L];
  double dP[L*L];
                                
  compute_P(cos(theta), L, &P, &dP);
                                
  double potential = 0;                              
                                
  for (int l = 0; l < L; l++){
      for (int m = 0; m<=l;m++){
          double mCos = cos(m*phi);
          double mSin = sin(m*phi);
          for (int n = 0;n < N; n++){
              double Acos_val = *(Acos +m + M*l + M*L*n);
              double Asin_val = *(Asin +m + M*l + M*L*n);
              potential += (Acos_val*mCos + Asin_val*mSin)*P[m*L + l]*phiTilde[l*N + n];
          }
      }
  }
  potential *= sqrt(4*M_PI);
  return potential;   
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

double SCFPotentialPlanarRforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  return SCFPotentialRforce(R,0, phi,t,potentialArgs);
  
}

double SCFPotentialPlanarphiforce(double R,double Z, double phi,
				double t,
				struct potentialArg * potentialArgs){
  return SCFPotentialphiforce(R,0, phi,t,potentialArgs);
}



double SCFPotentialPlanarR2deriv(double R, double Z, double phi,
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
  cyl_to_spher(R, 0, &r, &theta);
  
  double dr_dR = 1;
  double dtheta_dR = 0;
  double dphi_dR = 0;
  double d2r_d2R = 0;
  double d2theta_d2R = 0;
  double d2phi_d2R = 0;
  
  double Farray[9];
  printf("Calling Function");
 computePlanarderiv(a, N, L, M, Acos, Asin, r, theta, phi, &Farray);
 
    return dr_dR*dr_dR*Farray[3];
 
                                                        
   }


double SCFPotentialPlanarphi2deriv(double R, double Z, double phi,
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
  cyl_to_spher(R, 0, &r, &theta);
  
  double dr_dphi = 0;
  double dtheta_dphi = 0;
  double dphi_dphi = 1;
  double d2r_d2phi = 0;
  double d2theta_d2phi = 0;
  double d2phi_d2phi = 0;
  
  double Farray[9];
  
 computePlanarderiv(a, N, L, M, Acos, Asin, r, theta, phi, &Farray);
 
    return dphi_dphi*dphi_dphi*Farray[5];
 
                                                        
   }
   
double SCFPotentialPlanarRphideriv(double R, double Z, double phi,
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
  cyl_to_spher(R, 0, &r, &theta);
  
  double dr_dR = 1;
  double dtheta_dR = 0;
  double dphi_dR = 0;
  
  double dr_dZ= 0;
  double dtheta_dZ = -R;
  double dphi_dZ = 0;

  
  double dr_dphi = 0;
  double dtheta_dphi = 0;
  double dphi_dphi = 1;
  
  double d2r_dRphi = 0;
  double d2theta_dRphi = 0;
  double d2phi_dRphi = 0;

  
  
  double F[9];
  
  computePlanarderiv(a, N, L, M, Acos, Asin, r, theta, phi, &F);
 
    return F[7]*(dr_dR*dphi_dphi);
        
                                                        
   }
