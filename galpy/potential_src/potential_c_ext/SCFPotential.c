#include <math.h>
#include <galpy_potentials.h>
#include <stdio.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_legendre.h>

//SCF Disk potential
//4 arguments: amp, Acos, Asin, a


inline void cyl_to_spher(double R, double Z, double phi,double *r, double *theta){
  *r = sqrt(R*R + Z*Z);
  *theta = atan2(R, Z);
}

inline void calculateXi(double r, double a, double *xi){
*xi = (r - a)/(r + a);
}

inline void compute_phiTilde(double r, double a, int N, int L, double * phiTilde){
    double xi;
    calculateXi(r, a, &xi);
    double rterms = -1./(r + a);


    for (int l = 0; l < L; l++){
        gsl_sf_gegenpoly_array(N - 1, 3./2 + 2*l, xi, &phiTilde[l*N]);

        if (l != 0)
            rterms *= r*a/((a + r)*(a + r));
        for (int n = 0; n < N; n++){
            phiTilde[l*N + n] *= rterms;
        }
    }

}

inline void compute_dphiTilde(double r, double a, int N, int L, double * phiTilde){
    double xi;
    calculateXi(r, a, &xi);
    double rterm1 = -1./((a + r)*(a + r));
    double rterm2 = -1.;
    double rterm3 = -(1. - xi)*(1. - xi) / (2*a*(a + r));

    double dC[N*L];
    dGegenPoly(xi, N, L, &dC);

    for (int l = 0; l < L; l++){
        double C[N];
        gsl_sf_gegenpoly_array(N - 1, 3./2 + 2*l, xi, &C);

        if (l != 0){
            rterm1 *= (a*r)/((a + r)*(a + r));
            rterm2 = (l*(a + r)/r - (2*l + 1));
            rterm3 *= (a*r)/(a + r);
            }
        for (int n = 0; n < N; n++){
            phiTilde[l*N + n] = rterm1*rterm2*C[n] +
            rterm3*dC[l*N + n];
        }
    }

}

inline void dGegenPoly(double xi, int N, int L, double *dC){
    for (int l = 0; l < L; l++){
        dC[l*N] = 0;
        gsl_sf_gegenpoly_array(N - 1, 5./2 + 2*l, xi, &dC[l*N + 1]);
    }

}


inline double computeForce(double a, int N, int L, int M, 
                        double *Acos, double *Asin,
                        double dr_dx, double dtheta_dx, double dphi_dx,
                        double r, double theta, double phi){

                           
                                
                                double phiTilde[L*N];
                                compute_phiTilde(r, a, N, L,phiTilde);
                                
                                double dphiTilde[L*N];
                                compute_dphiTilde(r, a, N, L, dphiTilde);
                                double dPhi_dr[N][L][M];
                                double dPhi_dtheta[N][L][M];
                                double dPhi_dphi[N][L][M];
                                for (int l = 0; l < L; l++){
                                    for (int m = 0; m<=l;m++){
                                        double P[L];
                                        double dP[L];
                                        gsl_sf_legendre_Plm_deriv_array(l,m,cos(theta), P, dP);
                                        for (int n = 0;n < N; n++){
                                            dPhi_dr[n][l][m] = (*(Acos +n + l*N + m*L*N)*cos(m*phi) +
                                                            *(Asin +n + l*N + m*L*N)*sin(m*phi))*
                                                            P[l]*dphiTilde[l*N + n];
                                                            
                                            dPhi_dtheta[n][l][m] = (*(Acos +n + l*N + m*L*N)*cos(m*phi) +
                                                            *(Asin +n + l*N + m*L*N)*sin(m*phi))*
                                                            dP[l]*phiTilde[l*N + n]*(-sin(theta));
                                            dPhi_dphi[n][l][m] = m*(*(Asin +n + l*N + m*L*N)*cos(m*phi) -
                                                            *(Acos +n + l*N + m*L*N)*sin(m*phi))*
                                                            P[l]*phiTilde[l*N + n];

                                        }                                    
                                    }                                
                                
                            }
                            double force = 0;
                            for (int l = 0; l < L; l++){
                                    for (int m = 0; m<=l;m++){
                                        for (int n = 0;n < N; n++){
                                            force -= (dPhi_dr[n][l][m]*dr_dx + 
                                            dPhi_dtheta[n][l][m]*dtheta_dx + 
                                            dPhi_dphi[n][l][m]*dphi_dx)*
                                            sqrt(4*M_PI);
                                        }
                                    }
                             }
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
  cyl_to_spher(R, Z, phi,&r, &theta);

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
  cyl_to_spher(R, Z, phi,&r, &theta);
  
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
  cyl_to_spher(R, Z, phi,&r, &theta);
  
  double dr_dphi = 0; 
  double dtheta_dphi = 0; 
  double dphi_dphi = 1;
  
 return computeForce(a, N, L, M, Acos, Asin, dr_dphi,dtheta_dphi, dphi_dphi, r, theta, phi);
}
