#include <math.h>
#include <galpy_potentials.h>
#include <stdio.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_legendre.h>

#ifndef GSL_MAJOR_VERSION
#define GSL_MAJOR_VERSION 1
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//SCF Disk potential
//4 arguments: amp, Acos, Asin, a

const int FORCE =1;
const int DERIV =2;

//Useful Functions

//Converts from cylindrical coordinates to spherical
static inline void cyl_to_spher(double R, double Z,double *r, double *theta)
{
    *r = sqrt(R*R + Z*Z);
    *theta = atan2(R, Z);
}

//Integer power
double power(double x, int i)
{
    if (i==0)
        return 1;
    return x*power(x,i - 1);
}

//Calculates xi
static inline void calculateXi(double r, double a, double *xi)
{
    *xi = (r - a)/(r + a);
}


//Potentials, forces, and derivative functions
//LCOV_EXCL_START
double computePhi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double phiTilde, int m)
{
    return (Acos_val*mCos + Asin_val*mSin)*P*phiTilde;
}
//LCOV_EXCL_STOP
double computeAxiPhi(double Acos_val, double P, double phiTilde)
{
    return Acos_val*P*phiTilde;
}

double computeF_r(double Acos_val, double Asin_val, double mCos, double mSin, double P, double dphiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*P*dphiTilde;
}
double computeAxiF_r(double Acos_val, double P, double dphiTilde)
{
    return -Acos_val*P*dphiTilde;
}

double computeF_theta(double Acos_val, double Asin_val, double mCos, double mSin, double dP, double phiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*dP*phiTilde;
}
double computeAxiF_theta(double Acos_val, double dP, double phiTilde)
{
    return -Acos_val*dP*phiTilde;
}

double computeF_phi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double phiTilde, int m)
{
    return m*(Acos_val*mSin - Asin_val*mCos)*P*phiTilde;
}
double computeAxiF_phi(double Acos_val, double P, double phiTilde)
{
    return 0.;
}

double computeF_rr(double Acos_val, double Asin_val, double mCos, double mSin, double P, double d2phiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*P*d2phiTilde;
}
double computeAxiF_rr(double Acos_val, double P, double d2phiTilde)
{
    return -Acos_val*P*d2phiTilde;
}

double computeF_rphi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double dphiTilde, int m)
{
    return m*(Acos_val*mSin - Asin_val*mCos)*P*dphiTilde;
}
double computeAxiF_rphi(double Acos_val, double P, double d2phiTilde)
{
    return 0.;
}

double computeF_phiphi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double phiTilde, int m)
{
    return m*m*(Acos_val*mCos + Asin_val*mSin)*P*phiTilde;
}
double computeAxiF_phiphi(double Acos_val, double P, double d2phiTilde)
{
    return 0.;
}

//Calculates the Gegenbauer polynomials
void compute_C(double xi, int N, int L, double * C_array)
{
    int l;
    for (l = 0; l < L; l++)
    {
        gsl_sf_gegenpoly_array(N - 1, 3./2 + 2*l, xi, C_array + l*N);
    }
}

//Calculates the derivative of the Gegenbauer polynomials
void compute_dC(double xi, int N, int L, double * dC_array)
{
    int n,l;
    for (l = 0; l < L; l++)
    {
        *(dC_array +l*N) = 0;
        if (N != 1)
            gsl_sf_gegenpoly_array(N - 2, 5./2 + 2*l, xi, dC_array + l*N + 1);
        for (n = 0; n<N; n++)
        {
            *(dC_array +l*N + n) *= 2*(2*l + 3./2);
        }
    }

}

//Calculates the second derivative of the Gegenbauer polynomials
void compute_d2C(double xi, int N, int L, double * d2C_array)
{
    int n,l;
    for (l = 0; l < L; l++)
    {
        *(d2C_array +l*N) = 0;
        if (N >1)
            *(d2C_array +l*N + 1) = 0;
        if (N > 2)
            gsl_sf_gegenpoly_array(N - 3, 7./2 + 2*l, xi, d2C_array + l*N + 2);
        for (n = 0; n<N; n++)
        {
            *(d2C_array +l*N + n) *= 4*(2*l + 3./2)*(2*l + 5./2);
        }
    }

}

//Compute phi_Tilde
void compute_phiTilde(double r, double a, int N, int L, double* C, double * phiTilde)
{
    double xi;
    calculateXi(r, a, &xi);
    double rterms = -1./(r + a);
    int n,l;
    for (l = 0; l < L; l++)
    {

        if (l != 0)
            rterms *= r*a/((a + r)*(a + r));

        for (n = 0; n < N; n++)
        {
            *(phiTilde + l*N + n)  = rterms*(*(C + n + l*N));
        }
    }

}

//Computes the derivative of phiTilde with respect to r
void compute_dphiTilde(double r, double a, int N, int L, double * C, double * dC, double * dphiTilde)
{
    double xi;
    calculateXi(r, a, &xi);
    double rterm = 1./(r*power(a + r, 3));
    int n,l;
    for (l = 0; l < L; l++)
    {
        if (l != 0)
        {
            rterm *= (a*r)/power(a + r, 2) ;
        }
        for (n = 0; n < N; n++)
        {
            *( dphiTilde + l*N + n) = rterm *(((2*l + 1)*r*(a + r) - l*power(a + r,2))*(*(C + l*N + n)) -
                                              2*a*r*(*(dC + l*N + n)));

        }
    }
}

//Computes the second derivative of phiTilde with respect to r
void compute_d2phiTilde(double r, double a, int N, int L, double * C, double * dC,double * d2C, double * d2phiTilde)
{
    double xi;
    calculateXi(r, a, &xi);
    double rterm = 1./(r*r) / power(a + r,5);
    int n,l;
    for (l = 0; l < L; l++)
    {


        if (l != 0)
        {
            rterm *= (a*r)/power(a + r, 2);
        }
        for (n = 0; n < N; n++)
        {

            double C_val = *(C + l*N + n);
            double dC_val = *(dC + l*N + n);
            double d2C_val = *(d2C + l*N + n);
            *( d2phiTilde + l*N + n) = rterm*(C_val*(l*(1 - l)*power(a + r, 4) - (4*l*l + 6*l + 2.)*r*r*power(a + r,2) +
                                              l*(4*l + 2)*r*power(a + r,3))
                                              + a*r*((4*r*r + 4*a*r + (8*l + 4)*r*(a + r)-
                                                      4*l*power(a + r,2))*dC_val - 4*a*r*d2C_val));

        }
    }
}


//Computes the associated Legendre polynomials
void compute_P(double x, int L, int M, double * P_array)
{
    if (M == 1){
        gsl_sf_legendre_Pl_array (L - 1, x, P_array);
    } else {
        #if GSL_MAJOR_VERSION == 2
            gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_NONE,L - 1, x, -1, P_array);
        #else
            int m;
            for (m = 0; m < M; m++)
            {
                gsl_sf_legendre_Plm_array(L - 1, m, x, P_array);
                P_array += L - m;
            }
        #endif
    }
    
      
    
}

//Computes the associated Legendre polynomials and its derivative
void compute_P_dP(double x, int L, int M, double * P_array, double *dP_array)
{
    if (M == 1){
        gsl_sf_legendre_Pl_deriv_array (L - 1, x, P_array, dP_array);
    } else {
        #if GSL_MAJOR_VERSION == 2
            gsl_sf_legendre_deriv_array_e(GSL_SF_LEGENDRE_NONE, L - 1, x, -1,P_array, dP_array);
        
        #else
            int m;
            for (m = 0; m < M; m++)
            {
                gsl_sf_legendre_Plm_deriv_array(L - 1, m, x, P_array, dP_array);
                P_array += L - m;
                dP_array += L - m;
            }
        #endif
    }
}





typedef struct equations equations;
struct equations
{
    double ((**Eq)(double, double, double, double, double, double, int));
    double *(*phiTilde);
    double *(*P);
    double *Constant;
};

typedef struct axi_equations axi_equations;
struct axi_equations
{
    double ((**Eq)(double, double, double));
    double *(*phiTilde);
    double *(*P);
    double *Constant;
};

//Compute axi symmetric potentials.
void compute(double a, int N, int L, int M,
	     double r, double theta, double phi,
	     double *Acos, int eq_size,
	     axi_equations e,
	     double *F)
{
    int i,n,l;
    for (i = 0; i < eq_size; i++)
    {
        *(F + i) =0; //Initialize each F
    }

    for (l = 0; l < L; l++)
    {

        for (n = 0; n < N; n++)
        {



            double Acos_val = *(Acos + M*l + M*L*n);
            for (i = 0; i < eq_size; i++)
            {
                double (*Eq)(double, double, double) = *(e.Eq + i);
                double *P = *(e.P + i);
                double *phiTilde = *(e.phiTilde + i);
                *(F + i) += (*Eq)(Acos_val, P[l], phiTilde[l*N + n]);
            }


        }
    }

    //Multiply F by constants
    for (i = 0; i < eq_size; i++)
    {
        double constant = *(e.Constant + i);
        *(F + i) *= constant*sqrt(4*M_PI);
    }

}

//Compute Non Axi symmetric Potentials
void computeNonAxi(double a, int N, int L, int M,
		   double r, double theta, double phi,
		   double *Acos, double *Asin, int eq_size,
		   equations e,
		   double *F)
{
    int i,n,l,m;
    for (i = 0; i < eq_size; i++)
    {
        *(F + i) =0; //Initialize each F
    }

    int v1counter=0;
    int v2counter = 0;
    int Pindex = 0;
    for (l = 0; l < L; l++)
    {
        v1counter = 0;
        for (m = 0; m<=l; m++)
        {
            #if GSL_MAJOR_VERSION == 2
                Pindex = v2counter;
            #else
                Pindex = v1counter + l;
            #endif
            
            double mCos = cos(m*phi);
            double mSin = sin(m*phi);
            for (n = 0; n < N; n++)
            {
                double Acos_val = *(Acos +m + M*l + M*L*n);
                double Asin_val = *(Asin +m + M*l + M*L*n);
                for (i = 0; i < eq_size; i++)
                {
                    double (*Eq)(double, double, double, double, double, double, int) = *(e.Eq + i);
                    double *P = *(e.P + i);
                    double *phiTilde = *(e.phiTilde + i);
                    *(F + i) += (*Eq)(Acos_val, Asin_val, mCos, mSin, P[Pindex], phiTilde[l*N + n], m);
                }


            }
            
           
            
                v2counter++; 
                v1counter+=M-m - 1;  
                
        }
        
        
            
        
    }
    //Multiply F by constants
    for (i = 0; i < eq_size; i++)
    {
        double constant = *(e.Constant + i);
        *(F + i) *= constant*sqrt(4*M_PI);
    }

}



//Compute the Forces
void computeForce(double R,double Z, double phi,
		  double t,
		  struct potentialArg * potentialArgs, double * F)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int isNonAxi = (int)*args++;
    int N = (int)*args++;
    int L = (int)*args++;
    int M = (int)*args++;

    double* Acos = args;

    double* caching_i = (args + (isNonAxi + 1)*N*L*M);
    double *Asin;
    if (isNonAxi == 1)
    {
        Asin = args + N*L*M;
    }
    double *cached_type = caching_i;
    double * cached_coords = (caching_i+ 1);
    double * cached_values = (caching_i + 4);
    if ((int)*cached_type==FORCE)
    {
        if (*cached_coords == R && *(cached_coords + 1) == Z && *(cached_coords + 2) == phi)
        {
            *F = *cached_values;
            *(F + 1) = *(cached_values + 1);
            *(F + 2) = *(cached_values + 2);
            return;
        }
    }
    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);

    double xi;
    calculateXi(r, a, &xi);

//Compute the gegenbauer polynomials and its derivative.
    double *C= (double *) malloc ( N*L * sizeof(double) );
    double *dC= (double *) malloc ( N*L * sizeof(double) );
    double *phiTilde= (double *) malloc ( N*L * sizeof(double) );
    double *dphiTilde= (double *) malloc ( N*L * sizeof(double) );

    compute_C(xi, N, L, C);
    compute_dC(xi, N, L, dC);

//Compute phiTilde and its derivative
    compute_phiTilde(r, a, N, L, C, phiTilde);

    compute_dphiTilde(r, a, N, L, C, dC, dphiTilde);

//Compute Associated Legendre Polynomials
    int M_eff = M;
    int size = 0;
    
    if (isNonAxi==0)
    {
    M_eff = 1;
    size = L;    
    } else{
    size = L*L - L*(L-1)/2;
    }
    
    double *P= (double *) malloc ( size * sizeof(double) );
    double *dP= (double *) malloc ( size * sizeof(double) );
    compute_P_dP(cos(theta), L, M_eff, P, dP);

    double (*PhiTilde_Pointer[3]) = {dphiTilde,phiTilde,phiTilde};
    double (*P_Pointer[3]) = {P, dP, P};

    double Constant[3] = {1., -sin(theta), 1.};

    if (isNonAxi == 1)
    {
        double (*Eq[3])(double, double, double, double, double, double, int) = {&computeF_r, &computeF_theta, &computeF_phi};
        equations e = {Eq,&PhiTilde_Pointer[0], &P_Pointer[0], &Constant[0]};
        computeNonAxi(a, N, L, M,r, theta, phi, Acos, Asin, 3, e, F);
    }
    else
    {
        double (*Eq[3])(double, double, double) = {&computeAxiF_r, &computeAxiF_theta, &computeAxiF_phi};
        axi_equations e = {Eq,&PhiTilde_Pointer[0], &P_Pointer[0], &Constant[0]};
        compute(a, N, L, M,r, theta, phi, Acos, 3, e, F);
    }



    //Caching

    *cached_type = (double)FORCE;

    * cached_coords = R;
    * (cached_coords + 1) = Z;
    * (cached_coords + 2) = phi;
    * (cached_values) = *F;
    * (cached_values + 1) = *(F + 1);
    * (cached_values + 2) = *(F + 2);

    // Free memory
    free(C);
    free(dC);
    free(phiTilde);
    free(dphiTilde);
    free(P);
    free(dP);

}

//Compute the Derivatives
void computeDeriv(double R,double Z, double phi,
		  double t,
		  struct potentialArg * potentialArgs, double * F)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int isNonAxi = (int)*args++;
    int N = (int) *args++;
    int L = (int) *args++;
    int M = (int) *args++;
    double* Acos = args;

    double * caching_i = (args + (isNonAxi + 1)*N*L*M);
    double *Asin;
    if (isNonAxi == 1)
    {
        Asin = args + N*L*M;
    }

    double *cached_type = caching_i;
    double * cached_coords = (caching_i+ 1);
    double * cached_values = (caching_i + 4);
    if ((int)*cached_type==DERIV)
    {
        if (*cached_coords == R && *(cached_coords + 1) == Z && *(cached_coords + 2) == phi)
        {
            *F = *cached_values;
            *(F + 1) = *(cached_values + 1);
            *(F + 2) = *(cached_values + 2);
            return;
        }
    }

    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);

    double xi;
    calculateXi(r, a, &xi);

//Compute the gegenbauer polynomials and its derivative.
    double *C= (double *) malloc ( N*L * sizeof(double) );
    double *dC= (double *) malloc ( N*L * sizeof(double) );
    double *d2C= (double *) malloc ( N*L * sizeof(double) );
    double *phiTilde= (double *) malloc ( N*L * sizeof(double) );
    double *dphiTilde= (double *) malloc ( N*L * sizeof(double) );
    double *d2phiTilde= (double *) malloc ( N*L * sizeof(double) );

    compute_C(xi, N, L, C);
    compute_dC(xi, N, L, dC);
    compute_d2C(xi, N, L, d2C);

//Compute phiTilde and its derivative
    compute_phiTilde(r, a, N, L, C, phiTilde);
    compute_dphiTilde(r, a, N, L, C, dC, dphiTilde);
    compute_d2phiTilde(r, a, N, L, C, dC, d2C, d2phiTilde);


//Compute Associated Legendre Polynomials
    int M_eff = M;
    int size = 0;
    
    if (isNonAxi==0)
    {
    M_eff = 1;
    size = L;    
    } else{
    size = L*L - L*(L-1)/2;
    }
    double *P= (double *) malloc ( size * sizeof(double) );

    compute_P(cos(theta), L,M_eff, P);

    double (*PhiTilde_Pointer[3])= {d2phiTilde,phiTilde,dphiTilde};
    double (*P_Pointer[3]) = {P, P, P};

    double Constant[3] = {1., 1., 1.};

    if (isNonAxi==1)
    {
        double (*Eq[3])(double, double, double, double, double, double, int) = {&computeF_rr, &computeF_phiphi, &computeF_rphi};
        equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        computeNonAxi(a, N, L, M,r, theta, phi, Acos, Asin, 3, e, F);
    }
    else
    {
        double (*Eq[3])(double, double, double) = {&computeAxiF_rr, &computeAxiF_phiphi, &computeAxiF_rphi};
        axi_equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        compute(a, N, L, M,r, theta, phi, Acos, 3, e, F);
    }


    //Caching

    *cached_type = (double)DERIV;

    * cached_coords = R;
    * (cached_coords + 1) = Z;
    * (cached_coords + 2) = phi;
    * (cached_values) = *F;
    * (cached_values + 1) = *(F + 1);
    * (cached_values + 2) = *(F + 2);

    //Free memory
    free(C);
    free(dC);
    free(d2C);
    free(phiTilde);
    free(dphiTilde);
    free(d2phiTilde);
    free(P);
}

//Compute the Potential
double SCFPotentialEval(double R,double Z, double phi,
                        double t,
                        struct potentialArg * potentialArgs)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int isNonAxi = (int)*args++;
    int N = (int) *args++;
    int L = (int) *args++;
    int M = (int) *args++;
    double* Acos = args;
    double* Asin;
    if (isNonAxi==1) //LCOV_EXCL_START
    {
        Asin = args + N*L*M;
    } //LCOV_EXCL_STOP
    //convert R,Z to r, theta
    double r;
    double theta;
    cyl_to_spher(R, Z,&r, &theta);
    double xi;
    calculateXi(r, a, &xi);

    //Compute the gegenbauer polynomials and its derivative.
    double *C= (double *) malloc ( N*L * sizeof(double) );
    double *phiTilde= (double *) malloc ( N*L * sizeof(double) );

    compute_C(xi, N, L, C);

    //Compute phiTilde and its derivative
    compute_phiTilde(r, a, N, L, C, phiTilde);
    //Compute Associated Legendre Polynomials

    int M_eff = M;
    int size = 0;
    
    if (isNonAxi==0)
    {
    M_eff = 1;
    size = L;    
    } else{ //LCOV_EXCL_START
    size = L*L - L*(L-1)/2;
    } //LCOV_EXCL_STOP
    
    double *P= (double *) malloc ( size * sizeof(double) );

    compute_P(cos(theta), L,M_eff, P);

    double potential;

    double (*PhiTilde_Pointer[1]) = {phiTilde};
    double (*P_Pointer[1]) = {P};

    double Constant[1] = {1.};

    if (isNonAxi==1) //LCOV_EXCL_START
    {
        double (*Eq[1])(double, double, double, double, double, double, int) = {&computePhi};
        equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        computeNonAxi(a, N, L, M,r, theta, phi, Acos, Asin, 1, e, &potential);
    } //LCOV_EXCL_STOP
    else
    {
        double (*Eq[1])(double, double, double) = {&computeAxiPhi};
        axi_equations e = {Eq,&PhiTilde_Pointer[0],&P_Pointer[0],&Constant[0]};
        compute(a, N, L, M,r, theta, phi, Acos, 1, e, &potential);
    }

    //Free memory
    free(C);
    free(phiTilde);
    free(P);

    return potential;

}

//Compute the force in the R direction
double SCFPotentialRforce(double R,double Z, double phi,
                          double t,
                          struct potentialArg * potentialArgs)
{
    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);
    // The derivatives
    double dr_dR = R/r;
    double dtheta_dR = Z/(r*r);
    double dphi_dR = 0;


    double F[3];
    computeForce(R, Z, phi, t,potentialArgs, &F[0]) ;

    return *(F + 0)*dr_dR + *(F + 1)*dtheta_dR + *(F + 2)*dphi_dR;


}

//Compute the force in the z direction
double SCFPotentialzforce(double R,double Z, double phi,
                          double t,
                          struct potentialArg * potentialArgs)
{
    double r;
    double theta;
    cyl_to_spher(R, Z,&r, &theta);

    double dr_dz = Z/r;
    double dtheta_dz = -R/(r*r);
    double dphi_dz = 0;

    double F[3];
    computeForce(R, Z, phi, t,potentialArgs, &F[0]) ;
    return *(F + 0)*dr_dz + *(F + 1)*dtheta_dz + *(F + 2)*dphi_dz;
}

//Compute the force in the phi direction
double SCFPotentialphiforce(double R,double Z, double phi,
                            double t,
                            struct potentialArg * potentialArgs)
{

    double r;
    double theta;
    cyl_to_spher(R, Z, &r, &theta);

    double dr_dphi = 0;
    double dtheta_dphi = 0;
    double dphi_dphi = 1;

    double F[3];
    computeForce(R, Z, phi, t,potentialArgs, &F[0]) ;

    return *(F + 0)*dr_dphi + *(F + 1)*dtheta_dphi + *(F + 2)*dphi_dphi;
}

//Compute the planar force in the R direction
double SCFPotentialPlanarRforce(double R,double phi,
                                double t,
                                struct potentialArg * potentialArgs)
{
    return SCFPotentialRforce(R,0., phi,t,potentialArgs);

}

//Compute the planar force in the phi direction
double SCFPotentialPlanarphiforce(double R,double phi,
                                  double t,
                                  struct potentialArg * potentialArgs)
{
    return SCFPotentialphiforce(R,0., phi,t,potentialArgs);
}


//Compute the planar double derivative of the potential with respect to R
double SCFPotentialPlanarR2deriv(double R, double phi,
                                 double t,
                                 struct potentialArg * potentialArgs)
{
    double Farray[3];
    computeDeriv(R, 0, phi, t,potentialArgs, &Farray[0]) ;
    return *Farray;
}

//Compute the planar double derivative of the potential with respect to phi
double SCFPotentialPlanarphi2deriv(double R, double phi,
                                   double t,
                                   struct potentialArg * potentialArgs)
{
    double Farray[3];
    computeDeriv(R, 0, phi, t,potentialArgs, &Farray[0]) ;
    return *(Farray + 1);
}

//Compute the planar double derivative of the potential with respect to R, Phi
double SCFPotentialPlanarRphideriv(double R, double phi,
                                   double t,
                                   struct potentialArg * potentialArgs)
{
    double Farray[3];
    computeDeriv(R, 0, phi, t,potentialArgs, &Farray[0]) ;
    return *(Farray + 2);
}
