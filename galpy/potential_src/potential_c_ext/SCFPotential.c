#include <math.h>
#include <galpy_potentials.h>
#include <stdio.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_legendre.h>

//SCF Disk potential
//4 arguments: amp, Acos, Asin, a

const int FORCE =1;
const int DERIV =2;

//Useful Functions

//Converts from cylindrical coordinates to spherical
inline void cyl_to_spher(double R, double Z,double *r, double *theta)
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
inline void calculateXi(double r, double a, double *xi)
{
    *xi = (r - a)/(r + a);
}


//Potentials, forces, and derivative functions

double computePhi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double phiTilde, int m)
{
    return (Acos_val*mCos + Asin_val*mSin)*P*phiTilde;
}

double computeF_r(double Acos_val, double Asin_val, double mCos, double mSin, double P, double dphiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*P*dphiTilde;
}

double computeF_theta(double Acos_val, double Asin_val, double mCos, double mSin, double dP, double phiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*dP*phiTilde;
}

double computeF_phi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double phiTilde, int m)
{
    return m*(Acos_val*mSin - Asin_val*mCos)*P*phiTilde;
}

double computeF_rr(double Acos_val, double Asin_val, double mCos, double mSin, double P, double d2phiTilde, int m)
{
    return -(Acos_val*mCos + Asin_val*mSin)*P*d2phiTilde;
}

double computeF_rphi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double dphiTilde, int m)
{
    return m*(Acos_val*mSin - Asin_val*mCos)*P*dphiTilde;
}

double computeF_phiphi(double Acos_val, double Asin_val, double mCos, double mSin, double P, double phiTilde, int m)
{
    return m*m*(Acos_val*mCos + Asin_val*mSin)*P*phiTilde;
}

//Calculates the Gegenbauer polynomials
inline void compute_C(double xi, int N, int L, double * C_array)
{
    int l;
    for (l = 0; l < L; l++)
    {
        gsl_sf_gegenpoly_array(N - 1, 3./2 + 2*l, xi, C_array + l*N);
    }
}

//Calculates the derivative of the Gegenbauer polynomials
inline void compute_dC(double xi, int N, int L, double * dC_array)
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
inline void compute_d2C(double xi, int N, int L, double * d2C_array)
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
inline void compute_phiTilde(double r, double a, int N, int L, double* C, double * phiTilde)
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
inline void compute_dphiTilde(double r, double a, int N, int L, double * C, double * dC, double * dphiTilde)
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
inline void compute_d2phiTilde(double r, double a, int N, int L, double * C, double * dC,double * d2C, double * d2phiTilde)
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
inline void compute_P(double x, int L, double * P_array)
{
    int l;
    for (l = 0; l < L; l++)
    {
        int shift = l*L + l;
        gsl_sf_legendre_Plm_array(L - 1, l, x, P_array + shift);
    }
}

//Computes the associated Legendre polynomials and its derivative 
inline void compute_P_dP(double x, int L, double * P_array, double *dP_array)
{
    int l;
    for (l = 0; l < L; l++)
    {
        int shift = l*L + l;
        gsl_sf_legendre_Plm_deriv_array(L - 1, l, x, P_array + shift, dP_array + shift);
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

//Compute 
inline void compute(double a, int N, int L, int M,
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

    
    for (l = 0; l < L; l++)
    {
        for (m = 0; m<=l; m++)
        {
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
                    *(F + i) += (*Eq)(Acos_val, Asin_val, mCos, mSin, P[m*L + l], phiTilde[l*N + n], m);
                }


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



//Compute the Forces
inline void computeForce(double R,double Z, double phi,
                  double t,
                  struct potentialArg * potentialArgs, double * F)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int N = (int)*args++;
    int L = (int)*args++;
    int M = (int)*args++;
    double* Acos = args;
    double* Asin = args + N*L*M;

    double *cached_type = (args + 2*N*L*M);
    double * cached_coords = (args + 2*N*L*M + 1);
    double * cached_values = (args + 2*N*L*M + 4);
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

    compute_P_dP(cos(theta), L, &P, &dP);

    double (*Eq[3])(double, double, double, double, double, double, int) = {&computeF_r, &computeF_theta, &computeF_phi};
    double (*PhiTilde_Pointer[3]) = {&dphiTilde, &phiTilde, &phiTilde};
    double (*P_Pointer[3]) = {&P, &dP, &P};

    double Constant[3] = {1., -sin(theta), 1.};
    equations e = {Eq,&PhiTilde_Pointer, &P_Pointer, &Constant};


    compute(a, N, L, M,r, theta, phi, Acos, Asin, 3, e, F);

    //Caching

    *cached_type = (double)FORCE;

    * cached_coords = R;
    * (cached_coords + 1) = Z;
    * (cached_coords + 2) = phi;
    * (cached_values) = *F;
    * (cached_values + 1) = *(F + 1);
    * (cached_values + 2) = *(F + 2);



}

//Compute the Derivatives
inline void computeDeriv(double R,double Z, double phi,
                  double t,
                  struct potentialArg * potentialArgs, double * F)
{
    double * args= potentialArgs->args;
    //Get args
    double a = *args++;
    int N = *args++;
    int L = *args++;
    int M = *args++;
    double* Acos = args;
    double* Asin = args + N*L*M;

    double *cached_type = (args + 2*N*L*M);
    double * cached_coords = (args + 2*N*L*M + 1);
    double * cached_values = (args + 2*N*L*M + 4);
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
    double C[N*L];
    double dC[N*L];
    double d2C[N*L];

    compute_C(xi, N, L, &C);
    compute_dC(xi, N, L, &dC);
    compute_d2C(xi, N, L, &d2C);

//Compute phiTilde and its derivative
    double phiTilde[L*N];
    compute_phiTilde(r, a, N, L, &C, &phiTilde);

    double dphiTilde[L*N];
    compute_dphiTilde(r, a, N, L, &C, &dC, &dphiTilde);

    double d2phiTilde[L*N];
    compute_d2phiTilde(r, a, N, L, &C, &dC, &d2C, &d2phiTilde);


//Compute Associated Legendre Polynomials
    double P[L*L];

    compute_P(cos(theta), L, &P);

    int num_eq = 3;

    double (*Eq[3])(double, double, double, double, double, double, int) = {&computeF_rr, &computeF_phiphi, &computeF_rphi};
    double (*PhiTilde_Pointer[3]) = {&d2phiTilde, &phiTilde, &dphiTilde};
    double (*P_Pointer[3]) = {&P, &P, &P};

    double Constant[3] = {1., 1., 1.};
    equations e = {Eq,&PhiTilde_Pointer, &P_Pointer, &Constant};

    compute(a, N, L, M,r, theta, phi, Acos, Asin, 3, e, F);

    //Caching

    *cached_type = (double)DERIV;

    * cached_coords = R;
    * (cached_coords + 1) = Z;
    * (cached_coords + 2) = phi;
    * (cached_values) = *F;
    * (cached_values + 1) = *(F + 1);
    * (cached_values + 2) = *(F + 2);

}

//Compute the Potential
double SCFPotentialEval(double R,double Z, double phi,
                        double t,
                        struct potentialArg * potentialArgs)
{
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

    compute_P(cos(theta), L, &P);

    double potential;

    int num_eq = 1;
    double (*Eq[1])(double, double, double, double, double, double, int) = {&computePhi};
    double (*PhiTilde_Pointer[1]) = {&phiTilde};
    double (*P_Pointer[1]) = {&P};

    double Constant[1] = {1.};
    equations e = {Eq,&PhiTilde_Pointer, &P_Pointer, &Constant};


    compute(a, N, L, M,r, theta, phi, Acos, Asin, 1, e, &potential);
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
    computeForce(R, Z, phi, t,potentialArgs, &F) ;

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
    computeForce(R, Z, phi, t,potentialArgs, &F) ;
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
    computeForce(R, Z, phi, t,potentialArgs, &F) ;

    return *(F + 0)*dr_dphi + *(F + 1)*dtheta_dphi + *(F + 2)*dphi_dphi;
}

//Compute the planar force in the R direction
double SCFPotentialPlanarRforce(double R,double phi,
                                double t,
                                struct potentialArg * potentialArgs)
{
    double * args= potentialArgs->args;
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

    computeDeriv(R, 0, phi, t,potentialArgs, &Farray) ;

    return *(Farray + 0);


}

//Compute the planar double derivative of the potential with respect to phi
double SCFPotentialPlanarphi2deriv(double R, double phi,
                                   double t,
                                   struct potentialArg * potentialArgs)
{


    double Farray[3];

    computeDeriv(R, 0, phi, t,potentialArgs, &Farray) ;

    return *(Farray + 1);


}

//Compute the planar double derivative of the potential with respect to R, Phi
double SCFPotentialPlanarRphideriv(double R, double phi,
                                   double t,
                                   struct potentialArg * potentialArgs)
{

    double Farray[3];
    computeDeriv(R, 0, phi, t,potentialArgs, &Farray) ;
    return *(Farray + 2);

}
