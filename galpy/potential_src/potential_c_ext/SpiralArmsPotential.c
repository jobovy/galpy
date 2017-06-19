#include <math.h>
#include <galpy_potentials.h>

double gamma(double R, double phi, double N, double phi_ref, double r_ref, double alpha);
double K(double R, double n, double N, double alpha);
double B(double R, double H, double n, double N, double alpha);
double D(double R, double H, double n, double N, double alpha);
double dK_dR(double R, double n, double N, double alpha);
double dB_dR(double R, double H, double n, double N, double alpha);
double dD_dR(double R, double H, double n, double N, double alpha);

double SpiralArmsPotentialEval(double R, double z, double phi, double t,
                                struct potentialArg* potentialArgs){
    double *args = potentialArgs->args;
    /*
     * Evaluate the potential at (R, z, phi, t)
     */

    // Get args
    double nCs = *args++;
    double amp = *args++;
    double N = *args++;
    double alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;
    double Cs[nCs];
    double g = gamma(R, phi, N, phi_ref, r_ref, alpha);

    for (int k=0; k < nCs; k++) {
        Cs[k] = *args++;
    }

    // Return potential
    sum = 0;
    for (int n=1; n <= nCs; n++){
        Cn = C[n-1];
        Kn = K(R, n, N, alpha);
        Bn = B(R, H, n, N, alpha);
        Dn = D(R, H, n, N, alpha);

        sum += C / K / D * cos(n * g) / pow(cosh(K * z / B), B);
    }

    return -H * exp(-(R - r_ref) / Rs) * sum;
}

double SpiralArmsPotentialRforce(double R, double z, double phi, double t,
                                struct potentialArg* potentialArgs){
    double *args= potentialArgs->args;

    // Get args
    double nCs = *args++;
    double amp = *args++;
    double N = *args++;
    double alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;
    double Cs[nCs];

    for (int k=0; k < nCs; k++) {
        Cs[k] = *args++;
    }

    return 1.0;
}

double SpiralArmsPotentialzforce(double R, double z, double phi, double t,
                                struct potentialArg* potentialArgs){
    double *args= potentialArgs->args;

    // Get args
    double nCs = *args++;
    double amp = *args++;
    double N = *args++;
    double alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;
    double Cs[nCs];

    for (int k=0; k < nCs; k++) {
        Cs[k] = *args++;
    }

    return 1.0;
}

double SpiralArmsPotentialphiforce(double R, double z, double phi, double t,
                                struct potentialArg* potentialArgs){
    double *args= potentialArgs->args;

    // Get args
    double nCs = *args++;
    double amp = *args++;
    double N = *args++;
    double alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;
    double Cs[nCs];

    for (int k=0; k < nCs; k++) {
        Cs[k] = *args++;
    }

    return 1.0;
}

double gamma(double R, double phi, double N, double phi_ref, double r_ref, double alpha){
    return N * (phi - phi_ref - log(R / r_ref) / tan(alpha));
}

double K(double R, double n, double N, double alpha) {
    return n * N / R / sin(alpha);
}

double B(double R, double H, double n, double N, double alpha) {
    HNn_R = H * N * n / R;
    sin_alpha = sin(alpha);
    return HNn_R / sin_alpha * (0.4 * HNn_R / sin_alpha + 1.0);
}

double D(double R, double H, double n, double N, double alpha){
    sin_alpha = sin(alpha);
    HNn = H * N * n;
    return (0.3 * HNn * HNn / sin_alpha / R
            + HNn + R * sin_alpha) / (0.3 * HNn + R * sin_alpha);
}

double dK_dR(double R, double n, double N, double alpha){
    return -n * N / (R*R) / sin(alpha);
}

double dB_dR(double R, double H, double n, double N, double alpha){
    sin_alpha = sin(alpha);
    HNn = H * N * n;
    return -HNn / R / R / R / sin_alpha / sin_alpha * (0.8 * HNn + R * sin_alpha);
}

double dD_dR(double R, double H, double n, double N, double alpha){
    HNn_R_sina = H * N * n / R / sin(alpha);
    return HNn_R_sina * (0.3 * (HNn_R_sina + 0.3 * HNn_R_sina * HNn_R_sina + 1) / R / pow((0.3 * HNn_R_sina + 1), 2)
                             - (1/R * (1 + 0.6 * HNn_R_sina) / (0.3 * HNn_R_sina + 1)));

}

