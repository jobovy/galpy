#include <math.h>
#include <galpy_potentials.h>

double gam(double R, double phi, double N, double phi_ref, double r_ref, double tan_alpha);

double dgam_dR(double R, double N, double tan_alpha);

double K(double R, double n, double N, double sin_alpha);

double B(double R, double H, double n, double N, double sin_alpha);

double D(double R, double H, double n, double N, double sin_alpha);

double dK_dR(double R, double n, double N, double sin_alpha);

double dB_dR(double R, double H, double n, double N, double sin_alpha);

double dD_dR(double R, double H, double n, double N, double sin_alpha);

//LCOV_EXCL_START
double SpiralArmsPotentialEval(double R, double z, double phi, double t,
                               struct potentialArg *potentialArgs) {
    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);

    // Return value of the potential.
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Bn;
    double Dn;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        sum += Cn / Kn / Dn * cos(n * g) / pow(cosh(Kn * z / Bn), Bn);
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}
//LCOV_EXCL_STOP

double SpiralArmsPotentialRforce(double R, double z, double phi, double t,
                                 struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);
    double dg_dR = dgam_dR(R, N, tan_alpha);

    // Return the Rforce (-dPhi / dR)
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Bn;
    double Dn;

    double dKn_dR;
    double dBn_dR;
    double dDn_dR;

    double cos_ng;
    double sin_ng;

    double zKB;
    double sechzKB;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        dKn_dR = dK_dR(R, n, N, sin_alpha);
        dBn_dR = dB_dR(R, H, n, N, sin_alpha);
        dDn_dR = dD_dR(R, H, n, N, sin_alpha);

        cos_ng = cos(n * g);
        sin_ng = sin(n * g);

        zKB = z * Kn / Bn;
        sechzKB = 1 / cosh(zKB);

        sum += Cn * pow(sechzKB, Bn) / Dn * ((n * dg_dR / Kn * sin_ng
                                              + cos_ng * (z * tanh(zKB) * (dKn_dR / Kn - dBn_dR / Bn)
                                                          - dBn_dR / Kn * log(sechzKB)
                                                          + dKn_dR / Kn / Kn
                                                          + dDn_dR / Dn / Kn))
                                             + cos_ng / Kn / Rs);
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}


double SpiralArmsPotentialzforce(double R, double z, double phi, double t,
                                 struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);

    // Return the zforce (-dPhi / dz)
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Bn;
    double Dn;

    double zKn_Bn;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        zKn_Bn = z * Kn / Bn;

        sum += Cn / Dn * cos(n * g) * tanh(zKn_Bn) / pow(cosh(zKn_Bn), Bn);
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}

double SpiralArmsPotentialphiforce(double R, double z, double phi, double t,
                                   struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);

    // Return the phiforce (-dPhi / dphi)
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Bn;
    double Dn;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        sum += N * n * Cn / Dn / Kn / pow(cosh(z * Kn / Bn), Bn) * sin(n * g);
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}

//LCOV_EXCL_START
double SpiralArmsPotentialR2deriv(double R, double z, double phi, double t,
                                  struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);
    double dg_dR = dgam_dR(R, N, tan_alpha);
    double d2g_dR2 = N / R / R / tan_alpha;

    // Return the second derivative of the potential wrt R (d2Phi / dR2)
    double sum = 0;

    int n;

    double Cn;

    double Kn;
    double Bn;
    double Dn;

    double dKn_dR;
    double dBn_dR;
    double dDn_dR;

    double HNn;
    double HNn_R;
    double R_sina;
    double HNn_R_sina;
    double HNn_R_sina_2;
    double x;

    double d2Kn_dR2;
    double d2Bn_dR2;
    double d2Dn_dR2;

    double cos_ng;
    double sin_ng;

    double zKB;
    double sechzKB;
    double sechzKB_B;
    double log_sechzKB;
    double tanhzKB;
    double ztanhzKB;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;

        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        dKn_dR = dK_dR(R, n, N, sin_alpha);
        dBn_dR = dB_dR(R, H, n, N, sin_alpha);
        dDn_dR = dD_dR(R, H, n, N, sin_alpha);

        HNn = H * N * n;
        HNn_R = HNn / R;
        R_sina = R * sin_alpha;
        HNn_R_sina = HNn / R_sina;
        HNn_R_sina_2 = HNn_R_sina * HNn_R_sina;
        x = R * (0.3 * HNn_R_sina + 1) * sin_alpha;

        d2Kn_dR2 = 2 * N * n / R / R / R / sin_alpha;
        d2Bn_dR2 = HNn_R / R / R / sin_alpha * (2.4 * HNn_R / sin_alpha + 2);
        d2Dn_dR2 = sin_alpha / R / x * (HNn * (0.18 * HNn * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x / x
                                                      + 2 / R_sina
                                                      - 0.6 * HNn_R_sina * (1 + 0.6 * HNn_R_sina) / x
                                                      - 0.6 * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x
                                                      + 1.8 * HNn / R_sina / R_sina));

        cos_ng = cos(n * g);
        sin_ng = sin(n * g);

        zKB = z * Kn / Bn;
        sechzKB = 1 / cosh(zKB);
        sechzKB_B = pow(sechzKB, Bn);
        log_sechzKB = log(sechzKB);
        tanhzKB = tanh(zKB);
        ztanhzKB = z * tanhzKB;

        sum += (Cn * sechzKB_B / Dn * ((n * dg_dR / Kn * sin_ng
                                        + cos_ng * (ztanhzKB * (dKn_dR / Kn - dBn_dR / Bn)
                                                    - dBn_dR / Kn * log_sechzKB
                                                    + dKn_dR / Kn / Kn
                                                    + dDn_dR / Dn / Kn))
                                       - (Rs * (1 / Kn * ((ztanhzKB * (dBn_dR / Bn * Kn - dKn_dR)
                                                           + log_sechzKB * dBn_dR)
                                                          - dDn_dR / Dn) * (n * dg_dR * sin_ng
                                                                            + cos_ng * (ztanhzKB * Kn *
                                                                                        (dKn_dR / Kn - dBn_dR / Bn)
                                                                                        - dBn_dR * log_sechzKB
                                                                                        + dKn_dR / Kn / Kn
                                                                                        + dDn_dR / Dn))
                                                + (n * (sin_ng * (d2g_dR2 / Kn - dg_dR / Kn / Kn * dKn_dR)
                                                        + dg_dR * dg_dR / Kn * cos_ng * n)
                                                   + z * (-sin_ng * n * dg_dR * tanhzKB * (dKn_dR / Kn - dBn_dR / Bn)
                                                          + cos_ng * (z * (dKn_dR / Bn - dBn_dR / Bn / Bn * Kn) *
                                                                      (1 - tanhzKB * tanhzKB) *
                                                                      (dKn_dR / Kn - dBn_dR / Bn)
                                                                      + tanhzKB *
                                                                        (d2Kn_dR2 / Kn - (dKn_dR / Kn) * (dKn_dR / Kn) -
                                                                         d2Bn_dR2 / Bn +
                                                                         (dBn_dR / Bn) * (dBn_dR / Bn))))
                                                   + (cos_ng *
                                                      (dBn_dR / Kn * ztanhzKB * (dKn_dR / Bn - dBn_dR / Bn / Bn * Kn)
                                                       - (d2Bn_dR2 / Kn - dBn_dR * dKn_dR / Kn / Kn) * log_sechzKB)
                                                      + dBn_dR / Kn * log_sechzKB * sin_ng * n * dg_dR)
                                                   + ((cos_ng * (d2Kn_dR2 / Kn / Kn - 2 * dKn_dR * dKn_dR / Kn / Kn / Kn)
                                                     - dKn_dR / Kn / Kn * sin_ng * n * dg_dR)
                                                    + (cos_ng * (d2Dn_dR2 / Dn / Kn
                                                                 - (dDn_dR / Dn) * (dDn_dR / Dn) / Kn
                                                                 - dDn_dR / Dn / Kn / Kn * dKn_dR)
                                                       - sin_ng * n * dg_dR * dDn_dR / Dn / Kn))))
                                          - 1 / Kn * (cos_ng / Rs
                                                      + (cos_ng * ((dDn_dR * Kn + Dn * dKn_dR) / (Dn * Kn)
                                                                   - (ztanhzKB * (dBn_dR / Bn * Kn - dKn_dR)
                                                                      + log_sechzKB * dBn_dR))
                                                         + sin_ng * n * dg_dR)))));
    }

    return -amp * H * exp(-(R - r_ref) / Rs) / Rs * sum;
}
//LCOV_EXCL_STOP

//LCOV_EXCL_START
double SpiralArmsPotentialz2deriv(double R, double z, double phi, double t,
                                  struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);

    // Return the second derivative of the potential wrt z (d2Phi / dz2)
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Bn;
    double Dn;

    double zKB;
    double tanh2_zKB;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        zKB = z * Kn / Bn;
        tanh2_zKB = tanh(zKB) * tanh(zKB);

        sum += Cn * Kn / Dn * ((tanh2_zKB - 1) / Bn + tanh2_zKB) * cos(n * g) / pow(cosh(zKB), Bn);
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}
//LCOV_EXCL_STOP

//LCOV_EXCL_START
double SpiralArmsPotentialphi2deriv(double R, double z, double phi, double t,
                                    struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);

    // Return the second derivative of the potential wrt phi (d2Phi / dphi2)
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Bn;
    double Dn;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        sum += Cn * N * N * n * n / Dn / Kn / pow(cosh(z * Kn / Bn), Bn) * cos(n * g);
    }

    return amp * H * exp(-(R - r_ref) / Rs) * sum;
}
//LCOV_EXCL_STOP

//LCOV_EXCL_START
double SpiralArmsPotentialRzderiv(double R, double z, double phi, double t,
                                  struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);
    double dg_dR = dgam_dR(R, N, tan_alpha);

    // Return the mixed (cylindrical) radial and vertical derivative of the potential (d^2 potential / dR dz).
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Bn;
    double Dn;

    double dKn_dR;
    double dBn_dR;
    double dDn_dR;

    double cos_ng;
    double sin_ng;

    double zKB;
    double sechzKB;
    double sechzKB_B;
    double log_sechzKB;
    double tanhzKB;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;

        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        dKn_dR = dK_dR(R, n, N, sin_alpha);
        dBn_dR = dB_dR(R, H, n, N, sin_alpha);
        dDn_dR = dD_dR(R, H, n, N, sin_alpha);

        cos_ng = cos(n * g);
        sin_ng = sin(n * g);

        zKB = z * Kn / Bn;
        sechzKB = 1 / cosh(zKB);
        sechzKB_B = pow(sechzKB, Bn);
        log_sechzKB = log(sechzKB);
        tanhzKB = tanh(zKB);

        sum += sechzKB_B * Cn / Dn * (Kn * tanhzKB * (n * dg_dR / Kn * sin_ng
                                                      + cos_ng * (z * tanhzKB * (dKn_dR / Kn - dBn_dR / Bn)
                                                                  - dBn_dR / Kn * log_sechzKB
                                                                  + dKn_dR / Kn / Kn
                                                                  + dDn_dR / Dn / Kn))
                                      - cos_ng * ((zKB * (dKn_dR / Kn - dBn_dR / Bn) * (1 - tanhzKB * tanhzKB)
                                                   + tanhzKB * (dKn_dR / Kn - dBn_dR / Bn)
                                                   + dBn_dR / Bn * tanhzKB)
                                                  - tanhzKB / Rs));
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}
//LCOV_EXCL_STOP

//LCOV_EXCL_START
double SpiralArmsPotentialRphideriv(double R, double z, double phi, double t,
                                    struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);
    double dg_dR = dgam_dR(R, N, tan_alpha);

    // Return the mixed (cylindrical) radial and azimuthal derivative of the potential (d^2 potential / dR dphi).
    double sum = 0;
    int n;

    double Cn;

    double Kn;
    double Bn;
    double Dn;

    double dKn_dR;
    double dBn_dR;
    double dDn_dR;

    double cos_ng;
    double sin_ng;

    double zKB;
    double sechzKB;
    double sechzKB_B;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;

        Kn = K(R, n, N, sin_alpha);
        Bn = B(R, H, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        dKn_dR = dK_dR(R, n, N, sin_alpha);
        dBn_dR = dB_dR(R, H, n, N, sin_alpha);
        dDn_dR = dD_dR(R, H, n, N, sin_alpha);

        cos_ng = cos(n * g);
        sin_ng = sin(n * g);

        zKB = z * Kn / Bn;
        sechzKB = 1 / cosh(zKB);
        sechzKB_B = pow(sechzKB, Bn);

        sum += Cn * sechzKB_B / Dn * n * N
               * (-n * dg_dR / Kn * cos_ng
                  + sin_ng * (z * tanh(zKB) * (dKn_dR / Kn - dBn_dR / Bn)
                              + 1 / Kn * (-dBn_dR * log(sechzKB)
                                          + dKn_dR / Kn
                                          + dDn_dR / Dn
                                          + 1 / Rs)));
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}
//LCOV_EXCL_STOP

double SpiralArmsPotentialPlanarRforce(double R, double phi, double t,
                                       struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);
    double dg_dR = dgam_dR(R, N, tan_alpha);

    // Return the planar Rforce (-dPhi / dR)
    double sum = 0;
    int n;

    double Cn;

    double Kn;
    double Dn;

    double dKn_dR;
    double dDn_dR;

    double cos_ng;
    double sin_ng;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;

        Kn = K(R, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        dKn_dR = dK_dR(R, n, N, sin_alpha);
        dDn_dR = dD_dR(R, H, n, N, sin_alpha);

        cos_ng = cos(n * g);
        sin_ng = sin(n * g);

        sum += Cn / Dn * ((n * dg_dR / Kn * sin_ng
                           + cos_ng * (dKn_dR / Kn / Kn + dDn_dR / Dn / Kn))
                          + cos_ng / Kn / Rs);
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}

double SpiralArmsPotentialPlanarphiforce(double R, double phi, double t,
                                         struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);

    // Return the planar phiforce (-dPhi / dphi)
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Dn;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        sum += N * n * Cn / Dn / Kn * sin(n * g);
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}

double SpiralArmsPotentialPlanarR2deriv(double R, double phi, double t,
                                        struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);
    double dg_dR = dgam_dR(R, N, tan_alpha);
    double d2g_dR2 = N / R / R / tan_alpha;

    // Return the planar second derivative of the potential wrt R (d2Phi / dR2)
    double sum = 0;
    int n;

    double Cn;

    double Kn;
    double Dn;

    double dKn_dR;
    double dDn_dR;

    double HNn;
    double R_sina;
    double HNn_R_sina;
    double HNn_R_sina_2;
    double x;

    double d2Kn_dR2;
    double d2Dn_dR2;

    double cos_ng;
    double sin_ng;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;

        Kn = K(R, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        dKn_dR = dK_dR(R, n, N, sin_alpha);
        dDn_dR = dD_dR(R, H, n, N, sin_alpha);

        HNn = H * N * n;
        R_sina = R * sin_alpha;
        HNn_R_sina = HNn / R_sina;
        HNn_R_sina_2 = HNn_R_sina * HNn_R_sina;
        x = R * (0.3 * HNn_R_sina + 1) * sin_alpha;

        d2Kn_dR2 = 2 * N * n / R / R / R / sin_alpha;
        d2Dn_dR2 = sin_alpha / R / x * (HNn * (0.18 * HNn * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x / x
                                                      + 2 / R_sina
                                                      - 0.6 * HNn_R_sina * (1 + 0.6 * HNn_R_sina) / x
                                                      - 0.6 * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x
                                                      + 1.8 * HNn / R_sina / R_sina));

        cos_ng = cos(n * g);
        sin_ng = sin(n * g);

        sum += (Cn / Dn * ((n * dg_dR / Kn * sin_ng
                            + cos_ng * (dKn_dR / Kn / Kn
                                        + dDn_dR / Dn / Kn))
                           - (Rs * (1 / Kn * (-dDn_dR / Dn) * (n * dg_dR * sin_ng
                                                               + cos_ng * (dKn_dR / Kn / Kn
                                                                           + dDn_dR / Dn))
                                    + (n * (sin_ng * (d2g_dR2 / Kn - dg_dR / Kn / Kn * dKn_dR)
                                            + dg_dR * dg_dR / Kn * cos_ng * n)
                                       + ((cos_ng * (d2Kn_dR2 / Kn / Kn - 2 * dKn_dR * dKn_dR / Kn / Kn / Kn)
                                           - dKn_dR / Kn / Kn * sin_ng * n * dg_dR)
                                          + (cos_ng * (d2Dn_dR2 / Dn / Kn
                                                       - (dDn_dR / Dn) * (dDn_dR / Dn) / Kn
                                                       - dDn_dR / Dn / Kn / Kn * dKn_dR)
                                             - sin_ng * n * dg_dR * dDn_dR / Dn / Kn))))
                              - 1 / Kn * (cos_ng / Rs
                                          + (cos_ng * ((dDn_dR * Kn + Dn * dKn_dR) / (Dn * Kn))
                                             + sin_ng * n * dg_dR)))));
    }

    return -amp * H * exp(-(R - r_ref) / Rs) / Rs * sum;
}

double SpiralArmsPotentialPlanarphi2deriv(double R, double phi, double t,
                                          struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);

    // Return the second derivative of the potential wrt phi (d2Phi / dphi2)
    double sum = 0;
    int n;

    double Cn;
    double Kn;
    double Dn;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;
        Kn = K(R, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        sum += Cn * N * N * n * n / Dn / Kn * cos(n * g);
    }

    return amp * H * exp(-(R - r_ref) / Rs) * sum;
}

double SpiralArmsPotentialPlanarRphideriv(double R, double phi, double t,
                                          struct potentialArg *potentialArgs) {

    // Get args
    double *args = potentialArgs->args;

    int nCs = (int) *args++;
    double amp = *args++;
    double N = *args++;
    double sin_alpha = *args++;
    double tan_alpha = *args++;
    double r_ref = *args++;
    double phi_ref = *args++;
    double Rs = *args++;
    double H = *args++;
    double omega = *args++;

    double g = gam(R, phi-omega*t, N, phi_ref, r_ref, tan_alpha);
    double dg_dR = dgam_dR(R, N, tan_alpha);

    // Return the mixed (cylindrical) radial and azimuthal derivative of the potential (d^2 potential / dR dphi).
    double sum = 0;
    int n;

    double Cn;

    double Kn;
    double Dn;

    double dKn_dR;
    double dDn_dR;

    double cos_ng;
    double sin_ng;

    for (n = 1; n <= nCs; n++) {
        Cn = *args++;

        Kn = K(R, n, N, sin_alpha);
        Dn = D(R, H, n, N, sin_alpha);

        dKn_dR = dK_dR(R, n, N, sin_alpha);
        dDn_dR = dD_dR(R, H, n, N, sin_alpha);

        cos_ng = cos(n * g);
        sin_ng = sin(n * g);

        sum += Cn / Dn * n * N * (-n * dg_dR / Kn * cos_ng
                                  + sin_ng * (1 / Kn * (dKn_dR / Kn
                                                        + dDn_dR / Dn
                                                        + 1 / Rs)));
    }

    return -amp * H * exp(-(R - r_ref) / Rs) * sum;
}

double gam(double R, double phi, double N, double phi_ref, double r_ref, double tan_alpha) {
    return N * (phi - phi_ref - log(R / r_ref) / tan_alpha);
}

double dgam_dR(double R, double N, double tan_alpha) {
    return -N / R / tan_alpha;
}

double K(double R, double n, double N, double sin_alpha) {
    return n * N / R / sin_alpha;
}

double B(double R, double H, double n, double N, double sin_alpha) {
    double HNn_R = H * N * n / R;
    return HNn_R / sin_alpha * (0.4 * HNn_R / sin_alpha + 1);
}

double D(double R, double H, double n, double N, double sin_alpha) {
    double HNn = H * N * n;
    return (0.3 * HNn * HNn / sin_alpha / R
            + HNn + R * sin_alpha) / (0.3 * HNn + R * sin_alpha);
}

double dK_dR(double R, double n, double N, double sin_alpha) {
    return -n * N / (R * R) / sin_alpha;
}

double dB_dR(double R, double H, double n, double N, double sin_alpha) {
    double HNn = H * N * n;
    return -HNn / R / R / R / sin_alpha / sin_alpha * (0.8 * HNn + R * sin_alpha);
}

double dD_dR(double R, double H, double n, double N, double sin_alpha) {
    double HNn_R_sina = H * N * n / R / sin_alpha;
    return HNn_R_sina * (0.3 * (HNn_R_sina + 0.3 * HNn_R_sina * HNn_R_sina + 1) / R / pow((0.3 * HNn_R_sina + 1), 2)
                         - (1 / R * (1 + 0.6 * HNn_R_sina) / (0.3 * HNn_R_sina + 1)));
}

