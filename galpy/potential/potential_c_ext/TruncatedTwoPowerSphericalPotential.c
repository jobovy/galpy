#include <math.h>
#include <galpy_potentials.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//TruncatedTwoPowerSphericalPotential: NFW (k=2) and Hernquist (k=3) profiles
//with the Baltz, Marshall & Oguri (2009) truncation (rc^2/(rc^2+r^2))^n.
//Mirrors TruncatedTwoPowerSphericalPotential.py, which has the derivation.
//Arguments: amp, a, k, n, Nt, then either rc (Nt = 0) or tgrid[Nt] followed
//by the power-basis coefficients of log rc(t)'s not-a-knot cubic spline, four per
//interval (highest power first).
#define TTPS_SMALL_X 0.1
#define TTPS_LARGE_X 10.0
#define TTPS_NSERIES 20
// binomial(-m,i), the coefficient of s^i in (1+s)^(-m)
static double ttps_binom_neg(int m, int i) {
  double out = 1.0;
  int ii;
  for (ii = 1; ii <= i; ii++)
    out *= (double)(m + ii - 1) / ii;
  return (i % 2) ? -out : out;
}
// tau^(2n) P / (1+tau^2)^p without forming tau^(2n)
static double ttps_c(double P, int p, double rn, double opT, int n) {
  return rn * P * pow(opT, n - p);
}
// Partial-fraction coefficients of the enclosed-mass (F, which=0) or
// outer-potential (H, which=1) integrand: A[0..k-1] here, and the j-th
// (B_j, C_j) pair through ttps_BC
static void ttps_A(int k, int n, int which, double tau, double *A) {
  double T = tau * tau, opT = 1.0 + T, rn = pow(T / opT, n);
  if (k == 2 && which == 0) {
    A[0] = ttps_c(T - (2 * n - 1), n + 1, rn, opT, n);
    A[1] = ttps_c(-1.0, n, rn, opT, n);
  } else if (k == 2) {
    A[0] = ttps_c(2.0 * n, n + 1, rn, opT, n);
    A[1] = ttps_c(1.0, n, rn, opT, n);
  } else if (which == 0) {
    A[0] = ttps_c(3 * n * T - n * (2 * n - 1), n + 2, rn, opT, n);
    A[1] = ttps_c(T - (2 * n - 1), n + 1, rn, opT, n);
    A[2] = ttps_c(-1.0, n, rn, opT, n);
  } else {
    A[0] = ttps_c(n * (2 * n + 1) - n * T, n + 2, rn, opT, n);
    A[1] = ttps_c(2.0 * n, n + 1, rn, opT, n);
    A[2] = ttps_c(1.0, n, rn, opT, n);
  }
}
static void ttps_BC(int k, int n, int which, double tau, int j, double *B,
                    double *C) {
  double T = tau * tau, opT = 1.0 + T, rn = pow(T / opT, n);
  int m = n - j + 1;
  if (k == 2 && which == 0) {
    *B = ttps_c((2 * m - 1) - T, m + 1, rn, opT, n);
    *C = ttps_c(2.0 * T - 2 * (m - 1), m + 1, rn, opT, n);
  } else if (k == 2) {
    *B = ttps_c(-2.0 * m, m + 1, rn, opT, n);
    *C = ttps_c((2 * m - 1) - T, m + 1, rn, opT, n);
  } else if (which == 0) {
    *B = ttps_c(m * (2 * m - 1) - 3 * m * T, m + 2, rn, opT, n);
    *C = ttps_c(-T * T + (5 * m - 2) * T - (m - 1) * (2 * m - 1), m + 2, rn,
                opT, n);
  } else {
    *B = ttps_c(m * T - m * (2 * m + 1), m + 2, rn, opT, n);
    *C = ttps_c(m * (2 * m - 1) - 3 * m * T, m + 2, rn, opT, n);
  }
}
// log[(1+x)/sqrt(1+x^2/tau^2)], in a form that stays bounded at large x
static double ttps_log_ratio(double x, double tau) {
  double v;
  if (x > (tau > 1.0 ? tau : 1.0)) {
    v = 1.0 / x;
    return log(tau) + log1p(v) - 0.5 * log1p(tau * v * tau * v);
  }
  return log1p(x) - 0.5 * log1p((x / tau) * (x / tau));
}
static double ttps_F_closed(double x, double tau, int k, int n) {
  double A[3], B, C, out, xq, r, L, lx, J, q;
  int j;
  ttps_A(k, n, 0, tau, A);
  xq = x < 1e100 ? x : 1e100;
  r = xq / tau;
  L = log1p(r * r);
  lx = log1p(x);
  out = A[0] * ttps_log_ratio(x, tau);
  for (j = 2; j <= k; j++)
    out -= A[j - 1] / (j - 1) * expm1(-(j - 1) * lx);
  for (j = 2; j <= n; j++) {
    ttps_BC(k, n, 0, tau, j, &B, &C);
    out -= B / (2 * (j - 1) * pow(tau, 2 * (j - 1))) * expm1(-(j - 1) * L);
  }
  J = atan(r) / tau;
  ttps_BC(k, n, 0, tau, 1, &B, &C);
  out += C * J;
  q = xq * xq + tau * tau;
  for (j = 1; j < n; j++) {
    J = xq * pow(1.0 / q, j) / (2 * j * tau * tau) +
        (2 * j - 1) / (2 * j * tau * tau) * J;
    ttps_BC(k, n, 0, tau, j + 1, &B, &C);
    out += C * J;
  }
  return out;
}
static double ttps_H_closed(double x, double tau, int k, int n) {
  double A[3], B, C, out, q, K;
  int j;
  ttps_A(k, n, 1, tau, A);
  out = A[0] * (log(tau) - ttps_log_ratio(x, tau));
  for (j = 2; j <= k; j++)
    out += A[j - 1] / ((j - 1) * pow(1.0 + x, j - 1));
  q = x * x + tau * tau;
  for (j = 2; j <= n; j++) {
    ttps_BC(k, n, 1, tau, j, &B, &C);
    out += B / (2 * (j - 1) * pow(q, j - 1));
  }
  K = atan2(tau, x) / tau;
  ttps_BC(k, n, 1, tau, 1, &B, &C);
  out += C * K;
  for (j = 1; j < n; j++) {
    K = (2 * j - 1) / (2 * j * tau * tau) * K -
        x / (2 * j * tau * tau * pow(q, j));
    ttps_BC(k, n, 1, tau, j + 1, &B, &C);
    out += C * K;
  }
  return out;
}
static double ttps_H_series(double x, double tau, int k, int n) {
  double v = 1.0 / x, w2 = (tau * v) * (tau * v), out = 0.0, w2j = 1.0, inner;
  int i, j;
  for (j = 0; j <= TTPS_NSERIES / 2; j++) {
    inner = 0.0;
    for (i = TTPS_NSERIES - 2 * j; i >= 0; i--)
      inner = inner * v + ttps_binom_neg(k, i) / (k + 2 * n - 1 + i + 2 * j);
    out += ttps_binom_neg(n, j) * w2j * inner;
    w2j *= w2;
  }
  return pow(v, k - 1) * pow(w2, n) * out;
}
// D_m = d_m mu^m: the Taylor coefficients of the enclosed-mass integrand,
// scaled by its radius of convergence mu = min(1,tau)
static void ttps_series_coeffs(double tau, double mu, int k, int n,
                               double *D) {
  double rho2 = (mu / tau) * (mu / tau), mu_p[TTPS_NSERIES + 1],
         rho2_p[TTPS_NSERIES / 2 + 1];
  int m, j;
  mu_p[0] = 1.0;
  rho2_p[0] = 1.0;
  for (m = 1; m <= TTPS_NSERIES; m++)
    mu_p[m] = mu_p[m - 1] * mu;
  for (j = 1; j <= TTPS_NSERIES / 2; j++)
    rho2_p[j] = rho2_p[j - 1] * rho2;
  for (m = 0; m <= TTPS_NSERIES; m++) {
    D[m] = 0.0;
    for (j = 0; j <= m / 2; j++)
      D[m] += ttps_binom_neg(k, m - 2 * j) * ttps_binom_neg(n, j) *
              mu_p[m - 2 * j] * rho2_p[j];
  }
}
// F(x)/x^p, p = 0, 1, 2
static double ttps_Fx(double x, double tau, int k, int n, int p) {
  double mu = tau < 1.0 ? tau : 1.0, D[TTPS_NSERIES + 1], u, P;
  int m;
  if (x < TTPS_SMALL_X * mu) {
    ttps_series_coeffs(tau, mu, k, n, D);
    u = x / mu;
    P = 0.0;
    for (m = TTPS_NSERIES; m >= 0; m--)
      P = P * u + D[m] / (m + 2);
    return p < 2 ? pow(x, 2 - p) * P : P;
  }
  return ttps_F_closed(x, tau, k, n) / pow(x, p);
}
static double ttps_H(double x, double tau, int k, int n) {
  double nu = tau > 1.0 ? tau : 1.0;
  if (x > TTPS_LARGE_X * nu)
    return ttps_H_series(x, tau, k, n);
  return ttps_H_closed(x, tau, k, n);
}
// rc(t): exp of the log rc spline, held at its end values outside tgrid
static double ttps_rc(double t, double *args) {
  int Nt = (int) *(args + 4);
  double *tgrid, *c, dt;
  int lo, hi, mid;
  if (Nt == 0)
    return *(args + 5);
  tgrid = args + 5;
  c = args + 5 + Nt;
  if (t < tgrid[0])
    t = tgrid[0];
  if (t > tgrid[Nt - 1])
    t = tgrid[Nt - 1];
  lo = 0;
  hi = Nt - 2;
  while (lo < hi) { // last interval with tgrid[i] <= t
    mid = (lo + hi + 1) / 2;
    if (tgrid[mid] <= t)
      lo = mid;
    else
      hi = mid - 1;
  }
  c += 4 * lo;
  dt = t - tgrid[lo];
  return exp(((c[0] * dt + c[1]) * dt + c[2]) * dt + c[3]);
}
double TruncatedTwoPowerSphericalPotentialrdens(
    double r, double t, struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  double a = *(args + 1);
  int k = (int) *(args + 2), n = (int) *(args + 3);
  double tau = ttps_rc(t, args) / a, x = r / a;
  return pow(1.0 + (x / tau) * (x / tau), -n) / (x * pow(1.0 + x, k)) /
         (4.0 * M_PI * a * a * a);
}
double TruncatedTwoPowerSphericalPotentialrevaluate(
    double r, double t, struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  double a = *(args + 1);
  int k = (int) *(args + 2), n = (int) *(args + 3);
  double tau = ttps_rc(t, args) / a, x = r / a;
  return -(ttps_Fx(x, tau, k, n, 1) + ttps_H(x, tau, k, n)) / a;
}
double TruncatedTwoPowerSphericalPotentialrforce(
    double r, double t, struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  double a = *(args + 1);
  int k = (int) *(args + 2), n = (int) *(args + 3);
  double tau = ttps_rc(t, args) / a, x = r / a;
  return -ttps_Fx(x, tau, k, n, 2) / (a * a);
}
double TruncatedTwoPowerSphericalPotentialr2deriv(
    double r, double t, struct potentialArg *potentialArgs) {
  double *args = potentialArgs->args;
  double a = *(args + 1);
  int k = (int) *(args + 2), n = (int) *(args + 3);
  double tau = ttps_rc(t, args) / a, x = r / a;
  double mu = tau < 1.0 ? tau : 1.0, D[TTPS_NSERIES + 1], u, P;
  int m;
  if (x < TTPS_SMALL_X * mu) {
    // the 1/x terms of 4 pi rho and -2 M / r^3 cancel analytically
    ttps_series_coeffs(tau, mu, k, n, D);
    u = x / mu;
    P = 0.0;
    for (m = TTPS_NSERIES; m >= 1; m--)
      P = P * u + D[m] * m / (m + 2);
    return P / mu / (a * a * a);
  }
  return (pow(1.0 + (x / tau) * (x / tau), -n) / (x * pow(1.0 + x, k)) -
          2.0 * ttps_F_closed(x, tau, k, n) / (x * x * x)) /
         (a * a * a);
}
