#ifndef __GALPY_INCOMPLETE_BETA_H__
#define __GALPY_INCOMPLETE_BETA_H__
#ifdef __cplusplus
extern "C" {
#endif
// 2F1(1, b; c; z) for b > -1, c > 0, 0 <= z < 1
double galpy_hyp2f1_1(double b, double c, double z);
// the split point of galpy_incomplete_beta: (p+1)/(p+q+2), at most 0.9
double galpy_incomplete_beta_split(double p, double q);
// B_z(p, q) for p > 0, q > -1, 0 <= z < 1, given s = 1 - z (exact)
double galpy_incomplete_beta(double p, double q, double z, double s);
#ifdef __cplusplus
}
#endif
#endif /* incomplete_beta.h */
