#include "xsf/hyp2f1.h"
#include <complex>

extern "C" double hyp2f1(double a, double b, double c, double z) {
    // Use complex version for broader domain coverage (|z| > 1)
    // Cast real z to complex as scipy does
    std::complex<double> z_complex(z, 0.0);
    std::complex<double> result = xsf::hyp2f1(a, b, c, z_complex);
    return result.real();
}
