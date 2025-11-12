#include "xsf/hyp2f1.h"

extern "C" double hyp2f1(double a, double b, double c, double z) {
    return xsf::hyp2f1(a, b, c, z);
}
