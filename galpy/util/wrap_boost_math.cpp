#include <boost/math/special_functions/hypergeometric_2f1.hpp>

extern "C" double hyp2f1(double a, double b, double c, double z) {
    using boost::math::hypergeometric_2f1;
    return hypergeometric_2f1(a, b, c, z);
}
