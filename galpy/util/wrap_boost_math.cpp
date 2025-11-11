#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <initializer_list>

extern "C" double hyp2f1(double a, double b, double c, double z) {
    // Use Boost's pFq function to compute 2F1(a, b; c; z)
    // pFq takes vectors for numerator and denominator parameters
    std::initializer_list<double> a_vec = {a, b};
    std::initializer_list<double> b_vec = {c};
    return boost::math::hypergeometric_pFq(a_vec, b_vec, z);
}
