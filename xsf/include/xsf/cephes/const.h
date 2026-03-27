/* Translated into C++ by SciPy developers in 2024.
 * Original header with Copyright information appears below.
 *
 * Since we support only IEEE-754 floating point numbers, conditional logic
 * supporting other arithmetic types has been removed.
 */

/*
 *
 *
 *                                                   const.c
 *
 *     Globally declared constants
 *
 *
 *
 * SYNOPSIS:
 *
 * extern double nameofconstant;
 *
 *
 *
 *
 * DESCRIPTION:
 *
 * This file contains a number of mathematical constants and
 * also some needed size parameters of the computer arithmetic.
 * The values are supplied as arrays of hexadecimal integers
 * for IEEE arithmetic, and in a normal decimal scientific notation for
 * other machines.  The particular notation used is determined
 * by a symbol (IBMPC, or UNK) defined in the include file
 * mconf.h.
 *
 * The default size parameters are as follows.
 *
 * For UNK mode:
 * MACHEP =  1.38777878078144567553E-17       2**-56
 * MAXLOG =  8.8029691931113054295988E1       log(2**127)
 * MINLOG = -8.872283911167299960540E1        log(2**-128)
 *
 * For IEEE arithmetic (IBMPC):
 * MACHEP =  1.11022302462515654042E-16       2**-53
 * MAXLOG =  7.09782712893383996843E2         log(2**1024)
 * MINLOG = -7.08396418532264106224E2         log(2**-1022)
 *
 * The global symbols for mathematical constants are
 * SQ2OPI =  7.9788456080286535587989E-1      sqrt( 2/pi )
 * LOGSQ2 =  3.46573590279972654709E-1        log(2)/2
 * THPIO4 =  2.35619449019234492885           3*pi/4
 *
 * These lists are subject to change.
 */
/*                                                     const.c */

/*
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1995 by Stephen L. Moshier
 */
#pragma once

namespace xsf {
namespace cephes {
    namespace detail {
        constexpr std::uint64_t MAXITER = 500;
        constexpr double MACHEP = 1.1102230246251565404236316680908203125E-16;      // 2**-53
        constexpr double MAXLOG = 7.097827128933839730962063185871E2;               // log(DBL_MAX)
        constexpr double MINLOG = -7.451332191019412076235245305675398106812E2;     // log(2**(-1075))
        constexpr double SQRT1OPI = 5.641895835477562869480794515607725858441E-1;   // sqrt(1/pi)
        constexpr double SQRT2OPI = 7.978845608028653558798921198687637369517E-1;   // sqrt(2/pi)
        constexpr double SQRT2PI = 2.506628274631000502415765284811045253007;       // sqrt(2*pi)
        constexpr double LOGSQ2 = 3.465735902799726547086160607290882840378E-1;     // log(2)/2
        constexpr double THPIO4 = 2.356194490192344928846982537459627163148;        // 3*pi/4
        constexpr double SQRT3 = 1.732050807568877293527446341505872366943;         // sqrt(3)
        constexpr double PI180 = 1.745329251994329576923690768488612713443E-2;      // pi/180
        constexpr double LOGPI = 1.144729885849400174143427351353058711647;         // log(pi)
        constexpr double LOGSQRT2PI = 9.189385332046727417803297364056176398614E-1; // log(sqrt(2*pi))
        constexpr double MAXGAM = 171.624376956302725; // Largest x such that Gamma(x) is finite

        // Following two added by SciPy developers.
        // Euler's constant
        constexpr double SCIPY_EULER = 0.577215664901532860606512090082402431;
        // e as long double
        constexpr long double SCIPY_El = 2.718281828459045235360287471352662498L;
    } // namespace detail
} // namespace cephes
} // namespace xsf
