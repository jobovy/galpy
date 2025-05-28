/*
  C code for actionAngle calculations
*/
#ifndef __GALPY_ACTIONANGLE_H__
#define __GALPY_ACTIONANGLE_H__
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
#include <Python.h>
#endif
#include <stdbool.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_spline.h>
#include "interp_2d.h"
/*
  Macro for dealing with potentially unused variables due to OpenMP
 */
/* If we're not using GNU C, elide __attribute__ if it doesn't exist*/
#ifndef __has_attribute      // Compatibility with non-clang compilers.
#define __has_attribute(x) 0
#endif
#if defined(__GNUC__) || __has_attribute(unused)
#  define UNUSED __attribute__((unused))
#else
#  define UNUSED /*NOTHING*/
#endif
/*
  Structure declarations
*/
struct pragmasolver{
  gsl_root_fsolver *s;
};
#ifdef __cplusplus
}
#endif
#endif /* actionAngle.h */
