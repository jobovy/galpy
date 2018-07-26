#ifndef __INTEGRATEFULLORBIT_H__
#define __INTEGRATEFULLORBIT_H__
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
#include <Python.h>
#endif
#include <galpy_potentials.h>
void parse_leapFuncArgs_Full(int, struct potentialArg *,int **,double **);
#ifdef _WIN32
// On Windows, *need* to define this function to allow the package to be imported
#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_galpy_integrate_c(void); // Python 3
#else
  PyMODINIT_FUNC initgalpy_integrate_c(void); // Python 2
#endif
#endif
#ifdef __cplusplus
}
#endif
#endif /* integrateFullOrbit.h */
