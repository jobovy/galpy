#ifndef __INTEGRATEFULLORBIT_H__
#define __INTEGRATEFULLORBIT_H__
#ifdef __cplusplus
extern "C" {
#endif
#ifdef _WIN32
#include <Python.h>
#endif
#include <galpy_potentials.h>
typedef void (*orbint_callback_type)(); // Callback function
void parse_leapFuncArgs_Full(int, struct potentialArg *,int **,double **,tfuncs_type_arr *);
#ifdef _WIN32
// On Windows, *need* to define this function to allow the package to be imported
#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_libgalpy(void); // Python 3
#else
  PyMODINIT_FUNC initlibgalpy(void); // Python 2
#endif
#endif
//OpenMP
#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
static inline omp_int_t omp_get_thread_num(void) { return 0;}
static inline omp_int_t omp_get_max_threads(void) { return 1;}
#endif
#ifdef __cplusplus
}
#endif
#endif /* integrateFullOrbit.h */
