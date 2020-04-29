/*****************************************************************************
 *	This code is based on work by Phillipe Thevenaz, which can be found
 * at http://bigwww.epfl.ch/thevenaz/interpolation/
 ****************************************************************************/ 
 
#ifdef __cplusplus
extern "C" {
#endif
#include	<float.h>
#include	<math.h>
#include	<stddef.h>
#include	<stdio.h>
#include	<stdlib.h>

//Macros to export functions in DLL on different OS
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
// Just do nothing?
#define EXPORT
#endif
/*--------------------------------------------------------------------------*/
void put_row(double *,long,double *,long);
EXPORT int samples_to_coefficients(double *,long,long);
#ifdef __cplusplus
}
#endif
