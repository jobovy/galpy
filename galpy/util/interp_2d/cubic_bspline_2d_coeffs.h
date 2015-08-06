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

/*--------------------------------------------------------------------------*/
void put_row(double *,long,double *,long);
int samples_to_coefficients(double *,long,long);
#ifdef __cplusplus
}
#endif
