/*****************************************************************************
 *	This code is based on work by Phillipe Thevenaz, which can be found
 * at http://bigwww.epfl.ch/thevenaz/interpolation/
 ****************************************************************************/ 
#ifdef __cplusplus
extern "C" {
#endif
  
#include	<math.h>
#include	<stddef.h>
#include	<stdio.h>
#include	<stdlib.h>

/*--------------------------------------------------------------------------*/
extern double	cubic_bspline_2d_interpol
(
    double	*coeffs,	/* input B-spline array of coefficients */
    long	width,		/* width of the image */
    long	height,		/* height of the image */
    double	x,			/* x coordinate where to interpolate */
    double	y			/* y coordinate where to interpolate */
);

extern double	cubic_bspline_2d_interpol_dx
(
    double	*coeffs,	/* input B-spline array of coefficients */
    long	width,		/* width of the image */
    long	height,		/* height of the image */
    double	x,			/* x coordinate where to interpolate */
    double	y			/* y coordinate where to interpolate */
);

extern double	cubic_bspline_2d_interpol_dy
(
    double	*coeffs,	/* input B-spline array of coefficients */
    long	width,		/* width of the image */
    long	height,		/* height of the image */
    double	x,			/* x coordinate where to interpolate */
    double	y			/* y coordinate where to interpolate */
);
#ifdef __cplusplus
}
#endif
