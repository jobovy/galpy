/*****************************************************************************
 *	This code is based the scipy version of cephes
 ****************************************************************************/ 
#ifdef __cplusplus
extern "C" {
#endif
  
#include	<math.h>
//#include	<stddef.h>
//#include	<stdio.h>
//#include	<stdlib.h>
//#include    <mconf.h>	

/*--------------------------------------------------------------------------*/
extern double	Gamma
(
    double	x			/* the value */
);

extern double	lgam
(
    double x
);


extern double	lgam_sgn
(
    double x,
    int *sign
);

extern double hyp2f1
(
	double a,
	double b,
	double c,
    double x			/* the value */
);

#ifdef __cplusplus
}
#endif
