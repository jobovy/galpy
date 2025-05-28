/*****************************************************************************
 *	This code is based on work by Phillipe Thevenaz, which can be found
 * at http://bigwww.epfl.ch/thevenaz/interpolation/
 ****************************************************************************/

#include	"cubic_bspline_2d_interpol.h"

/*--------------------------------------------------------------------------*/
extern double	cubic_bspline_2d_interpol
(
    double	*coeffs,	/* input B-spline array of coefficients */
    long	width,		/* width of the image */
    long	height,		/* height of the image */
    double	x,			/* x coordinate where to interpolate */
    double	y			/* y coordinate where to interpolate */
)

{ /* begin InterpolatedValue */

    int spline_degree = 3;
	//double	*p;
	long	x_index[4], y_index[4];
	double	x_weight[4], y_weight[4];

	double	interpolated;
	double	w/*, w2, w4, t, t0, t1*/;

	long	width2 = 2L * width - 2L, height2 = 2L * height - 2L;
	long	i, j, k;

	/* compute the interpolation indexes: floor(x) + {-1,0,1,2}, floor(y) + {-1,0,1,2} */
	i = (long)floor(x) - spline_degree / 2L;
	j = (long)floor(y) - spline_degree / 2L;
	for (k = 0L; k <= spline_degree; k++)
	{
		x_index[k] = i++;
		y_index[k] = j++;
	}

	/* compute the interpolation weights */
	/* x */
	w = x - (double)x_index[1];
	x_weight[3] = (1.0 / 6.0) * w * w * w;
	x_weight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - x_weight[3];
	x_weight[2] = w + x_weight[0] - 2.0 * x_weight[3];
	x_weight[1] = 1.0 - x_weight[0] - x_weight[2] - x_weight[3];
	/* y */
	w = y - (double)y_index[1];
	y_weight[3] = (1.0 / 6.0) * w * w * w;
	y_weight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - y_weight[3];
	y_weight[2] = w + y_weight[0] - 2.0 * y_weight[3];
	y_weight[1] = 1.0 - y_weight[0] - y_weight[2] - y_weight[3];

	/* apply the mirror boundary conditions */
    for (k = 0L; k <= spline_degree; k++)
    {
        x_index[k] = (width == 1L) ? (0L) : ((x_index[k] < 0L) ? (-x_index[k] - width2 * ((-x_index[k]) / width2)) : (x_index[k] - width2 * (x_index[k] / width2)));
        if (width <= x_index[k])
        {
            x_index[k] = width2 - x_index[k];
        }
        y_index[k] = (height == 1L) ? (0L) : ((y_index[k] < 0L) ?(-y_index[k] - height2 * ((-y_index[k]) / height2)) : (y_index[k] - height2 * (y_index[k] / height2)));
        if (height <= y_index[k])
        {
            y_index[k] = height2 - y_index[k];
        }
    }

	/* perform interpolation */
	interpolated = 0.0;
	/*for (j = 0L; j <= spline_degree; j++) {
		p = coeffs + (ptrdiff_t)(y_index[j] * width);
		w = 0.0;
		for (i = 0L; i <= spline_degree; i++) {
			w += x_weight[i] * p[x_index[i]];
		}
		interpolated += y_weight[j] * w;
	}*/
	for(i=0L; i<=spline_degree; i++)
	{
	    for(j=0L; j<=spline_degree; j++)
	    {
	        interpolated += coeffs[x_index[i]*height+y_index[j]] * x_weight[i] * y_weight[j];
	    }
    }

	return(interpolated);
}
// LCOV_EXCL_START
/*--------------------------------------------------------------------------*/
extern double	cubic_bspline_2d_interpol_dx
(
    double	*coeffs,	/* input B-spline array of coefficients */
    long	width,		/* width of the image */
    long	height,		/* height of the image */
    double	x,			/* x coordinate where to interpolate */
    double	y			/* y coordinate where to interpolate */
)

{ /* begin InterpolatedValue */

    int spline_degree = 3;
	//double	*p;
	long	x_index[3], y_index[4];
	double	x_weight[3], y_weight[4];

	double	interpolated;
	double	w/*, w2, w4, t, t0, t1*/;

	long	width2 = 2L * width - 2L, height2 = 2L * height - 2L;
	long	i, j, k;

	/* compute the interpolation indexes */
	/* here we have to calculate B^2(x-j+0.5) */
	i = (long)floor(x+1) - (spline_degree-1) / 2L;
	j = (long)floor(y) - spline_degree / 2L;
	for (k = 0L; k <= spline_degree; k++)
	{
	    if(k<spline_degree)
	    {
		    x_index[k] = i++;
		}
		y_index[k] = j++;

	}

	/* compute the interpolation weights */
	/* x + 0.5 */
	w = x +0.5 - (double)x_index[1];
	x_weight[1] = 3.0 / 4.0 - w * w;
	x_weight[2] = (1.0 / 2.0) * (w - x_weight[1] + 1.0);
	x_weight[0] = 1.0 - x_weight[1] - x_weight[2];
	/* y */
	w = y - (double)y_index[1];
	y_weight[3] = (1.0 / 6.0) * w * w * w;
	y_weight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - y_weight[3];
	y_weight[2] = w + y_weight[0] - 2.0 * y_weight[3];
	y_weight[1] = 1.0 - y_weight[0] - y_weight[2] - y_weight[3];

	/* apply the mirror boundary conditions */
    for (k = 0L; k <= spline_degree; k++)
    {
        if(k<spline_degree)
        {
            x_index[k] = (width == 1L) ? (0L) : ((x_index[k] < 0L) ? (-x_index[k] - width2 * ((-x_index[k]) / width2)) : (x_index[k] - width2 * (x_index[k] / width2)));
            if (width <= x_index[k])
            {
                x_index[k] = width2 - x_index[k];
            }
        }

        y_index[k] = (height == 1L) ? (0L) : ((y_index[k] < 0L) ?(-y_index[k] - height2 * ((-y_index[k]) / height2)) : (y_index[k] - height2 * (y_index[k] / height2)));
        if (height <= y_index[k])
        {
            y_index[k] = height2 - y_index[k];
        }
    }

	/* perform interpolation */
	interpolated = 0.0;
	/*for (j = 0L; j <= spline_degree; j++) {
		p = coeffs + (ptrdiff_t)(y_index[j] * width);
		w = 0.0;
		for (i = 0L; i < spline_degree; i++) {
			w += (p[x_index[i]] - p[x_index[i-1]]) * x_weight[i];
		}
		interpolated += y_weight[j] * w;
	}*/

	for(i=0L; i<spline_degree; i++)
	{
	    for(j=0L; j<=spline_degree; j++)
	    {
	      interpolated += ( coeffs[x_index[i]*width+y_index[j]] - coeffs[(x_index[i]-1)*width+y_index[j]] ) * x_weight[i] * y_weight[j]; //these widths might have to be heights, like in the function above
	    }
    }

	return(interpolated);
}

/*--------------------------------------------------------------------------*/
extern double	cubic_bspline_2d_interpol_dy
(
    double	*coeffs,	/* input B-spline array of coefficients */
    long	width,		/* width of the image */
    long	height,		/* height of the image */
    double	x,			/* x coordinate where to interpolate */
    double	y			/* y coordinate where to interpolate */
)

{ /* begin InterpolatedValue */

    int spline_degree = 3;
	//double	*p;
	long	x_index[4], y_index[3];
	double	x_weight[4], y_weight[3];

	double	interpolated;
	double	w/*, w2, w4, t, t0, t1*/;

	long	width2 = 2L * width - 2L, height2 = 2L * height - 2L;
	long	i, j, k;

	/* compute the interpolation indexes */
	/* here we have to calculate B^2(y-j+0.5) */
	i = (long)floor(x) - spline_degree / 2L;
	j = (long)floor(y+1) - (spline_degree-1) / 2L;
	for (k = 0L; k <= spline_degree; k++)
	{
	    x_index[k] = i++;
	    if(k<spline_degree)
	    {
		    y_index[k] = j++;
		}

	}

	/* compute the interpolation weights */
	/* x */
	w = x - (double)x_index[1];
	x_weight[3] = (1.0 / 6.0) * w * w * w;
	x_weight[0] = (1.0 / 6.0) + (1.0 / 2.0) * w * (w - 1.0) - x_weight[3];
	x_weight[2] = w + x_weight[0] - 2.0 * x_weight[3];
	x_weight[1] = 1.0 - x_weight[0] - x_weight[2] - x_weight[3];
	/* y */
	w = y +0.5 - (double)y_index[1];
	y_weight[1] = 3.0 / 4.0 - w * w;
	y_weight[2] = (1.0 / 2.0) * (w - y_weight[1] + 1.0);
	y_weight[0] = 1.0 - y_weight[1] - y_weight[2];

	/* apply the mirror boundary conditions */
    for (k = 0L; k <= spline_degree; k++)
    {

        x_index[k] = (width == 1L) ? (0L) : ((x_index[k] < 0L) ? (-x_index[k] - width2 * ((-x_index[k]) / width2)) : (x_index[k] - width2 * (x_index[k] / width2)));
        if (width <= x_index[k])
        {
            x_index[k] = width2 - x_index[k];
        }

        if(k<spline_degree)
        {
            y_index[k] = (height == 1L) ? (0L) : ((y_index[k] < 0L) ?(-y_index[k] - height2 * ((-y_index[k]) / height2)) : (y_index[k] - height2 * (y_index[k] / height2)));
            if (height <= y_index[k])
            {
                y_index[k] = height2 - y_index[k];
            }
        }
    }

	/* perform interpolation */
	interpolated = 0.0;
	/*for (j = 0L; j <= spline_degree; j++) {
		p = coeffs + (ptrdiff_t)(y_index[j] * width);
		w = 0.0;
		for (i = 0L; i < spline_degree; i++) {
			w += (p[x_index[i]] - p[x_index[i-1]]) * x_weight[i];
		}
		interpolated += y_weight[j] * w;
	}*/

	for(i=0L; i<=spline_degree; i++)
	{
	    for(j=0L; j<spline_degree; j++)
	    {
	        interpolated += ( coeffs[x_index[i]*width+y_index[j]] - coeffs[x_index[i]*width+y_index[j]-1] ) * x_weight[i] * y_weight[j]; //these widths might have to be heights, like in the function above
	    }
    }

	return(interpolated);
}
// LCOV_EXCL_STOP
