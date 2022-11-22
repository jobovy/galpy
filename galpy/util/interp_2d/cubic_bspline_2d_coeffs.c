/*****************************************************************************
 *	This code is based on work by Phillipe Thevenaz, which can be found
 * at http://bigwww.epfl.ch/thevenaz/interpolation/
 ****************************************************************************/

#include	"cubic_bspline_2d_coeffs.h"

/*--------------------------------------------------------------------------*/
static void convert_to_interpolation_coefficients
(
	double	c[],		/* input samples --> output coefficients */
	long	data_length,	/* number of samples or coefficients */
	double	z[],		/* poles */
	long	nb_poles,	/* number of poles */
	double	tolerance	/* admissible relative error */
);

/*--------------------------------------------------------------------------*/
static void get_column
(
	double	*data,		/* input image array */
	long	width,		/* width of the image */
	long	x,			/* x coordinate of the selected line */
	double	line[],		/* output linear array */
	long	height		/* length of the line and height of the image */
);

/*--------------------------------------------------------------------------*/
static void get_row
(
	double	*data,		/* input image array */
	long	y,			/* y coordinate of the selected line */
	double	line[],		/* output linear array */
	long	width		/* length of the line and width of the image */
);

/*--------------------------------------------------------------------------*/
static double initial_causal_coefficient
(
	double	c[],		/* coefficients */
	long	data_length,	/* number of coefficients */
	double	z,			/* actual pole */
	double	tolerance	/* admissible relative error */
);

/*--------------------------------------------------------------------------*/
static double initial_anticausal_coefficient
(
	double	c[],		/* coefficients */
	long	data_length,	/* number of samples or coefficients */
	double	z			/* actual pole */
);

/*--------------------------------------------------------------------------*/
static void put_column
(
	double	*data,		/* output image array */
	long	width,		/* width of the image */
	long	x,			/* x coordinate of the selected line */
	double	line[],		/* input linear array */
	long	height		/* length of the line and height of the image */
);

/*--------------------------------------------------------------------------*/
//void put_row now declared in header file

/*****************************************************************************
 *	Definition of static procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
static void convert_to_interpolation_coefficients
(
	double	c[],		/* input samples --> output coefficients */
	long	data_length,	/* number of samples or coefficients */
	double	z[],		/* poles */
	long	nb_poles,	/* number of poles */
	double	tolerance	/* admissible relative error */
)

{ /* begin convert_to_interpolation_coefficients */

	double	lambda = 1.0;
	long	n, k;

	/* special case required by mirror boundaries */
// LCOV_EXCL_START
	if (data_length == 1L) {
		return;
	}
// LCOV_EXCL_STOP
	/* compute the overall gain */
	for (k = 0L; k < nb_poles; k++) {
		lambda = lambda * (1.0 - z[k]) * (1.0 - 1.0 / z[k]);
	}
	/* apply the gain */
	for (n = 0L; n < data_length; n++) {
		c[n] *= lambda;
	}
	/* loop over all poles */
	for (k = 0L; k < nb_poles; k++) {
		/* causal initialization */
		c[0] = initial_causal_coefficient(c, data_length, z[k], tolerance);
		/* causal recursion */
		for (n = 1L; n < data_length; n++) {
			c[n] += z[k] * c[n - 1L];
		}
		/* anticausal initialization */
		c[data_length - 1L] = initial_anticausal_coefficient(c, data_length, z[k]);
		/* anticausal recursion */
		for (n = data_length - 2L; 0 <= n; n--) {
			c[n] = z[k] * (c[n + 1L] - c[n]);
		}
	}
} /* end convert_to_interpolation_coefficients */

/*--------------------------------------------------------------------------*/
static double initial_causal_coefficient
(
	double	c[],		/* coefficients */
	long	data_length,	/* number of coefficients */
	double	z,			/* actual pole */
	double	tolerance	/* admissible relative error */
)

{ /* begin initial_causal_coefficient */

	double	sum, zn, z2n, iz;
	long	n, horizon;

	/* this initialization corresponds to mirror boundaries */
	horizon = data_length;
	if (tolerance > 0.0) {
		horizon = (long)ceil(log(tolerance) / log(fabs(z)));
	}
	if (horizon < data_length) {
		/* accelerated loop */
		zn = z;
		sum = c[0];
		for (n = 1L; n < horizon; n++) {
			sum += zn * c[n];
			zn *= z;
		}
		return(sum);
	}
	else {
		/* full loop */
// LCOV_EXCL_START
		zn = z;
		iz = 1.0 / z;
		z2n = pow(z, (double)(data_length - 1L));
		sum = c[0] + z2n * c[data_length - 1L];
		z2n *= z2n * iz;
		for (n = 1L; n <= data_length - 2L; n++) {
			sum += (zn + z2n) * c[n];
			zn *= z;
			z2n *= iz;
		}
		return(sum / (1.0 - zn * zn));
// LCOV_EXCL_STOP
	}
} /* end initial_causal_coefficient */

/*--------------------------------------------------------------------------*/
static void get_column
(
	double	*data,		/* input image array */
	long	width,		/* width of the image */
	long	x,			/* x coordinate of the selected line */
	double	line[],		/* output linear array */
	long	height		/* length of the line */
)

{ /* begin get_column */

	long	y;

	data = data + (ptrdiff_t)x;
	for (y = 0L; y < height; y++) {
		line[y] = (double)*data;
		data += (ptrdiff_t)width;
	}
} /* end get_column */

/*--------------------------------------------------------------------------*/
static void get_row
(
	double	*data,		/* input image array */
	long	y,			/* y coordinate of the selected line */
	double	line[],		/* output linear array */
	long	width		/* length of the line */
)

{ /* begin get_row */

	long	x;

	data = data + (ptrdiff_t)(y * width);
	for (x = 0L; x < width; x++) {
		line[x] = (double)*data++;
	}
} /* end get_row */

/*--------------------------------------------------------------------------*/
static double initial_anticausal_coefficient
(
	double	c[],		/* coefficients */
	long	data_length,	/* number of samples or coefficients */
	double	z			/* actual pole */
)

{ /* begin initial_anticausal_coefficient */

	/* this initialization corresponds to mirror boundaries */
	return((z / (z * z - 1.0)) * (z * c[data_length - 2L] + c[data_length - 1L]));
} /* end initial_anticausal_coefficient */

/*--------------------------------------------------------------------------*/
static void put_column
(
	double	*data,		/* output image array */
	long	width,		/* width of the image */
	long	x,			/* x coordinate of the selected line */
	double	line[],		/* input linear array */
	long	height		/* length of the line and height of the image */
)

{ /* begin put_column */

	long	y;

	data = data + (ptrdiff_t)x;
	for (y = 0L; y < height; y++) {
		*data = (double)line[y];
		data += (ptrdiff_t)width;
	}
} /* end put_column */

/*--------------------------------------------------------------------------*/
void put_row
(
	double	*data,		/* output image array */
	long	y,			/* y coordinate of the selected line */
	double	line[],		/* input linear array */
	long	width		/* length of the line and width of the image */
)

{ /* begin put_row */

	long	x;

	data = data + (ptrdiff_t)(y * width);
	for (x = 0L; x < width; x++) {
		*data++ = (double)line[x];
	}
} /* end put_row */

/*****************************************************************************
 *	Definition of extern procedures
 ****************************************************************************/
/*--------------------------------------------------------------------------*/
extern int samples_to_coefficients
(
	double	*data,		/* in-place processing */
	long	width,		/* width of the image */
	long	height		/* height of the image */
)

{ /* begin SamplesToCoefficients */

	double	*line;
	double	pole[4];
	long	nb_poles;
	long	x, y;

	nb_poles = 1L;
	pole[0] = sqrt(3.0) - 2.0;

	/* convert the image samples into interpolation coefficients */
	/* in-place separable process, along x */
	line = (double *)malloc((size_t)(width * (long)sizeof(double)));
// LCOV_EXCL_START
	if (line == (double *)NULL) {
		printf("Row allocation failed\n");
		return(1);
	}
// LCOV_EXCL_STOP
	for (y = 0L; y < height; y++) {
		get_row(data, y, line, width);
		convert_to_interpolation_coefficients(line, width, pole, nb_poles, DBL_EPSILON);
		put_row(data, y, line, width);
	}
	free(line);

	/* in-place separable process, along y */
	line = (double *)malloc((size_t)(height * (long)sizeof(double)));
// LCOV_EXCL_START
	if (line == (double *)NULL) {
		printf("Column allocation failed\n");
		return(1);
	}
// LCOV_EXCL_STOP
	for (x = 0L; x < width; x++) {
		get_column(data, width, x, line, height);
		convert_to_interpolation_coefficients(line, height, pole, nb_poles, DBL_EPSILON);
		put_column(data, width, x, line, height);
	}
	free(line);

	return(0);
} /* end SamplesToCoefficients */
