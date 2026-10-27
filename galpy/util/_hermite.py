###############################################################################
# _hermite.py: Hermite interpolants of families of tori with exact slopes at
#   the nodes, shared by the inverse action-angle transformations: a single
#   node with its slope (the one-torus family), the slope at the harmonic
#   or circular edge of a family from the next nodes, and a tensor-product
#   Hermite interpolant on a rectangular grid, quintic in one variable and
#   cubic in the other
###############################################################################
import numpy
from scipy.interpolate import CubicSpline


def slope_at_zero(js, ys, dys):
    """Slope at J = 0 of the polynomial with value 0 there and the given
    values and slopes at the (one or two) actions js; ys may be 2D with the
    action along the first axis."""
    ys = numpy.atleast_1d(ys)
    dys = numpy.atleast_1d(dys)
    if len(js) == 1:
        return 2.0 * ys[0] / js[0] - dys[0]
    # quartic c1 J + c2 J^2 + c3 J^3 + c4 J^4 through (y, y') at two actions
    A = numpy.array(
        [
            [js[0], js[0] ** 2.0, js[0] ** 3.0, js[0] ** 4.0],
            [1.0, 2.0 * js[0], 3.0 * js[0] ** 2.0, 4.0 * js[0] ** 3.0],
            [js[1], js[1] ** 2.0, js[1] ** 3.0, js[1] ** 4.0],
            [1.0, 2.0 * js[1], 3.0 * js[1] ** 2.0, 4.0 * js[1] ** 3.0],
        ]
    )
    b = numpy.array([ys[0], dys[0], ys[1], dys[1]])
    return numpy.linalg.solve(A, b.reshape(4, -1))[0].reshape(numpy.shape(ys[0]))


class LinearHermite:
    """A single node with its slope, as the value and derivative of a
    linear function of the action: the one-torus family."""

    def __init__(self, j0, y0, dy0):
        self._j0, self._y0, self._dy0 = j0, numpy.array(y0), numpy.array(dy0)

    def __call__(self, j):
        return self._y0 + self._dy0 * (j - self._j0)

    def derivative(self):
        return LinearHermite(self._j0, self._dy0, 0.0 * self._dy0)


class HermiteFamily2D:
    """A tensor-product Hermite interpolant on a rectangular grid, quintic
    in the first variable and cubic in the second, of one table or of a
    stack of tables on the same grid.  The values and the first partials
    are prescribed at every node and reproduced exactly there, together
    with the first partial's derivative along the second variable; the
    second derivatives in the first variable (and their derivative along
    the second) are estimated by differentiating cubic splines of the
    prescribed first partials, which makes the interpolant's first
    derivative in that variable accurate to one order beyond a cubic
    Hermite's.  Called like a RectBivariateSpline: ip(x, y, dx=, dy=)[0, 0],
    which is a number for one table and a vector for a stack of tables."""

    # the coefficient matrices of the unit-interval Hermite polynomials:
    # quintic through (f, f', f'') at both ends, cubic through (f, f')
    _Mq = numpy.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [-10.0, -6.0, -1.5, 10.0, -4.0, 0.5],
            [15.0, 8.0, 1.5, -15.0, 7.0, -1.0],
            [-6.0, -3.0, -0.5, 6.0, -3.0, 0.5],
        ]
    )
    _Mc = numpy.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-3.0, 3.0, -2.0, -1.0],
            [2.0, -2.0, 1.0, 1.0],
        ]
    )

    def __init__(self, x, y, f, fx, fy):
        # f, fx, fy: (nx, ny) for one table or (nx, ny, k) for a stack of k
        # tables on the same grid, interpolated together
        self._x, self._y = numpy.asarray(x), numpy.asarray(y)
        nx, ny = len(x), len(y)
        f, fx, fy = (numpy.asarray(t, dtype="float") for t in (f, fx, fy))
        self._scalar = f.ndim == 2
        if self._scalar:
            f, fx, fy = f[..., None], fx[..., None], fy[..., None]
        k = f.shape[2]
        fxy = CubicSpline(self._y, fx, axis=1)(self._y, 1)
        fxx = CubicSpline(self._x, fx, axis=0)(self._x, 1)
        fxxy = CubicSpline(self._x, fxy, axis=0)(self._x, 1)
        self._c = numpy.empty((nx - 1, ny - 1, 6, 4, k))
        for i in range(nx - 1):
            hx = self._x[i + 1] - self._x[i]
            for j in range(ny - 1):
                hy = self._y[j + 1] - self._y[j]
                # rows: (f, hx f_x, hx^2 f_xx) at x_i then at x_{i+1};
                # columns: values at y_j, y_{j+1}, then hy times the
                # y-derivatives there
                F = numpy.empty((6, 4, k))
                for r, (tab, sc) in enumerate(((f, 1.0), (fx, hx), (fxx, hx * hx))):
                    for q, ii in enumerate((i, i + 1)):
                        F[r + 3 * q, 0] = tab[ii, j] * sc
                        F[r + 3 * q, 1] = tab[ii, j + 1] * sc
                for r, (tab, sc) in enumerate(
                    ((fy, hy), (fxy, hx * hy), (fxxy, hx * hx * hy))
                ):
                    for q, ii in enumerate((i, i + 1)):
                        F[r + 3 * q, 2] = tab[ii, j] * sc
                        F[r + 3 * q, 3] = tab[ii, j + 1] * sc
                self._c[i, j] = numpy.einsum("ab,bcK,dc->adK", self._Mq, F, self._Mc)

    def __call__(self, x, y, dx=0, dy=0):
        i = min(
            max(numpy.searchsorted(self._x, x, side="right") - 1, 0), len(self._x) - 2
        )
        j = min(
            max(numpy.searchsorted(self._y, y, side="right") - 1, 0), len(self._y) - 2
        )
        hx, hy = self._x[i + 1] - self._x[i], self._y[j + 1] - self._y[j]
        sv = (x - self._x[i]) / hx
        tv = (y - self._y[j]) / hy
        if dx == 0:
            ps = sv ** numpy.arange(6)
        else:
            ps = (
                numpy.array([0.0, 1.0, 2.0 * sv, 3.0 * sv**2, 4.0 * sv**3, 5.0 * sv**4])
                / hx
            )
        if dy == 0:
            pt = tv ** numpy.arange(4)
        else:
            pt = numpy.array([0.0, 1.0, 2.0 * tv, 3.0 * tv**2]) / hy
        v = numpy.einsum("a,abk,b->k", ps, self._c[i, j], pt)
        return v.reshape(1, 1) if self._scalar else v[None, None, :]
