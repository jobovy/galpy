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


class HermiteFamily3D:
    """A tensor-product Hermite interpolant on a rectangular grid in three
    variables, quintic in the first and cubic in the other two, of one table
    or of a stack of tables on the same grid.  The values and the three
    first partials are prescribed at every node and reproduced exactly
    there; the mixed partials the tensor product needs, and the second
    partials in the first variable (which make the interpolant's first
    derivative in that variable accurate to one order beyond a cubic
    Hermite's), are estimated by differentiating cubic splines of the
    prescribed first partials, as HermiteFamily2D does.  Called like
    ip(x, y, z, dx=, dy=, dz=)[0, 0], which is the vector of the stack's
    values."""

    _Mq, _Mc = HermiteFamily2D._Mq, HermiteFamily2D._Mc

    def __init__(self, x, y, z, f, fx, fy, fz):
        # f, fx, fy, fz: (nx, ny, nz, k) for a stack of k tables on the same
        # grid, interpolated together
        self._x, self._y, self._z = (numpy.asarray(t, dtype="float") for t in (x, y, z))
        f, fx, fy, fz = (numpy.asarray(t, dtype="float") for t in (f, fx, fy, fz))

        def d(tab, axis, nodes):
            return CubicSpline(nodes, tab, axis=axis)(nodes, 1)

        fxy, fxz, fyz = d(fx, 1, self._y), d(fx, 2, self._z), d(fy, 2, self._z)
        fxyz = d(fxy, 2, self._z)
        fxx, fxxy, fxxz = d(fx, 0, self._x), d(fxy, 0, self._x), d(fxz, 0, self._x)
        fxxyz = d(fxyz, 0, self._x)
        # the twelve quantities of a node, ordered as (f, f_x, f_xx) for each
        # of (as is, d/dy, d/dz, d^2/dy dz)
        self._nodes = numpy.stack(
            [f, fx, fxx, fy, fxy, fxxy, fz, fxz, fxxz, fyz, fxyz, fxxyz]
        )

    @staticmethod
    def _cell(nodes, val):
        i = min(
            max(numpy.searchsorted(nodes, val, side="right") - 1, 0), len(nodes) - 2
        )
        h = nodes[i + 1] - nodes[i]
        return i, h, (val - nodes[i]) / h

    def __call__(self, x, y, z, dx=0, dy=0, dz=0):
        i, hx, s = self._cell(self._x, x)
        j, hy, t = self._cell(self._y, y)
        l, hz, w = self._cell(self._z, z)
        if dx == 0:
            ps = s ** numpy.arange(6)
        else:
            ps = (
                numpy.array([0.0, 1.0, 2.0 * s, 3.0 * s**2, 4.0 * s**3, 5.0 * s**4])
                / hx
            )
        pt = (
            t ** numpy.arange(4)
            if dy == 0
            else numpy.array([0.0, 1.0, 2.0 * t, 3.0 * t**2]) / hy
        )
        pw = (
            w ** numpy.arange(4)
            if dz == 0
            else numpy.array([0.0, 1.0, 2.0 * w, 3.0 * w**2]) / hz
        )
        wa, wb, wc = self._Mq.T @ ps, self._Mc.T @ pt, self._Mc.T @ pw
        # the cell's data tensor: rows (f, hx f_x, hx^2 f_xx) at x_i then at
        # x_{i+1}; columns the values at y_j, y_{j+1}, then hy times the
        # y-derivatives there; likewise along z
        blk = self._nodes[:, i : i + 2, j : j + 2, l : l + 2]
        F = numpy.empty((6, 4, 4, blk.shape[-1]))
        for r, sr in enumerate((1.0, hx, hx * hx)):
            for cy, sy in enumerate((1.0, hy)):
                for cz, sz in enumerate((1.0, hz)):
                    F[r:6:3, 2 * cy : 2 * cy + 2, 2 * cz : 2 * cz + 2] = blk[
                        r + 3 * cy + 6 * cz
                    ] * (sr * sy * sz)
        return numpy.einsum("a,b,c,abcK->K", wa, wb, wc, F)[None, None, :]
