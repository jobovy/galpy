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
        j = numpy.asarray(j, dtype="float")
        if j.ndim == 0:
            return self._y0 + self._dy0 * (j - self._j0)
        # an array of actions: one row per action
        return self._y0 + numpy.multiply.outer(j - self._j0, self._dy0)

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
    Hermite's.  Called like a RectBivariateSpline for one value: ip(x,
    y)[0, 0], which is a number for one table and a vector for a stack of
    tables; the derivatives, and many points at once, with ip.points(x, y,
    dx=, dy=), ip.points_dx and ip.points_grad, one row per point."""

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

    def __call__(self, x, y):
        """The interpolant's value at one point, cheaply (the derivatives
        and many points at once are points, points_dx and points_grad)"""
        i = min(
            max(numpy.searchsorted(self._x, x, side="right") - 1, 0), len(self._x) - 2
        )
        j = min(
            max(numpy.searchsorted(self._y, y, side="right") - 1, 0), len(self._y) - 2
        )
        sv = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
        tv = (y - self._y[j]) / (self._y[j + 1] - self._y[j])
        v = numpy.einsum(
            "a,abk,b->k", sv ** numpy.arange(6), self._c[i, j], tv ** numpy.arange(4)
        )
        return v.reshape(1, 1) if self._scalar else v[None, None, :]

    def _cells(self, x, y):
        """The cell of each point and its normalized coordinates in it"""
        x = numpy.atleast_1d(numpy.asarray(x, dtype="float"))
        y = numpy.atleast_1d(numpy.asarray(y, dtype="float"))
        i = numpy.minimum(
            numpy.maximum(numpy.searchsorted(self._x, x, side="right") - 1, 0),
            len(self._x) - 2,
        )
        j = numpy.minimum(
            numpy.maximum(numpy.searchsorted(self._y, y, side="right") - 1, 0),
            len(self._y) - 2,
        )
        hx, hy = self._x[i + 1] - self._x[i], self._y[j + 1] - self._y[j]
        return i, j, (x - self._x[i]) / hx, (y - self._y[j]) / hy, hx, hy

    @staticmethod
    def _powers(s, h, deriv, n):
        """The row of unit-interval monomials s^a, a < n, or of their
        derivatives with respect to the unnormalized variable"""
        P = numpy.empty((len(s), n))
        P[:, 0] = 1.0
        for a in range(1, n):
            P[:, a] = P[:, a - 1] * s
        if not deriv:
            return P
        D = numpy.zeros_like(P)
        D[:, 1:] = numpy.arange(1, n) * P[:, :-1]
        return D / h[:, None]

    def _eval_cells(self, i, j, pss, pts):
        """sum_ab ps_a c_{ij,abk} pt_b for every point, one pair (ps, pt) per
        requested quantity, grouped by cell so that the coefficient block of
        each cell is read once however many points fall in it"""
        n = len(i)
        if n == 1:  # one point: its cell's block directly
            block = self._c[i[0], j[0]]
            outs = [
                numpy.einsum("a,abk,b->k", ps[0], block, pt[0])[None]
                for ps, pt in zip(pss, pts)
            ]
            return [o[:, 0] if self._scalar else o for o in outs]
        outs = [numpy.empty((n, self._c.shape[-1])) for _ in pss]
        cell = i * (len(self._y) - 1) + j
        order = numpy.argsort(cell, kind="stable")
        cells, starts = numpy.unique(cell[order], return_index=True)
        for c, lo, hi in zip(cells, starts, numpy.append(starts[1:], n)):
            idx = order[lo:hi]
            block = self._c[c // (len(self._y) - 1), c % (len(self._y) - 1)]
            for out, ps, pt in zip(outs, pss, pts):
                out[idx] = numpy.einsum("ma,abk,mb->mk", ps[idx], block, pt[idx])
        return [o[:, 0] if self._scalar else o for o in outs]

    def points(self, x, y, dx=0, dy=0):
        """The interpolant (or its first partial, dx=1 or dy=1) at the points
        (x_i, y_i), vectorized over the points: a vector for one table or an
        (n, k) array for a stack of k tables"""
        i, j, sv, tv, hx, hy = self._cells(x, y)
        ps = self._powers(sv, hx, dx != 0, 6)
        pt = self._powers(tv, hy, dy != 0, 4)
        return self._eval_cells(i, j, (ps,), (pt,))[0]

    def points_dx(self, x, y):
        """The interpolant and its first partial in x at the points (x_i, y_i),
        from one read of the coefficients"""
        i, j, sv, tv, hx, hy = self._cells(x, y)
        pt = self._powers(tv, hy, False, 4)
        return self._eval_cells(
            i,
            j,
            (self._powers(sv, hx, False, 6), self._powers(sv, hx, True, 6)),
            (pt, pt),
        )

    def points_grad(self, x, y):
        """The interpolant and both of its first partials at the points
        (x_i, y_i), from one read of the coefficients"""
        i, j, sv, tv, hx, hy = self._cells(x, y)
        ps, psx = self._powers(sv, hx, False, 6), self._powers(sv, hx, True, 6)
        pt, pty = self._powers(tv, hy, False, 4), self._powers(tv, hy, True, 4)
        return self._eval_cells(i, j, (ps, psx, ps), (pt, pt, pty))


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
        # of (as is, d/dy, d/dz, d^2/dy dz), stored node by node as
        # (nx, ny, nz, 12, k) so that a node's quantities are one block
        self._nodes = numpy.stack(
            [f, fx, fxx, fy, fxy, fxxy, fz, fxz, fxxz, fyz, fxyz, fxxyz], axis=3
        )

    @staticmethod
    def _cell(nodes, val):
        i = min(
            max(numpy.searchsorted(nodes, val, side="right") - 1, 0), len(nodes) - 2
        )
        h = nodes[i + 1] - nodes[i]
        return i, h, (val - nodes[i]) / h

    def batch_all(self, x, y, z):
        """The interpolant and its three first partials at many points at
        once, as four (npoints, k) arrays: the twelve quantities of each of
        a point's eight nodes are gathered as blocks and contracted with
        the Hermite weights one variable after the other, each contraction
        serving every output that shares it, in chunks that keep the
        gathered blocks to a few megabytes"""
        x, y, z = (numpy.asarray(q, dtype="float").ravel() for q in (x, y, z))

        def cells(nodes, vals, M):
            # the cell of every point and the Hermite weights of its nodes'
            # quantities as (npoints, 2, node, order): for the interpolant
            # and for its derivative, with the powers of the cell's width
            # that the node's derivatives carry
            i = numpy.clip(
                numpy.searchsorted(nodes, vals, side="right") - 1, 0, len(nodes) - 2
            )
            h = nodes[i + 1] - nodes[i]
            s, n = (vals - nodes[i]) / h, len(M)
            P = HermiteFamily2D._powers
            w = numpy.stack([P(s, h, False, n) @ M, P(s, h, True, n) @ M], axis=1)
            if n == 4:  # cubic: the values at both nodes, then the slopes
                w = w.reshape(-1, 2, 2, 2).transpose(0, 1, 3, 2)
            else:  # quintic: (f, f_x, f_xx) at the lower node, then the upper
                w = w.reshape(-1, 2, 2, 3)
            return i, w * numpy.stack([h**a for a in range(n // 2)], axis=1)[
                :, None, None, :
            ]

        i, A = cells(self._x, x, self._Mq)
        j, B = cells(self._y, y, self._Mc)
        l, C = cells(self._z, z, self._Mc)
        k, o = self._nodes.shape[-1], numpy.arange(2)
        out = numpy.empty((4, len(x), k))
        step = max(1, 500_000 // (96 * k))
        for lo in range(0, len(x), step):
            sl = slice(lo, lo + step)
            n = len(i[sl])
            # the blocks of the eight nodes, with z's nodes outermost, and
            # the quantities split as (d/dz order, d/dy order, d/dx order)
            G = self._nodes[
                i[sl, None, None, None] + o,
                j[sl, None, None, None] + o[:, None],
                l[sl, None, None, None] + o[:, None, None],
            ].reshape(n, 2, 2, 2, 2, 2, 3, k)
            G = numpy.ascontiguousarray(G.transpose(0, 1, 4, 2, 5, 3, 6, 7))
            # z, then y, then x: each product carries the interpolant and
            # the derivative in its variable
            G = C[sl].reshape(n, 2, 4) @ G.reshape(n, 4, 24 * k)
            G = B[sl].reshape(n, 1, 2, 4) @ G.reshape(n, 2, 4, 6 * k)
            G = A[sl].reshape(n, 1, 1, 2, 6) @ G.reshape(n, 2, 2, 6, k)
            out[:, sl] = G[:, 0, 0, 0], G[:, 0, 0, 1], G[:, 0, 1, 0], G[:, 1, 0, 0]
        return tuple(out)

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
        blk = numpy.moveaxis(self._nodes[i : i + 2, j : j + 2, l : l + 2], 3, 0)
        F = numpy.empty((6, 4, 4, blk.shape[-1]))
        for r, sr in enumerate((1.0, hx, hx * hx)):
            for cy, sy in enumerate((1.0, hy)):
                for cz, sz in enumerate((1.0, hz)):
                    F[r:6:3, 2 * cy : 2 * cy + 2, 2 * cz : 2 * cz + 2] = blk[
                        r + 3 * cy + 6 * cz
                    ] * (sr * sy * sz)
        return numpy.einsum("a,b,c,abcK->K", wa, wb, wc, F)[None, None, :]
