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

from ..backend import asarray_on_device, device_of, is_backend_array
from ..backend._namespaces import namespace_from_arrays, stop_gradient, under_jax_trace
from ..backend.interpolate import cubic_spline_coeffs


def slope_at_zero(js, ys, dys, xp=numpy):
    """Slope at J = 0 of the polynomial with value 0 there and the given
    values and slopes at the (one or two) actions js; ys may be 2D with the
    action along the first axis. With xp a jax/torch namespace, ys and dys are
    its arrays and the slope is differentiable in them (and in js)."""
    if xp is numpy:
        ys = numpy.atleast_1d(ys)
        dys = numpy.atleast_1d(dys)
    if len(js) == 1:
        return 2.0 * ys[0] / js[0] - dys[0]
    # quartic c1 J + c2 J^2 + c3 J^3 + c4 J^4 through (y, y') at two actions
    if xp is numpy:
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
    js = [
        j if is_backend_array(j) else asarray_on_device(xp, j, device_of(ys))
        for j in (js[0], js[1])
    ]
    one, zero = js[0] ** 0.0, js[0] * 0.0
    A = xp.stack(
        [
            xp.stack([j, j**2.0, j**3.0, j**4.0])
            if row % 2 == 0
            else xp.stack([one + zero * j, 2.0 * j, 3.0 * j**2.0, 4.0 * j**3.0])
            for j in js
            for row in (0, 1)
        ]
    )
    b = xp.stack([ys[0], dys[0], ys[1], dys[1]])
    return xp.reshape(xp.linalg.solve(A, xp.reshape(b, (4, -1)))[0], tuple(ys[0].shape))


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
        self._xp, self._c_on = numpy, {}
        xp = _ns(x, y, f, fx, fy)
        if xp is not numpy:
            return self._init_xp(xp, x, y, f, fx, fy)
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

    def _init_xp(self, xp, x, y, f, fx, fy):
        """The construction in jax/torch, all cells at once, differentiable
        in the tables and in the grid (whose L nodes, say, may depend on a
        potential's parameters); the second partials come from not-a-knot
        cubic splines, scipy CubicSpline's default"""
        dev = device_of(x, y, f, fx, fy)
        x, y, f, fx, fy = (
            t if is_backend_array(t) else asarray_on_device(xp, t, dev) * 1.0
            for t in (x, y, f, fx, fy)
        )
        self._xp, self._x, self._y = xp, x, y
        self._scalar = f.ndim == 2
        if self._scalar:
            f, fx, fy = f[..., None], fx[..., None], fy[..., None]
        nx, ny, k = f.shape
        fxy = _spline_knot_slopes(xp, y, fx, axis=1)
        fxx = _spline_knot_slopes(xp, x, fx, axis=0)
        fxxy = _spline_knot_slopes(xp, x, fxy, axis=0)
        hx = (x[1:] - x[:-1])[:, None, None]
        hy = (y[1:] - y[:-1])[None, :, None]
        # the rows (f, hx f_x, hx^2 f_xx) at x_i then at x_{i+1}, and the
        # columns: the values at y_j, y_{j+1}, then hy times the
        # y-derivatives there, for every cell at once: F[a, b, i, j, K]
        sc = (1.0, hx, hx * hx)
        vals, ders = [], []
        for sl in (slice(None, -1), slice(1, None)):
            vals.extend(t[sl] * s for t, s in zip((f, fx, fxx), sc))
            ders.extend(t[sl] * s for t, s in zip((fy, fxy, fxxy), sc))
        V, D = xp.stack(vals), xp.stack(ders)  # (6, nx-1, ny, k)
        F = xp.stack(
            [V[:, :, :-1], V[:, :, 1:], hy * D[:, :, :-1], hy * D[:, :, 1:]],
            axis=1,
        )  # (6, 4, nx-1, ny-1, k)
        Mq = asarray_on_device(xp, self._Mq, dev)
        Mc = asarray_on_device(xp, self._Mc, dev)
        self._c = xp.einsum("ab,bcijK,dc->ijadK", Mq, F, Mc)
        return None

    def _coeffs(self, xp, dev):
        """The coefficients (and the grid) in namespace xp on device dev: a
        numpy-built family's are copied there once"""
        if self._xp is not numpy:
            return self._x, self._y, self._c
        key = (getattr(xp, "__name__", ""), str(dev))
        if key in self._c_on:
            return self._c_on[key]
        out = tuple(
            asarray_on_device(xp, numpy.ascontiguousarray(t), dev)
            for t in (self._x, self._y, self._c)
        )
        if not under_jax_trace(*out):  # a trace's constants must not outlive it
            self._c_on[key] = out
        return out

    def ev(self, x, y, dx=0, dy=0):
        """The interpolant (dx = 0) or its first partial in x (dx = 1), and
        likewise in y, at the points (x, y) in jax/torch, differentiable in
        x and y and in the family's tables: arrays of any (common) shape
        give that shape for one table and the shape plus (k,) for a stack.
        Outside the grid the edge cells' polynomials extrapolate."""
        xp = _ns(x, y)
        dev = device_of(x, y) if xp is not numpy else device_of(self._c)
        xp = xp if xp is not numpy else self._xp
        xg, yg, c = self._coeffs(xp, dev)
        x, y = (
            t if is_backend_array(t) else asarray_on_device(xp, t, dev) * 1.0
            for t in (x, y)
        )
        x, y = xp.broadcast_arrays(x, y)
        nx, ny = xg.shape[0], yg.shape[0]
        i = xp.clip(
            xp.searchsorted(stop_gradient(xg), _dense(x), side="right") - 1,
            0,
            nx - 2,
        )
        j = xp.clip(
            xp.searchsorted(stop_gradient(yg), _dense(y), side="right") - 1,
            0,
            ny - 2,
        )
        hx, hy = xg[i + 1] - xg[i], yg[j + 1] - yg[j]
        ps = _power_basis(xp, (x - xg[i]) / hx, 6, dx, hx)
        pt = _power_basis(xp, (y - yg[j]) / hy, 4, dy, hy)
        v = xp.einsum("...a,...abk,...b->...k", ps, c[i, j], pt)
        return v[..., 0] if self._scalar else v

    def ev3(self, x, y):
        """The interpolant and its first partials in x and y at (x, y), as
        three ev calls would give them, in one pass (one cell lookup and one
        coefficient gather)"""
        xp = _ns(x, y)
        dev = device_of(x, y) if xp is not numpy else device_of(self._c)
        xp = xp if xp is not numpy else self._xp
        xg, yg, c = self._coeffs(xp, dev)
        x, y = (
            t if is_backend_array(t) else asarray_on_device(xp, t, dev) * 1.0
            for t in (x, y)
        )
        x, y = xp.broadcast_arrays(x, y)
        i, j = (
            xp.clip(
                xp.searchsorted(stop_gradient(g), _dense(t), side="right") - 1,
                0,
                g.shape[0] - 2,
            )
            for g, t in ((xg, x), (yg, y))
        )
        hx, hy = xg[i + 1] - xg[i], yg[j + 1] - yg[j]
        sv, tv = (x - xg[i]) / hx, (y - yg[j]) / hy
        cij = c[i, j]
        ps, pt = _power_basis(xp, sv, 6, 0, hx), _power_basis(xp, tv, 4, 0, hy)
        dps, dpt = _power_basis(xp, sv, 6, 1, hx), _power_basis(xp, tv, 4, 1, hy)
        out = (
            xp.einsum("...a,...abk,...b->...k", ps, cij, pt),
            xp.einsum("...a,...abk,...b->...k", dps, cij, pt),
            xp.einsum("...a,...abk,...b->...k", ps, cij, dpt),
        )
        return tuple(v[..., 0] for v in out) if self._scalar else out

    def __call__(self, x, y):
        """The interpolant's value at one point, cheaply (the derivatives
        and many points at once are points, points_dx and points_grad; on a
        backend, ev and ev3)"""
        if self._xp is not numpy or _ns(x, y) is not numpy:
            v = self.ev(x, y)
            xp = _ns(v)
            return xp.reshape(v, (1, 1)) if self._scalar else v[None, None, :]
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
        self._xp, self._c_on = numpy, {}
        xp = _ns(x, y, z, f, fx, fy, fz)
        if xp is not numpy:
            return self._init_xp(xp, x, y, z, f, fx, fy, fz)
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

    def _init_xp(self, xp, x, y, z, f, fx, fy, fz):
        """The construction in jax/torch, differentiable in the tables and
        in the grid; the estimated partials from not-a-knot cubic splines,
        scipy CubicSpline's default"""
        dev = device_of(x, y, z, f, fx, fy, fz)
        x, y, z, f, fx, fy, fz = (
            t if is_backend_array(t) else asarray_on_device(xp, t, dev) * 1.0
            for t in (x, y, z, f, fx, fy, fz)
        )
        self._xp, self._x, self._y, self._z = xp, x, y, z

        def d(tabs, axis, nodes):
            # the tables' splines along one axis in one solve (stacked on
            # the last axis, the stack's)
            k = tabs[0].shape[-1]
            cat = getattr(xp, "concat", None) or xp.concatenate
            out = _spline_knot_slopes(xp, nodes, cat(tabs, axis=-1), axis)
            return [out[..., i * k : (i + 1) * k] for i in range(len(tabs))]

        (fxy,) = d([fx], 1, y)
        fxz, fyz, fxyz = d([fx, fy, fxy], 2, z)
        fxx, fxxy, fxxz, fxxyz = d([fx, fxy, fxz, fxyz], 0, x)
        self._nodes = xp.stack(
            [f, fx, fxx, fy, fxy, fxxy, fz, fxz, fxxz, fyz, fxyz, fxxyz]
        )
        return None

    def _nodes_on(self, xp, dev):
        """The grid and the node data in namespace xp on device dev: a
        numpy-built family's are copied there once"""
        if self._xp is not numpy:
            return self._x, self._y, self._z, self._nodes
        key = (getattr(xp, "__name__", ""), str(dev))
        if key in self._c_on:
            return self._c_on[key]
        out = tuple(
            asarray_on_device(xp, numpy.ascontiguousarray(t), dev)
            for t in (self._x, self._y, self._z, self._nodes)
        )
        if not under_jax_trace(*out):  # a trace's constants must not outlive it
            self._c_on[key] = out
        return out

    def ev4(self, x, y, z):
        """The interpolant and its three first partials at the point (x, y, z)
        (0-d), in jax/torch, differentiable in the point and in the family's
        tables: four vectors of the stack's values"""
        xp = _ns(x, y, z)
        dev = device_of(x, y, z) if xp is not numpy else device_of(self._nodes)
        xp = xp if xp is not numpy else self._xp
        xg, yg, zg, nodes = self._nodes_on(xp, dev)
        x, y, z = (
            t if is_backend_array(t) else asarray_on_device(xp, t, dev) * 1.0
            for t in (x, y, z)
        )
        cells = []
        for g, t in ((xg, x), (yg, y), (zg, z)):
            i = xp.clip(
                xp.searchsorted(
                    stop_gradient(g), xp.reshape(_dense(t), (1,)), side="right"
                )[0]
                - 1,
                0,
                g.shape[0] - 2,
            )
            h = g[i + 1] - g[i]
            cells.append((i, h, (t - g[i]) / h))
        (i, hx, sv), (j, hy, tv), (l, hz, wv) = cells
        two = asarray_on_device(xp, numpy.arange(2), dev)
        blk = nodes[:, i + two][:, :, j + two][:, :, :, l + two]  # (12, 2, 2, 2, k)
        K = blk.shape[-1]
        B = xp.reshape(blk, (2, 2, 3, 2, 2, 2, K))  # (cz, cy, r, q, yn, zn, K)
        one = 1.0 + 0.0 * hx
        sr = xp.stack([one, hx, hx * hx])
        sy, sz = xp.stack([one, hy]), xp.stack([one, hz])
        B = (
            B
            * (sz[:, None, None] * sy[None, :, None] * sr[None, None, :])[
                ..., None, None, None, None
            ]
        )
        F = xp.reshape(xp.permute_dims(B, (3, 2, 1, 4, 0, 5, 6)), (6, 4, 4, K))
        Mq = asarray_on_device(xp, self._Mq, dev)
        Mc = asarray_on_device(xp, self._Mc, dev)
        ps, dps = (_power_basis(xp, sv, 6, k, hx) for k in (0, 1))
        pt, dpt = (_power_basis(xp, tv, 4, k, hy) for k in (0, 1))
        pw, dpw = (_power_basis(xp, wv, 4, k, hz) for k in (0, 1))
        wa, dwa = Mq.T @ ps, Mq.T @ dps
        wb, dwb = Mc.T @ pt, Mc.T @ dpt
        wc, dwc = Mc.T @ pw, Mc.T @ dpw
        return tuple(
            xp.einsum("a,b,c,abcK->K", A, Bv, C, F)
            for A, Bv, C in ((wa, wb, wc), (dwa, wb, wc), (wa, dwb, wc), (wa, wb, dwc))
        )

    @staticmethod
    def _cell(nodes, val):
        i = min(
            max(numpy.searchsorted(nodes, val, side="right") - 1, 0), len(nodes) - 2
        )
        h = nodes[i + 1] - nodes[i]
        return i, h, (val - nodes[i]) / h

    def __call__(self, x, y, z, dx=0, dy=0, dz=0):
        if self._xp is not numpy or _ns(x, y, z) is not numpy:
            out = self.ev4(x, y, z)
            v = out[1] if dx else (out[2] if dy else (out[3] if dz else out[0]))
            return v[None, None, :]
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


def _dense(t):
    """t without a derivative and in contiguous memory: a broadcast view
    makes torch's searchsorted copy it, with a warning"""
    return stop_gradient(t) * 1.0


def _ns(*args):
    """The namespace of the backend arrays among args, else numpy"""
    bk = tuple(a for a in args if is_backend_array(a))
    return namespace_from_arrays(bk) if bk else numpy


def _spline_knot_slopes(xp, x, t, axis):
    """The first derivative at the knots x of the not-a-knot cubic spline of
    the table t along the given axis (scipy's CubicSpline(x, t,
    axis=axis)(x, 1)), in jax/torch: every other axis is a separate spline"""
    tt = xp.moveaxis(t, axis, 0)
    shp = tuple(tt.shape)
    c = cubic_spline_coeffs(xp, x, xp.reshape(tt, (shp[0], -1)), bc="not-a-knot")
    h = xp.reshape(x[1:] - x[:-1], (-1, 1))
    last = 3.0 * c[0][-1:] * h[-1:] ** 2 + 2.0 * c[1][-1:] * h[-1:] + c[2][-1:]
    d = xp.reshape((getattr(xp, "concat", None) or xp.concatenate)([c[2], last]), shp)
    return xp.moveaxis(d, 0, axis)


def _power_basis(xp, s, n, deriv, h):
    """(1, s, ..., s^(n-1)) along a new last axis, or its derivative in the
    unscaled variable (deriv = 1: (0, 1, 2 s, ...) / h), by products (a power
    with a zero exponent has a NaN derivative at s = 0)"""
    pw = [xp.ones_like(s)]
    for _ in range(n - 1):
        pw.append(pw[-1] * s)
    if deriv == 0:
        return xp.stack(pw, axis=-1)
    return xp.stack([0.0 * s] + [m * pw[m - 1] / h for m in range(1, n)], axis=-1)
