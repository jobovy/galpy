###############################################################################
#   galpy.backend.interpolate: backend-agnostic interpolation.
#
#   numpy stays byte-identical to the scipy splines (the numpy code paths call
#   the fitted scipy spline / RectBivariateSpline.ev directly, never the xp
#   evaluators here); jax/torch evaluate the same piecewise polynomials natively
#   so the result is exactly autodifferentiable. Two complementary capabilities:
#
#     mode 1 (frozen, eval-point-differentiable): a scipy spline fitted once at
#       setup is converted to a power-basis piecewise-polynomial table
#       (``spline_to_ppoly`` / ``rect_bivariate_to_ppoly``); ``eval_ppoly`` /
#       ``eval_rect_ppoly`` then evaluate it through namespace ops, so the value
#       is differentiable w.r.t. the EVALUATION POINT. This is what backs
#       interpSphericalPotential and interpRZPotential.
#
#     mode 2 (in-backend construction, y-differentiable): ``cubic_spline_coeffs``
#       builds the piecewise-cubic coefficients from (x, y) ENTIRELY in xp (a
#       tridiagonal/linear solve via ``xp.linalg.solve``), so the spline is
#       differentiable w.r.t. the y-VALUES (interpax / jax-cosmo style). This is
#       what lets d(orbit)/d(param) flow through a parameter-dependent table
#       (e.g. the dynamical-friction sigma_r(r) table). ``interp_linear`` is the
#       trivially-differentiable degree-1 option.
#
#   Every ``xp.where`` here dead-branch-guards both sides (the namespaces
#   evaluate both branches eagerly): the unused branch's argument is clamped
#   into a safe region so no inf/nan poisons reverse-mode autodiff, exactly as
#   in galpy.backend.special._fallback.bessel_k._k01.
#
#   NO internal jit: these functions are jit-COMPATIBLE (pure namespace ops,
#   Python-level loops over the *static* polynomial degree only), so a caller
#   may jit their galpy-using code; galpy never jits internally.
###############################################################################
import numpy
from scipy import interpolate as _scipy_interpolate

from ._namespaces import (
    asarray_on_device,
    device_of,
    is_backend_array,
)

__all__ = [
    "spline_to_ppoly",
    "eval_ppoly",
    "cubic_spline_coeffs",
    "eval_cubic",
    "interp_linear",
    "rect_bivariate_to_ppoly",
    "eval_rect_ppoly",
    "Spline1D",
    "Spline2D",
]


###############################################################################
#   (1) Frozen 1D piecewise-power evaluator (promoted from
#       interpSphericalPotential.py; VERBATIM, plus an ``extrapolate`` knob).
###############################################################################
def spline_to_ppoly(spl):
    """Convert a FITPACK spline to de-duplicated piecewise-power coefficients.

    Returns ``(x, c)`` with ``x`` the distinct breakpoints (shape ``(m+1,)``)
    and ``c`` the power-basis coefficients (shape ``(k+1, m)``) such that on
    ``x[i] <= r < x[i+1]`` the spline is ``sum_j c[j, i] * (r - x[i])**(k-j)``.
    This is the exact piecewise-polynomial representation of the spline (scipy's
    ``PPoly.from_spline``), with the zero-width intervals coming from FITPACK's
    repeated boundary knots dropped so that interval lookup by ``searchsorted``
    is unambiguous. Called once at setup (init-time numpy/scipy is fine); the
    coefficients then feed the backend-agnostic ``eval_ppoly`` below.
    """
    ppoly = _scipy_interpolate.PPoly.from_spline(spl._eval_args)
    keep = numpy.diff(ppoly.x) > 0.0
    return numpy.append(ppoly.x[:-1][keep], ppoly.x[-1]), ppoly.c[:, keep]


def eval_ppoly(xp, x, c, r, *, extrapolate=True):
    """Evaluate a piecewise polynomial in the power basis at ``r``.

    ``(x, c)`` are as returned by ``spline_to_ppoly`` (or ``cubic_spline_coeffs``
    -- same layout); the evaluation (interval lookup by ``xp.searchsorted`` +
    Horner) uses only namespace operations, so the spline value is computed
    natively -- and is exactly autodifferentiable -- under jax/torch.
    Mathematically this is the same piecewise polynomial as the scipy spline
    (agreement at the ~1 ulp level); the numpy code paths keep calling the scipy
    splines directly and never come through here.

    ``extrapolate`` selects the out-of-range behaviour, matching the
    ``InterpolatedUnivariateSpline`` ``ext`` modes callers use:

    - ``True`` (default, scipy ``ext=0``): the edge polynomial is evaluated for
      ``r`` outside ``[x[0], x[-1]]`` (finite polynomial extrapolation). This is
      the original interpSphericalPotential behaviour and keeps the dead side of
      the callers' ``xp.where`` branch selections NaN-free under autodiff.
    - ``'clip'``: the evaluation point is clamped to ``[x[0], x[-1]]`` before
      evaluation, so out-of-range ``r`` returns the corresponding edge VALUE.
    - ``'const'`` (or the integer ``3``, scipy ``ext=3``): constant beyond the
      ends, i.e. the edge value -- realized identically to ``'clip'`` (clamp the
      evaluation point), since for the spline VALUE constant-beyond and
      clamp-the-point coincide.
    """
    # knots/coefficients stay float64 (precision is the point; the callers
    # exit-cast) but must live on the input's device (CUDA support)
    dev = device_of(r)
    xb = asarray_on_device(xp, x, dev)
    cb = asarray_on_device(xp, c, dev)
    if extrapolate is not True:
        if extrapolate not in ("clip", "const", 3):
            raise ValueError(
                "eval_ppoly extrapolate must be True, 'clip', 'const', or 3; "
                f"got {extrapolate!r}"
            )
        # 'clip' and 'const' both return the edge VALUE outside the range, which
        # is the edge polynomial evaluated at the clamped point -> clamp r into
        # [x[0], x[-1]]. xp.clip on both ends is itself dead-branch-safe (no
        # division/log), so this stays AD-friendly.
        r = xp.clip(r, xb[0], xb[-1])
    idx = xp.clip(xp.searchsorted(xb, r, side="right") - 1, 0, cb.shape[1] - 1)
    dr = r - xb[idx]
    out = cb[0, idx]
    for j in range(1, cb.shape[0]):
        out = out * dr + cb[j, idx]
    return out


###############################################################################
#   (2) Differentiable in-backend 1D spline construction (mode 2).
###############################################################################
def cubic_spline_coeffs(xp, x, y, bc="natural"):
    """Build piecewise-cubic power-basis coefficients from ``(x, y)`` in ``xp``.

    Returns ``c`` of shape ``(4, n-1)`` in the same layout as
    ``spline_to_ppoly`` -- on ``x[i] <= r < x[i+1]`` the spline is
    ``sum_j c[j, i] * (r - x[i])**(3-j)`` -- so it feeds straight into
    ``eval_ppoly``/``eval_cubic``. The whole construction (a tridiagonal linear
    system for the second derivatives, solved with ``xp.linalg.solve``) is built
    from namespace operations, so the coefficients -- and hence the spline value
    -- are DIFFERENTIABLE w.r.t. the ``y`` values. This is the capability the
    frozen scipy PPoly cannot provide: a parameter-dependent table (e.g. the
    dynamical-friction ``sigma_r(r)`` table) becomes differentiable in its
    parameters.

    ``bc`` selects the end conditions:

    - ``'natural'`` (default): zero second derivative at both ends (the
      interpax / jax-cosmo default).
    - ``'not-a-knot'``: continuous third derivative across the first/last
      interior knot (scipy ``CubicSpline``'s default), for byte-for-byte
      comparison against scipy.

    ``x`` must be strictly increasing. ``x`` may be a frozen numpy array (its
    spacing is just geometry); differentiability is in ``y``.
    """
    dev = device_of(y, x)
    xb = asarray_on_device(xp, numpy.asarray(x), dev) * 1.0
    yb = xp.astype(y, xb.dtype) if hasattr(xp, "astype") else y * 1.0
    n = xb.shape[0]
    if n < 3:
        raise ValueError("cubic_spline_coeffs requires at least 3 points")
    h = xb[1:] - xb[:-1]  # (n-1,)
    # slopes of the secants
    dslope = (yb[1:] - yb[:-1]) / h  # (n-1,)

    # Tridiagonal system A M = rhs for the second derivatives M (length n).
    # Interior rows i=1..n-2:  h[i-1] M[i-1] + 2(h[i-1]+h[i]) M[i] + h[i] M[i+1]
    #                          = 6 (dslope[i] - dslope[i-1]).
    # Build the dense (n, n) A from the *geometry only* (x), so A is a constant
    # w.r.t. y; rhs carries the y-dependence (and hence the gradient). Assembled
    # with numpy on x (init-time geometry), then placed on-device as a constant.
    A = numpy.zeros((n, n))
    hnp = numpy.asarray(numpy.diff(numpy.asarray(x, dtype=float)))
    for i in range(1, n - 1):
        A[i, i - 1] = hnp[i - 1]
        A[i, i] = 2.0 * (hnp[i - 1] + hnp[i])
        A[i, i + 1] = hnp[i]
    if bc == "natural":
        # zero second derivative at the ends: M[0] = M[n-1] = 0.
        A[0, 0] = 1.0
        A[n - 1, n - 1] = 1.0
    elif bc == "not-a-knot":
        # continuous third derivative across the first/last interior knot.
        A[0, 0] = hnp[1]
        A[0, 1] = -(hnp[0] + hnp[1])
        A[0, 2] = hnp[0]
        A[n - 1, n - 3] = hnp[-1]
        A[n - 1, n - 2] = -(hnp[-2] + hnp[-1])
        A[n - 1, n - 1] = hnp[-2]
    else:
        raise ValueError(
            f"cubic_spline_coeffs bc must be 'natural' or 'not-a-knot'; got {bc!r}"
        )
    Ab = asarray_on_device(xp, A, dev)
    Ab = xp.astype(Ab, xb.dtype) if hasattr(xp, "astype") else Ab

    # rhs (length n): interior entries 6*(dslope[i]-dslope[i-1]); both end rows
    # are homogeneous (0) for the two supported boundary conditions. The rhs
    # carries the whole y-dependence, so the gradient flows through here.
    zero = yb[:1] * 0.0  # (1,) on y's device/dtype, kept differentiable
    interior = 6.0 * (dslope[1:] - dslope[:-1])  # (n-2,)
    concat = getattr(xp, "concat", None) or xp.concatenate
    rhs = concat([zero, interior, zero])  # (n,)

    M = xp.linalg.solve(Ab, rhs)  # (n,)

    # power-basis coefficients on each interval (descending degree to match
    # spline_to_ppoly / Horner in eval_ppoly): c[0]=cubic, c[3]=constant.
    a3 = (M[1:] - M[:-1]) / (6.0 * h)
    a2 = M[:-1] / 2.0
    a1 = dslope - h * (2.0 * M[:-1] + M[1:]) / 6.0
    a0 = yb[:-1]
    return xp.stack([a3, a2, a1, a0], axis=0)  # (4, n-1)


def eval_cubic(xp, x, coeffs, r, *, extrapolate=True):
    """Evaluate cubic coefficients from ``cubic_spline_coeffs`` at ``r``.

    Thin alias for ``eval_ppoly`` (same power-basis layout); kept as a separate
    public name so callers building in-backend cubics read symmetrically. The
    spline value is differentiable w.r.t. both ``r`` and (through ``coeffs``) the
    ``y`` values used to build it.
    """
    return eval_ppoly(xp, x, coeffs, r, extrapolate=extrapolate)


def interp_linear(xp, x, y, r, *, extrapolate=True):
    """Piecewise-linear interpolation of ``(x, y)`` at ``r``, built in ``xp``.

    ``searchsorted`` for the interval + a lerp; trivially differentiable w.r.t.
    both ``r`` and ``y``. ``x`` must be increasing. ``extrapolate=True`` extends
    the edge line beyond the ends (matching ``eval_ppoly``'s finite
    extrapolation); ``'clip'``/``'const'``/``3`` clamp ``r`` to the range first
    (edge value beyond the ends).
    """
    dev = device_of(r, y, x)
    xb = asarray_on_device(xp, numpy.asarray(x), dev) * 1.0
    yb = y * 1.0
    if extrapolate is not True:
        if extrapolate not in ("clip", "const", 3):
            raise ValueError(
                "interp_linear extrapolate must be True, 'clip', 'const', or 3; "
                f"got {extrapolate!r}"
            )
        r = xp.clip(r, xb[0], xb[-1])
    # interval index in [0, n-2]
    idx = xp.clip(xp.searchsorted(xb, r, side="right") - 1, 0, xb.shape[0] - 2)
    x0 = xb[idx]
    x1 = xb[idx + 1]
    y0 = yb[idx]
    y1 = yb[idx + 1]
    # (x1 - x0) is a strictly positive geometry difference (x increasing), so no
    # dead-branch guard is needed: the denominator is never zero.
    t = (r - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


###############################################################################
#   (3) Frozen 2D tensor-product piecewise-power evaluator (mode 1, 2D).
###############################################################################
def rect_bivariate_to_ppoly(spl):
    """Convert a scipy ``RectBivariateSpline`` to a tensor-product PPoly block.

    Returns ``(xbr, ybr, c)`` where ``xbr`` (shape ``(mx+1,)``) and ``ybr``
    (shape ``(my+1,)``) are the distinct breakpoints in each dimension and ``c``
    has shape ``(kx+1, ky+1, mx, my)`` such that on
    ``xbr[ix] <= X < xbr[ix+1]`` and ``ybr[iy] <= Y < ybr[iy+1]`` the spline is

        sum_{px,py} c[px, py, ix, iy] * (X-xbr[ix])**(kx-px) * (Y-ybr[iy])**(ky-py).

    A ``RectBivariateSpline`` is a tensor product of 1D B-splines, so this is
    obtained by converting to the power basis one dimension at a time via the
    scipy 1D ``PPoly.from_spline`` (the y-direction for each x B-spline
    coefficient row, then the x-direction for each resulting (y-power,
    y-interval) slice). FITPACK's repeated boundary knots give zero-width
    intervals, which are dropped in BOTH dimensions so that the per-dimension
    ``searchsorted`` lookups are unambiguous. Called once at setup; the block
    then feeds the backend-agnostic ``eval_rect_ppoly`` below.
    """
    tx, ty, c = spl.tck
    kx, ky = spl.degrees
    nx = len(tx) - kx - 1
    ny = len(ty) - ky - 1
    C = numpy.asarray(c).reshape(nx, ny)

    # y-direction: convert each x-coefficient row to a 1D PPoly in y.
    Cy_list = []
    ybreak_full = None
    for i in range(nx):
        pp = _scipy_interpolate.PPoly.from_spline(
            _scipy_interpolate.BSpline(ty, C[i, :], ky)
        )
        ybreak_full = pp.x
        Cy_list.append(pp.c)  # (ky+1, my_full)
    Cy = numpy.stack(Cy_list, axis=0)  # (nx, ky+1, my_full)
    keepy = numpy.diff(ybreak_full) > 0.0
    ybr = numpy.append(ybreak_full[:-1][keepy], ybreak_full[-1])
    Cy = Cy[:, :, keepy]  # (nx, ky+1, my)
    my = Cy.shape[2]

    # x-direction: for each (y-power, y-interval) slice the nx values are B-spline
    # coefficients in x; convert each to a 1D PPoly in x.
    out = None
    xbreak_full = None
    keepx = None
    for py in range(ky + 1):
        for iy in range(my):
            pp = _scipy_interpolate.PPoly.from_spline(
                _scipy_interpolate.BSpline(tx, Cy[:, py, iy], kx)
            )
            if out is None:
                xbreak_full = pp.x
                keepx = numpy.diff(xbreak_full) > 0.0
                out = numpy.empty((kx + 1, ky + 1, int(keepx.sum()), my))
            out[:, py, :, iy] = pp.c[:, keepx]  # (kx+1, mx)
    xbr = numpy.append(xbreak_full[:-1][keepx], xbreak_full[-1])
    return xbr, ybr, out


def eval_rect_ppoly(xp, xbr, ybr, c, X, Y, *, extrapolate=True):
    """Evaluate a tensor-product piecewise polynomial at points ``(X, Y)``.

    ``(xbr, ybr, c)`` are as returned by ``rect_bivariate_to_ppoly``. Two
    ``xp.searchsorted`` interval lookups (one per dimension) + a nested 2D Horner
    using only namespace operations, so the value is computed natively and is
    differentiable w.r.t. both ``X`` and ``Y`` under jax/torch. Mathematically
    the same tensor-product spline as ``RectBivariateSpline.ev`` (agreement at
    the ~1 ulp level); the numpy code paths call ``.ev`` directly and never come
    through here.

    ``extrapolate`` matches ``eval_ppoly``: ``True`` evaluates the edge
    polynomial outside the grid (finite extrapolation, NaN-free dead branches);
    ``'clip'``/``'const'``/``3`` clamp ``(X, Y)`` to the grid first (edge value).
    """
    dev = device_of(X, Y)
    xb = asarray_on_device(xp, xbr, dev)
    yb = asarray_on_device(xp, ybr, dev)
    cb = asarray_on_device(xp, c, dev)
    if extrapolate is not True:
        if extrapolate not in ("clip", "const", 3):
            raise ValueError(
                "eval_rect_ppoly extrapolate must be True, 'clip', 'const', or 3; "
                f"got {extrapolate!r}"
            )
        X = xp.clip(X, xb[0], xb[-1])
        Y = xp.clip(Y, yb[0], yb[-1])
    kx = cb.shape[0] - 1
    ky = cb.shape[1] - 1
    ix = xp.clip(xp.searchsorted(xb, X, side="right") - 1, 0, cb.shape[2] - 1)
    iy = xp.clip(xp.searchsorted(yb, Y, side="right") - 1, 0, cb.shape[3] - 1)
    dx = X - xb[ix]
    dy = Y - yb[iy]
    # 2D Horner: outer Horner in dx over the x-powers; each x-power coefficient is
    # itself a Horner in dy over the y-powers.
    out = None
    for px in range(kx + 1):
        cyacc = None
        for py in range(ky + 1):
            coef = cb[px, py, ix, iy]
            cyacc = coef if cyacc is None else cyacc * dy + coef
        out = cyacc if out is None else out * dx + cyacc
    return out


###############################################################################
#   Spline1D / Spline2D convenience classes.
###############################################################################
class Spline1D:
    """Backend-agnostic 1D spline.

    Two construction modes, mirroring the two capabilities above:

    - **mode 1 (frozen, fast, eval-point-differentiable)** -- built from numpy
      ``(x, y)`` or a pre-fitted scipy spline. A scipy ``InterpolatedUnivariate``
      spline is fitted (or reused) and frozen to a power-basis PPoly table.
      ``__call__`` on numpy input calls the scipy spline (BYTE-IDENTICAL to using
      scipy directly); on a backend array it evaluates the frozen table through
      ``eval_ppoly`` (differentiable w.r.t. the evaluation point).

    - **mode 2 (in-backend, differentiable in y)** -- triggered when ``y`` is a
      backend (jax/torch) array. The cubic coefficients are built in-backend via
      ``cubic_spline_coeffs``, so the spline is differentiable w.r.t. the
      ``y`` values (for parameter-dependent tables). ``__call__`` evaluates those
      coefficients through ``eval_cubic``. (numpy input still goes through the
      scipy path when one was also fitted.)

    Parameters
    ----------
    x : array_like
        Strictly increasing abscissae.
    y : array_like
        Ordinates. A backend array selects mode 2 (and ``k`` must be 1 or 3).
    k : int, optional
        Spline degree for the scipy/frozen fit (default 3). For mode 2, ``k=3``
        builds a cubic and ``k=1`` a piecewise-linear interpolant.
    ext : int or str, optional
        Out-of-range behaviour for the frozen/backend evaluation, passed through
        to scipy's ``InterpolatedUnivariateSpline`` (``ext``) and mapped onto the
        ``eval_ppoly`` ``extrapolate`` knob: ``0`` -> finite extrapolation,
        ``3`` -> constant beyond the ends. (Default 0.)
    bc : str, optional
        Boundary condition for the mode-2 in-backend cubic (``'natural'`` or
        ``'not-a-knot'``; default ``'natural'``).
    """

    def __init__(self, x, y, k=3, ext=0, bc="natural"):
        self._k = int(k)
        self._ext = ext
        self._extrapolate = True if ext in (0, "extrapolate") else "const"
        self._bc = bc
        self._mode2 = is_backend_array(y)
        if self._mode2:
            import array_api_compat

            self._xp = array_api_compat.array_namespace(y)
            self._y = y
            self._x = numpy.asarray(x, dtype=float)
            if self._k == 3:
                self._coeffs = cubic_spline_coeffs(self._xp, self._x, y, bc=bc)
            elif self._k == 1:
                self._coeffs = None  # interp_linear evaluates directly from (x,y)
            else:
                raise ValueError("Spline1D mode-2 (backend y) supports only k=1 or k=3")
            self._spl = None
        else:
            self._x = numpy.asarray(x, dtype=float)
            yn = numpy.asarray(y, dtype=float)
            self._spl = _scipy_interpolate.InterpolatedUnivariateSpline(
                self._x, yn, k=self._k, ext=ext
            )
            self._ppoly_x, self._ppoly_c = spline_to_ppoly(self._spl)

    def __call__(self, r):
        # numpy / scalar input: byte-identical scipy path (mode 1) or numpy-eval
        # of the in-backend coefficients (mode 2 has no scipy spline).
        if not is_backend_array(r):
            if self._spl is not None:
                return self._spl(r)
            # mode 2 with a numpy query: evaluate the in-backend coeffs via numpy
            if self._k == 1:
                return interp_linear(
                    numpy,
                    self._x,
                    numpy.asarray(self._y),
                    r,
                    extrapolate=self._extrapolate,
                )
            return eval_ppoly(
                numpy,
                self._x,
                numpy.asarray(self._coeffs),
                r,
                extrapolate=self._extrapolate,
            )
        import array_api_compat

        xp = array_api_compat.array_namespace(r)
        if self._mode2:
            if self._k == 1:
                return interp_linear(
                    xp, self._x, self._y, r, extrapolate=self._extrapolate
                )
            return eval_cubic(
                xp, self._x, self._coeffs, r, extrapolate=self._extrapolate
            )
        return eval_ppoly(
            xp, self._ppoly_x, self._ppoly_c, r, extrapolate=self._extrapolate
        )


class Spline2D:
    """Backend-agnostic 2D tensor-product spline over a rectangular grid.

    Frozen (mode 1) only: built from a grid ``(x, y, z)`` or a pre-fitted scipy
    ``RectBivariateSpline``, and frozen to a tensor-product power-basis block.
    ``__call__(X, Y)`` on numpy input calls ``RectBivariateSpline.ev``
    (BYTE-IDENTICAL to scipy); on backend arrays it evaluates the frozen block
    through ``eval_rect_ppoly`` (differentiable w.r.t. the evaluation points).
    Serves interpRZPotential and the DF ``RectBivariateSpline`` tables.

    Parameters
    ----------
    x, y : array_like
        Strictly increasing grid abscissae (lengths ``nx``, ``ny``).
    z : array_like
        Grid values, shape ``(nx, ny)``.
    kx, ky : int, optional
        Spline degrees in each dimension (default 3).
    spl : scipy.interpolate.RectBivariateSpline, optional
        A pre-fitted spline to reuse instead of fitting from ``(x, y, z)``.
    ext : int or str, optional
        Out-of-range behaviour mapped onto ``eval_rect_ppoly``'s ``extrapolate``
        (``0`` -> finite extrapolation, ``3`` -> constant). Default 0. (scipy's
        ``.ev`` always extrapolates, so the numpy path matches ``ext=0``.)
    """

    def __init__(self, x=None, y=None, z=None, kx=3, ky=3, spl=None, ext=0):
        if spl is None:
            spl = _scipy_interpolate.RectBivariateSpline(
                numpy.asarray(x, dtype=float),
                numpy.asarray(y, dtype=float),
                numpy.asarray(z, dtype=float),
                kx=kx,
                ky=ky,
            )
        self._spl = spl
        self._extrapolate = True if ext in (0, "extrapolate") else "const"
        self._xbr, self._ybr, self._c = rect_bivariate_to_ppoly(spl)

    def __call__(self, X, Y):
        if not (is_backend_array(X) or is_backend_array(Y)):
            return self._spl.ev(X, Y)
        import array_api_compat

        ref = X if is_backend_array(X) else Y
        xp = array_api_compat.array_namespace(ref)
        return eval_rect_ppoly(
            xp, self._xbr, self._ybr, self._c, X, Y, extrapolate=self._extrapolate
        )
