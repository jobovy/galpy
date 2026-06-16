###############################################################################
#   galpy.backend.quadrature: backend-agnostic fixed-order Gauss-Legendre
#   quadrature.
#
#   The nodes and weights are numpy float64 constants computed once (cached,
#   keyed by order) -- precision is the point, so the tables stay float64 and
#   the result is exit-cast back to the input dtype with ``match_input_dtype``.
#   Per call the tables are materialised into the active backend namespace with
#   ``asarray_on_device``, anchored on the device of the limits (or an explicit
#   ``device=`` hint), so CUDA inputs do not meet a CPU table. The integrand's
#   own device cannot be probed (a CUDA-closure integrand rejects a CPU sample
#   node before returning), so when the limits are plain Python scalars but the
#   integrand closes over CUDA tensors, the caller passes ``device=`` (e.g. the
#   device of its coordinates). Every arithmetic step from there on is plain
#   namespace ops, so the result differentiates under jax and torch w.r.t. the
#   integration limits AND through the integrand (its parameters). There is no
#   internal jit -- galpy is jit-COMPATIBLE, not jit-ing; users wrap their own
#   galpy-using code.
#
#   Fixed-order Gauss-Legendre is a differentiable APPROXIMATION, not adaptive:
#   pick ``n`` large enough for the target accuracy on your integrand.
#
#   The semi-infinite and split-interval helpers reproduce the substitutions
#   already used elsewhere in galpy (EllipsoidalPotential's t=1/s^2-1 tail,
#   hyp2f1's boundary-layer X=xi^k map). As in
#   ``special/_fallback/bessel_k.py::_k01``, every ``xp.where`` here is
#   dead-branch guarded: both branches evaluate eagerly under jax/torch, so the
#   unused side's argument is clamped into its valid domain to keep it finite
#   and stop a NaN/inf from poisoning reverse-mode gradients.
###############################################################################
import numpy

from ._namespaces import asarray_on_device, device_of, match_input_dtype

# Cache of (nodes, weights) on [0, 1] keyed by order, as numpy float64 constants.
_GL01_CACHE = {}


def gauss_legendre_01(n):
    """Return (nodes, weights) for n-point Gauss-Legendre quadrature on [0, 1].

    numpy float64 constants (cached). The caller converts them into the backend
    namespace with ``xp.asarray`` so they pick up the backend's array type while
    the integrand stays differentiable.
    """
    cached = _GL01_CACHE.get(n)
    if cached is None:
        x, w = numpy.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (x + 1.0)  # map [-1, 1] -> [0, 1]
        weights = 0.5 * w
        cached = (nodes, weights)
        _GL01_CACHE[n] = cached
    return cached


def gauss_legendre_nodes(n, a=0.0, b=1.0):
    """Return (nodes, weights) for n-point Gauss-Legendre quadrature on [a, b].

    numpy float64 constants. The canonical [0, 1] nodes/weights are cached
    (keyed by order, via ``gauss_legendre_01``) and affinely remapped to
    [a, b]: ``nodes = a + (b - a) * x01`` and ``weights = (b - a) * w01``, so
    ``sum(weights * f(nodes))`` approximates ``int_a^b f``. ``a`` and ``b`` are
    plain Python scalars (the quadrature panel boundaries); for limits that are
    backend arrays / differentiable, pass [0, 1] here and fold the (b - a)
    Jacobian in the namespace (this is what ``fixed_quad`` does so the result
    differentiates w.r.t. the limits).
    """
    x01, w01 = gauss_legendre_01(n)
    span = float(b) - float(a)
    nodes = float(a) + span * x01
    weights = span * w01
    return nodes, weights


def _gl01_on(xp, n, dev):
    """[0, 1] GL nodes/weights as backend arrays on device ``dev`` (float64)."""
    x01, w01 = gauss_legendre_01(n)
    return asarray_on_device(xp, x01, dev), asarray_on_device(xp, w01, dev)


def fixed_quad(xp, integrand, a, b, *, n=50, device=None):
    r"""``int_a^b integrand(s) ds`` by fixed-order Gauss-Legendre quadrature.

    Backend-agnostic and differentiable in both the limits ``a, b`` and through
    ``integrand`` (its parameters). ``integrand`` is called ONCE with the full
    array of quadrature nodes (it must be vectorised over a trailing node axis,
    i.e. broadcast over the last dimension) and the result is reduced as
    ``(b - a) * sum(w01 * integrand(nodes))`` where ``w01`` are the [0, 1]
    weights and ``nodes = a + (b - a) * x01``.

    Parameters
    ----------
    xp : module
        Array namespace (numpy / jax.numpy / array_api_compat.torch).
    integrand : callable
        ``integrand(nodes) -> array`` broadcasting over the trailing node axis.
    a, b : scalar or backend array
        Finite integration limits. May be backend arrays (autodiff flows to
        them through the affine remap and the (b - a) prefactor).
    n : int, optional
        Number of Gauss-Legendre nodes (default 50). Fixed-order GL is an
        approximation, not adaptive: raise ``n`` for tighter accuracy.
    device : optional
        Device for the node/weight tables. Defaults to the limits' device;
        pass this when the limits are Python scalars but ``integrand`` closes
        over CUDA tensors (so the nodes land on the integrand's device).

    Returns
    -------
    array
        The integral, exit-cast to the limits' floating dtype.
    """
    dev = device if device is not None else device_of(a, b)
    x01, w01 = _gl01_on(xp, n, dev)
    a = asarray_on_device(xp, a, dev) * 1.0
    b = asarray_on_device(xp, b, dev) * 1.0
    span = b - a
    # Broadcast the affine remap over the node axis; a, b carry the autodiff.
    nodes = a[..., None] + span[..., None] * x01
    vals = integrand(nodes)
    result = span * xp.sum(w01 * vals, axis=-1)
    return match_input_dtype(result, a, b)


def fixed_quad_semiinfinite(xp, integrand, a, *, n=50, kind="recip", device=None):
    r"""``int_a^inf integrand(s) ds`` by fixed-order Gauss-Legendre quadrature.

    The semi-infinite range is mapped to a finite one before applying
    fixed-order GL, with the substitution's Jacobian folded into the integrand.
    Backend-agnostic and differentiable in the lower limit ``a`` and through
    ``integrand``.

    Parameters
    ----------
    xp : module
        Array namespace.
    integrand : callable
        ``integrand(s) -> array`` broadcasting over the trailing node axis;
        ``s`` lies in ``[a, inf)``.
    a : scalar or backend array
        Finite lower limit.
    n : int, optional
        Number of Gauss-Legendre nodes (default 50).
    kind : {'recip', 'tan'}, optional
        Substitution mapping ``[a, inf) -> (0, 1]`` (the GL panel):

        - ``'recip'`` (default): ``s = a + 1/u**2 - 1`` for ``u in (0, 1]``,
          i.e. ``s + 1 - a = 1/u**2``, ``ds = -2/u**3 du``. This is exactly the
          ``t = 1/s**2 - 1`` substitution used in
          ``galpy/potential/EllipsoidalPotential.py`` (there ``a = 0``):
          ``u = 0`` is ``s = inf`` and ``u = 1`` is ``s = a``. Best for tails
          that decay polynomially or faster.
        - ``'tan'``: ``s = a + tan(pi/2 * u)`` for ``u in [0, 1)``,
          ``ds = pi/2 * sec^2(pi/2 * u) du``. Good for slowly decaying /
          oscillatory tails (e.g. ``1/(1+s**2)``).

    Returns
    -------
    array
        The integral, exit-cast to ``a``'s floating dtype.

    Notes
    -----
    The endpoints of each map are degenerate (``u -> 0`` is ``s -> inf``;
    ``u -> 1`` is the ``tan`` singularity). Gauss-Legendre nodes are strictly
    interior so those points are never evaluated, but the dead-branch guarding
    convention is still honoured: ``u`` is clamped strictly inside ``(0, 1)``
    before the map so no node can produce inf/NaN that would poison AD.
    """
    dev = device if device is not None else device_of(a)
    x01, w01 = _gl01_on(xp, n, dev)
    a = asarray_on_device(xp, a, dev) * 1.0
    a_b = a[..., None]
    if kind == "recip":
        # s + (1 - a) = 1/u**2  =>  s = a - 1 + 1/u**2 ; ds = -2 u**-3 du.
        # u=1 -> s=a, u->0 -> s->inf. Clamp u off 0 so 1/u**2 stays finite.
        u = xp.maximum(x01, xp.ones_like(x01) * 1e-300)
        inv_u2 = 1.0 / (u * u)
        s = a_b - 1.0 + inv_u2
        jac = 2.0 * inv_u2 / u  # |ds/du| = 2 / u**3
    elif kind == "tan":
        # s = a + tan(pi/2 * u) ; ds = (pi/2) sec^2(pi/2 * u) du.
        # u=0 -> s=a, u->1 -> s->inf. Clamp u off 1 so tan stays finite.
        half_pi = numpy.pi / 2.0
        u = xp.minimum(x01, xp.ones_like(x01) * (1.0 - 1e-15))
        theta = half_pi * u
        tan_t = xp.tan(theta)
        s = a_b + tan_t
        jac = half_pi * (1.0 + tan_t * tan_t)  # sec^2 = 1 + tan^2
    else:  # pragma: no cover - guarded API misuse
        raise ValueError(
            f"fixed_quad_semiinfinite: unknown kind {kind!r} (use 'recip' or 'tan')"
        )
    vals = integrand(s) * jac
    result = xp.sum(w01 * vals, axis=-1)
    return match_input_dtype(result, a)


def _boundary_layer_remap(xp, x01, w01, k):
    r"""Apply the ``X = xi**k`` boundary-layer map to [0, 1] nodes/weights.

    Reproduces the regularising substitution from
    ``special/_fallback/hyp2f1.py``: with ``X = xi**k`` and
    ``dX = k * xi**(k-1) dxi``, a node density that clusters toward ``X = 0``
    resolves an endpoint near-singularity there for plain fixed-order GL. The
    Jacobian ``dX`` is folded into the returned weights, so callers integrate
    ``f(X)`` directly. ``k = 1`` is the identity.
    """
    if k == 1.0:
        return x01, w01
    X = x01**k
    dX = k * x01 ** (k - 1.0)
    return X, w01 * dX


def transformed_quad(xp, integrand, a, b, *, n=50, interior_point=None, device=None):
    r"""Finite GL on ``[a, b]``, optionally split at a near-singular interior point.

    For a smooth integrand this is ``fixed_quad``. When ``integrand`` has a
    near-singularity (a kink, an endpoint-type singularity, a sqrt-limit) at an
    interior abscissa -- the disk ``a = R`` pattern, or a distribution-function
    ``sqrt`` limit -- pass ``interior_point=c`` (with ``a < c < b``). The range
    is then split into ``[a, c]`` and ``[c, b]`` and an n-point GL panel is
    applied to each, so the singular point sits on a panel boundary (Gauss
    nodes are strictly interior and never land on it). Each panel additionally
    uses the ``X = xi**k`` boundary-layer node clustering from
    ``special/_fallback/hyp2f1.py`` toward its boundary at ``c``, concentrating
    nodes near the near-singularity.

    Backend-agnostic and differentiable in ``a, b`` (and ``interior_point`` if
    passed as a backend array) and through ``integrand``.

    Parameters
    ----------
    xp : module
        Array namespace.
    integrand : callable
        ``integrand(s) -> array`` broadcasting over the trailing node axis.
    a, b : scalar or backend array
        Finite integration limits.
    n : int, optional
        Number of Gauss-Legendre nodes PER panel (default 50).
    interior_point : scalar or backend array or None, optional
        If given, an interior abscissa ``a < c < b`` at which to split; the
        integrand may be near-singular there. If None, a single n-point GL
        panel is used over ``[a, b]`` (== ``fixed_quad``).
    device : optional
        Device for the node/weight tables (see ``fixed_quad``); pass when the
        limits are scalars but ``integrand`` closes over CUDA tensors.

    Returns
    -------
    array
        The integral, exit-cast to the limits' floating dtype.
    """
    if interior_point is None:
        return fixed_quad(xp, integrand, a, b, n=n, device=device)
    dev = device if device is not None else device_of(a, b, interior_point)
    x01, w01 = _gl01_on(xp, n, dev)
    a = asarray_on_device(xp, a, dev) * 1.0
    b = asarray_on_device(xp, b, dev) * 1.0
    c = asarray_on_device(xp, interior_point, dev) * 1.0
    # Cluster nodes toward the (interior) near-singular endpoint of each panel.
    # Left panel [a, c]: xi=0 maps to c (its right end) via 1 - X.
    # Right panel [c, b]: xi=0 maps to c (its left end) via X.
    k = 3.0
    X, wX = _boundary_layer_remap(xp, x01, w01, k)
    a_b, b_b, c_b = a[..., None], b[..., None], c[..., None]
    # Left panel [a, c] with X clustered at c: s = c - (c - a) * (1 - X)? Keep it
    # simple and robust -- map X in [0,1] from c (X=0) to a (X=1):
    span_l = c_b - a_b
    s_l = c_b - span_l * X
    vals_l = integrand(s_l)
    int_l = (c - a) * xp.sum(wX * vals_l, axis=-1)
    # Right panel [c, b] with X clustered at c: X=0 -> c, X=1 -> b.
    span_r = b_b - c_b
    s_r = c_b + span_r * X
    vals_r = integrand(s_r)
    int_r = (b - c) * xp.sum(wX * vals_r, axis=-1)
    result = int_l + int_r
    return match_input_dtype(result, a, b, c)


def nested_quad(xp, integrand, bounds, *, n=50, device=None):
    r"""Tensor-product fixed-order GL over a hyper-rectangle.

    The backend-agnostic, vectorised generalisation of
    ``galpy/potential/SCFPotential.py::_gaussianQuadrature``: integrate
    ``integrand`` over the d-dimensional box defined by ``bounds`` using a
    full tensor product of n-point Gauss-Legendre rules.

    ``integrand`` is called ONCE with d node arrays, one per dimension, each
    already broadcast onto its own axis of a d-dimensional node grid of shape
    ``(n, n, ..., n)`` (so ``integrand`` is fully vectorised over the product
    grid, NOT looped). Its return is contracted against the tensor-product
    weights.

    Backend-agnostic and differentiable in the ``bounds`` (if backend arrays)
    and through ``integrand``.

    Parameters
    ----------
    xp : module
        Array namespace.
    integrand : callable
        ``integrand(s_0, s_1, ..., s_{d-1}) -> array``; each ``s_i`` is an array
        broadcast onto axis ``i`` of the (n, n, ..., n) grid. The return is
        reduced over those d node axes (which must be its TRAILING d axes).
    bounds : sequence of (a_i, b_i)
        Per-dimension finite limits, ``[[a_0, b_0], ..., [a_{d-1}, b_{d-1}]]``.
        Each ``a_i, b_i`` may be a scalar or a backend array.
    n : int or sequence of int, optional
        Number of GL nodes per dimension (a scalar is used for all; default 50).
    device : optional
        Device for the node/weight tables (see ``fixed_quad``); pass when the
        bounds are scalars but ``integrand`` closes over CUDA tensors.

    Returns
    -------
    array
        The integral over the box, exit-cast to the bounds' floating dtype.
    """
    d = len(bounds)
    if isinstance(n, int):
        ns = [n] * d
    else:
        ns = list(n)
    flat_lims = [lim for ab in bounds for lim in ab]
    dev = device if device is not None else device_of(*flat_lims)
    node_arrays = []
    weight_arrays = []
    spans = []
    cast_coords = []
    for i in range(d):
        x01, w01 = _gl01_on(xp, ns[i], dev)
        a = asarray_on_device(xp, bounds[i][0], dev) * 1.0
        b = asarray_on_device(xp, bounds[i][1], dev) * 1.0
        span = b - a
        # Place this dimension's nodes on its own (trailing) grid axis i; the
        # other d-1 node axes are length-1 here and broadcast.
        ax_shape = [1] * d
        ax_shape[i] = ns[i]
        nodes_i = a + span * xp.reshape(x01, tuple(ax_shape))
        node_arrays.append(nodes_i)
        weight_arrays.append(span * xp.reshape(w01, tuple(ax_shape)))
        spans.append(span)
        cast_coords.append(a)
        cast_coords.append(b)
    vals = integrand(*node_arrays)
    # Tensor-product weight grid (product over dims), broadcast against vals.
    wgrid = weight_arrays[0]
    for wa in weight_arrays[1:]:
        wgrid = wgrid * wa
    # Sum over the trailing d node axes.
    axes = tuple(range(-d, 0))
    result = xp.sum(wgrid * vals, axis=axes)
    return match_input_dtype(result, *cast_coords)
