###############################################################################
#   Backend-agnostic associated Legendre functions P_l^m(x) for all degrees
#   l < L and orders 0 <= m < M, with the Condon-Shortley phase (matching
#   scipy.special.assoc_legendre_p_all(..., branch_cut=2)). Replaces
#   galpy.util.special.compute_legendre on the SCF / MultipoleExpansion path so
#   those potentials run and differentiate under every backend.
#
#   P is built by the standard forward (Bonnet) recurrences:
#     P_m^m   = (-1)^m (2m-1)!! (1-x^2)^{m/2}
#     P_{m+1}^m = x (2m+1) P_m^m
#     (l-m) P_l^m = x (2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
#   The optional first/second x-derivatives use
#     (x^2-1) dP/dx = l x P_l^m - (l+m) P_{l-1}^m
#     (1-x^2) d2P/dx^2 = 2x dP/dx - l(l+1) P + m^2/(1-x^2) P   (Legendre ODE)
#   (these diverge at the poles x=+-1 for m>=1, exactly as scipy returns, and are
#   multiplied by sin^2(theta) in the physical theta-derivatives downstream).
#
#   Everything is pure arithmetic built with lists + xp.stack (no in-place
#   mutation), so it differentiates cleanly under jax and torch -- and the
#   x-derivatives are also available straight from autodiff.
#
#   Under a jax trace the Bonnet l-recurrence is rolled into a single
#   ``lax.scan`` over l (see ``_bonnet_scan_jax``): the eager Python double loop
#   would bake all O(L*M) recurrence steps into the user's jaxpr, whereas the
#   scan traces the recurrence body once. Eager numpy/torch/jax keep the Python
#   loop unchanged (byte-identical); the scan carries the same values at the same
#   iteration count, so the traced result matches the unrolled loop to the XLA
#   fma-fusion floor (the same ~1e-11 that jitting the unrolled loop itself shows
#   vs the eager per-op result -- the scan adds no error beyond that).
###############################################################################
from ..._namespaces import under_jax_trace


def assoc_legendre(xp, L, M, x, deriv=0):
    """P_l^m(x), shape ``x.shape + (L, M)`` (Condon-Shortley phase).

    deriv: 0 -> P; 1 -> (P, dP/dx); 2 -> (P, dP/dx, d2P/dx2).
    L, M are static ints; x is a backend array (or scalar) with |x| <= 1.
    """
    x = xp.asarray(x) * 1.0
    one = xp.ones_like(x)
    zero = xp.zeros_like(x)
    # (1-x^2)^{1/2}; clip keeps it real at |x|=1 (interior x is unaffected).
    somx2 = xp.sqrt(xp.where(x * x < 1.0, 1.0 - x * x, zero))

    def _stack(grid):
        return xp.stack([xp.stack(row, axis=-1) for row in grid], axis=-2)

    # P[l][m] as a list-of-lists of backend arrays (functional, no mutation).
    if under_jax_trace(x):  # roll the Bonnet l-recurrence into one lax.scan
        Parr = _bonnet_scan_jax(L, M, x, somx2, one, zero)  # x.shape + (L, M)
        if deriv == 0:  # value-only: return the scan output straight (no unstack)
            return Parr
        # derivatives read P[l][m]; expose the scanned rows as that list of views.
        P = [[Parr[..., l, m] for m in range(M)] for l in range(L)]
    else:  # eager (numpy/torch/jax): keep the unrolled Python double loop
        P = [[zero for _ in range(M)] for _ in range(L)]
        pmm = one  # running P_m^m diagonal
        for m in range(M):
            if m > 0:
                pmm = pmm * (-(2 * m - 1)) * somx2
            if m < L:
                P[m][m] = pmm
            if m + 1 < L:
                P[m + 1][m] = x * (2 * m + 1) * pmm
            for l in range(m + 2, L):
                P[l][m] = (
                    x * (2 * l - 1) * P[l - 1][m] - (l + m - 1) * P[l - 2][m]
                ) / (l - m)
        Parr = _stack(P)
        if deriv == 0:
            return Parr

    den = x * x - 1.0  # (x^2-1); singular only at the poles |x|=1
    pole = x * x >= 1.0  # symmetry-axis poles (cos theta = +-1)
    den_safe = xp.where(pole, one, den)  # AD-safe: keep the dead branch finite
    dP = [[zero for _ in range(M)] for _ in range(L)]
    for m in range(M):
        for l in range(m, L):
            plm1 = P[l - 1][m] if l - 1 >= m else zero
            num = l * x * P[l][m] - (l + m) * plm1
            if m == 0:
                # m=0 derivative is a polynomial (finite at the poles): guard the
                # 0/0 and substitute the closed form P_l'(+-1) = x^{l+1} l(l+1)/2.
                dP[l][m] = xp.where(
                    pole, x ** (l + 1) * (l * (l + 1) / 2.0), num / den_safe
                )
            else:
                dP[l][m] = num / den  # m>=1 diverges at the poles, as scipy does
    dParr = _stack(dP)
    if deriv == 1:
        return Parr, dParr

    om = 1.0 - x * x
    om_safe = xp.where(pole, one, om)
    d2 = [[zero for _ in range(M)] for _ in range(L)]
    for m in range(M):
        for l in range(m, L):
            if m == 0:
                gen = (2.0 * x * dP[l][m] - l * (l + 1) * P[l][m]) / om_safe
                # finite limit at +-1 = (l-1)l(l+1)(l+2)/8 (scipy's exact-pole
                # convention returns this unsigned value at both poles)
                d2[l][m] = xp.where(
                    pole, ((l - 1) * l * (l + 1) * (l + 2) / 8.0) * one, gen
                )
            else:
                d2[l][m] = (
                    2.0 * x * dP[l][m] - l * (l + 1) * P[l][m] + (m * m) / om * P[l][m]
                ) / om
    return Parr, dParr, _stack(d2)


def _bonnet_scan_jax(L, M, x, somx2, one, zero):
    """Traced-only P_l^m via a single ``lax.scan`` over l (carry the two previous
    full m-rows). Returns ``Parr`` of shape ``x.shape + (L, M)`` (same as the
    eager list + ``xp.stack``).

    Each scanned row reproduces, element-for-element, the eager arithmetic:
    interior/sub-diagonal m come from the Bonnet three-term recurrence (the
    sub-diagonal falls out of it since ``P[l-2][l-1] == 0``), the diagonal m==l is
    injected from the running P_m^m, and the m>l upper triangle is zeroed. The
    m==l denominator (l-m == 0) is guarded to 1 so the discarded Bonnet branch
    stays finite (AD-safe); the ``where`` then selects the diagonal value there.
    """
    import jax
    import jax.numpy as jnp

    # Running diagonal P_m^m for all m, via the eager m-recurrence (M small).
    pmm = one
    diag = [one]
    for m in range(1, M):
        pmm = pmm * (-(2 * m - 1)) * somx2
        diag.append(pmm)
    pmm_vec = jnp.stack(diag, axis=-1)  # x.shape + (M,)
    mvec = jnp.arange(M)  # (M,)
    xM = x[..., None]  # x.shape + (1,)
    zeros_row = jnp.broadcast_to(zero[..., None], x.shape + (M,))

    def body(carry, ll):
        Pm1, Pm2 = carry  # P[l-1][:], P[l-2][:] rows
        den = jnp.where(mvec == ll, 1.0, ll - mvec)  # guard 0/0 at the diagonal
        bonnet = (xM * (2 * ll - 1) * Pm1 - (ll + mvec - 1) * Pm2) / den
        row = jnp.where(mvec == ll, pmm_vec, bonnet)  # inject P_m^m diagonal
        row = jnp.where(mvec > ll, zeros_row, row)  # exact 0 upper triangle
        return (row, Pm1), row

    _, Prows = jax.lax.scan(body, (zeros_row, zeros_row), jnp.arange(L))
    # Prows: (L,) + x.shape + (M,) -> x.shape + (L, M) (matches eager _stack).
    return jnp.moveaxis(Prows, 0, -2)
