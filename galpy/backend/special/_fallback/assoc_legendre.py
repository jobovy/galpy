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
###############################################################################


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

    # P[l][m] as a list-of-lists of backend arrays (functional, no mutation).
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
            P[l][m] = (x * (2 * l - 1) * P[l - 1][m] - (l + m - 1) * P[l - 2][m]) / (
                l - m
            )

    def _stack(grid):
        return xp.stack([xp.stack(row, axis=-1) for row in grid], axis=-2)

    Parr = _stack(P)
    if deriv == 0:
        return Parr

    den = x * x - 1.0  # (x^2-1); singular only at the poles
    dP = [[zero for _ in range(M)] for _ in range(L)]
    for m in range(M):
        for l in range(m, L):
            plm1 = P[l - 1][m] if l - 1 >= m else zero
            dP[l][m] = (l * x * P[l][m] - (l + m) * plm1) / den
    dParr = _stack(dP)
    if deriv == 1:
        return Parr, dParr

    om = 1.0 - x * x
    d2 = [[zero for _ in range(M)] for _ in range(L)]
    for m in range(M):
        for l in range(m, L):
            d2[l][m] = (
                2.0 * x * dP[l][m] - l * (l + 1) * P[l][m] + (m * m) / om * P[l][m]
            ) / om
    return Parr, dParr, _stack(d2)
