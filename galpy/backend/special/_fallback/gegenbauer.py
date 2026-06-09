###############################################################################
#   Backend-agnostic Gegenbauer (ultraspherical) polynomials C_n^alpha(x) for
#   0 <= n < N, via the standard three-term recurrence
#       C_0 = 1,  C_1 = 2 alpha x,
#       (n+1) C_{n+1} = 2(n+alpha) x C_n - (n+2 alpha-1) C_{n-1}.
#   This is the SCFPotential radial basis (galpy.potential.SCFPotential._C uses
#   the same recurrence with alpha = 2l + 3/2). Built with lists + xp.stack (no
#   in-place mutation), so it differentiates under jax and torch; the numpy path
#   reproduces SCF's existing recurrence value-for-value.
###############################################################################


def gegenbauer(xp, N, alpha, x):
    """C_n^alpha(x) for 0 <= n < N, shape ``x.shape + (N,)``.

    N is a static int, alpha a scalar, x a backend array (or scalar).
    """
    x = xp.asarray(x) * 1.0
    cols = [xp.ones_like(x)]  # C_0 = 1
    if N > 1:
        cnm1 = cols[0]
        cn = 2.0 * alpha * x  # C_1 = 2 alpha x
        cols.append(cn)
        for n in range(1, N - 1):
            cnp1 = (2.0 * (n + alpha) * x * cn - (n + 2.0 * alpha - 1.0) * cnm1) / (
                n + 1.0
            )
            cols.append(cnp1)
            cnm1, cn = cn, cnp1
    return xp.stack(cols, axis=-1)
