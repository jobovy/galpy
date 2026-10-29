###############################################################################
#   galpy.backend._reference.symplectic_coeffs: the coefficients of galpy's
#   fourth- and sixth-order symplectic integrators, shared by the jax and torch
#   in-backend steppers (galpy.backend._jax/_torch.orbit_symplec) and mirrored
#   from galpy/util/bovy_symplecticode.c, which defines them once for the C and
#   the variational (dxdv) steppers (a test checks the two copies agree).
#
#   A step of either order is the composition
#       drift(c_1 dt) kick(d_1 dt) drift(c_2 dt) ... kick(d_m dt) drift(c_{m+1} dt)
#   with the kick's force evaluated at the drifted position and the time
#   advanced by the drifts so far: symplec4 (Forest & Ruth 1990; Kinoshita et
#   al. 1991) with four drifts and three kicks, symplec6 (Yoshida 1990) with
#   eight drifts and seven kicks. The leapfrog is the order-2 member,
#   kick(dt/2) drift(dt) kick(dt/2), which orbit_leapfrog implements on its own
#   because its end-of-step force can be carried into the next step.
###############################################################################
SYMPLEC4_CS = (
    0.6756035959798289,
    -0.1756035959798288,
    -0.1756035959798288,
    0.6756035959798289,
)
SYMPLEC4_DS = (1.3512071919596578, -1.7024143839193153, 1.3512071919596578)
SYMPLEC6_CS = (
    0.392256805238780,
    0.510043411918458,
    -0.471053385409758,
    0.0687531682525198,
    0.0687531682525198,
    -0.471053385409758,
    0.510043411918458,
    0.392256805238780,
)
SYMPLEC6_DS = (
    0.784513610477560,
    0.235573213359357,
    -1.17767998417887,
    1.31518632068391,
    -1.17767998417887,
    0.235573213359357,
    0.784513610477560,
)
# order -> (drift coefficients, kick coefficients); len(cs) == len(ds) + 1
COEFFICIENTS = {4: (SYMPLEC4_CS, SYMPLEC4_DS), 6: (SYMPLEC6_CS, SYMPLEC6_DS)}
# force evaluations per step
FORCE_EVALUATIONS = {2: 1, 4: len(SYMPLEC4_DS), 6: len(SYMPLEC6_DS)}
