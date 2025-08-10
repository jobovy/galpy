import sympy
import sympy as sp
import time

r = sympy.Symbol("r", real=True, positive=True)
EARTH_RADIUS_KM = 6371.0
pieces = [
        (13.0885 - 8.8381 * (r / EARTH_RADIUS_KM) ** 2, 0, 1221.5),
        (12.5815 - 1.2638 * (r / EARTH_RADIUS_KM)
         - 3.6426 * (r / EARTH_RADIUS_KM) ** 2
         - 5.5281 * (r / EARTH_RADIUS_KM) ** 3, 1221.5, 3480.0),
        (7.9565 - 6.4761 * (r / EARTH_RADIUS_KM)
         + 5.5283 * (r / EARTH_RADIUS_KM) ** 2
         - 3.0807 * (r / EARTH_RADIUS_KM) ** 3, 3480.0, 5701.0),
        (5.3197 - 1.4836 * (r / EARTH_RADIUS_KM), 5701.0, 5771.0),
        (11.2494 - 8.0298 * (r / EARTH_RADIUS_KM), 5771.0, 5971.0),
        (7.1089 - 3.8045 * (r / EARTH_RADIUS_KM), 5971.0, 6151.0),
        (2.6910 + 0.6924 * (r / EARTH_RADIUS_KM), 6151.0, 6346.6),
        (2.6910 + 0.6924 * (r / EARTH_RADIUS_KM), 6346.6, 6356.0),
        (2.6, 6356.0, EARTH_RADIUS_KM),
        # # Explicitly include density for r >= EARTH_RADIUS_KM:
        (0.0, EARTH_RADIUS_KM, sympy.oo),
    ]

def earth_PREM_density_sym():
    r = sympy.Symbol("r", real=True, positive=True)
    # Build the Piecewise arguments from the list
    pw_args = []
    for expr, rmin, rmax in pieces:
        cond = (r >= rmin) & (r < rmax)
        pw_args.append((expr, cond))

    # Add fallback condition for r >= EARTH_RADIUS_KM (density = 0)
    pw_args.append((0, True))

    dens_sym = sympy.Piecewise(*pw_args)
    return dens_sym, r


# --- Approach 1: Direct integration of Piecewise ---
def earth_PREM_mass_sym_direct():
    dens_sym, r = earth_PREM_density_sym()
    # r = sp.Symbol("R", positive=True)
    r = sp.Symbol("r", positive=True)
    integrand = 4 * sp.pi * r**2 * dens_sym
    return sp.integrate(integrand, (r, 0, r))



if __name__ == "__main__":
    R_val = 6371  # test radius in km

    # --- Direct method timing ---
    t0 = time.time()
    M_direct_expr = earth_PREM_mass_sym_direct()
    t1 = time.time()
    # M_direct_val = M_direct_expr.subs({"R": R_val})
    # t2 = time.time()

    print("=== Direct integration ===")
    print(f"Symbolic build time: {t1 - t0:.6f} s")
    # print(f"Numeric eval time  : {t2 - t1:.6f} s")
    # print(f"Mass({R_val} km)   : {M_direct_val.evalf()} g (PREM units)")

    def sym_to_num(expr, R, R_val, t, t_val):
        expr_sub = expr.subs(R, R_val)
        expr_no_min = expr_sub.replace(
            lambda x: isinstance(x, sp.Min),
            lambda x: min(*[arg.evalf() for arg in x.args])
        )
        return float(expr_no_min.evalf())

    t = sp.Symbol("t", positive=True)
    r = sympy.Symbol("r", real=True, positive=True)
    value = sym_to_num(M_direct_expr, r, 6371, t, 0)
    # value = eval_with_min(M_pieceWise_expr, 6371, R)
    print(f"earth mass found: {value:.2e} g")
    print(f"time test {sym_to_num(M_direct_expr, r, 6371, t, 0)}")
    Phi = M_direct_expr
    d2Phi_dr2 = sympy.diff(Phi, r, 2)
    for R_val in [0, 1000, 6370]:
        print(d2Phi_dr2.evalf(subs={r: R_val}))


