#############################################################################
# DOP853 integrator
#############################################################################
#############################################################################
# Copyright (c) 2018, Henry Leung
# All rights reserved.
# This code is written with reference of the c-version dop853 here: http://www.unige.ch/~hairer/software.html
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import numpy

# time increment coefficients
c2 = 0.526001519587677318785587544488e-1
c3 = 0.789002279381515978178381316732e-1
c4 = 0.118350341907227396726757197510
c5 = 0.281649658092772603273242802490
c6 = 0.333333333333333333333333333333
c7 = 0.25
c8 = 0.307692307692307692307692307692
c9 = 0.651282051282051282051282051282
c10 = 0.6
c11 = 0.857142857142857142857142857142
c12 = 1
c13 = 1
c14 = 0.1
c15 = 0.2
c16 = 0.777777777777777777777777777778

# slope calculation coefficients
a21 = 5.26001519587677318785587544488e-2
a31 = 1.97250569845378994544595329183e-2
a32 = 5.91751709536136983633785987549e-2
a41 = 2.95875854768068491816892993775e-2
a43 = 8.87627564304205475450678981324e-2
a51 = 2.41365134159266685502369798665e-1
a53 = -8.84549479328286085344864962717e-1
a54 = 9.24834003261792003115737966543e-1
a61 = 3.7037037037037037037037037037e-2
a64 = 1.70828608729473871279604482173e-1
a65 = 1.25467687566822425016691814123e-1
a71 = 3.7109375e-2
a74 = 1.70252211019544039314978060272e-1
a75 = 6.02165389804559606850219397283e-2
a76 = -1.7578125e-2
a81 = 3.70920001185047927108779319836e-2
a84 = 1.70383925712239993810214054705e-1
a85 = 1.07262030446373284651809199168e-1
a86 = -1.53194377486244017527936158236e-2
a87 = 8.27378916381402288758473766002e-3
a91 = 6.24110958716075717114429577812e-1
a94 = -3.36089262944694129406857109825e0
a95 = -8.68219346841726006818189891453e-1
a96 = 2.75920996994467083049415600797e1
a97 = 2.01540675504778934086186788979e1
a98 = -4.34898841810699588477366255144e1
a101 = 4.77662536438264365890433908527e-1
a104 = -2.48811461997166764192642586468e0
a105 = -5.90290826836842996371446475743e-1
a106 = 2.12300514481811942347288949897e1
a107 = 1.52792336328824235832596922938e1
a108 = -3.32882109689848629194453265587e1
a109 = -2.03312017085086261358222928593e-2
a111 = -9.3714243008598732571704021658e-1
a114 = 5.18637242884406370830023853209e0
a115 = 1.09143734899672957818500254654e0
a116 = -8.14978701074692612513997267357e0
a117 = -1.85200656599969598641566180701e1
a118 = 2.27394870993505042818970056734e1
a119 = 2.49360555267965238987089396762e0
a1110 = -3.0467644718982195003823669022e0
a121 = 2.27331014751653820792359768449e0
a124 = -1.05344954667372501984066689879e1
a125 = -2.00087205822486249909675718444e0
a126 = -1.79589318631187989172765950534e1
a127 = 2.79488845294199600508499808837e1
a128 = -2.85899827713502369474065508674e0
a129 = -8.87285693353062954433549289258e0
a1210 = 1.23605671757943030647266201528e1
a1211 = 6.43392746015763530355970484046e-1
a141 = 5.61675022830479523392909219681e-2
a147 = 2.53500210216624811088794765333e-1
a148 = -2.46239037470802489917441475441e-1
a149 = -1.24191423263816360469010140626e-1
a1410 = 1.5329179827876569731206322685e-1
a1411 = 8.20105229563468988491666602057e-3
a1412 = 7.56789766054569976138603589584e-3
a1413 = -8.298e-3
a151 = 3.18346481635021405060768473261e-2
a156 = 2.83009096723667755288322961402e-2
a157 = 5.35419883074385676223797384372e-2
a158 = -5.49237485713909884646569340306e-2
a1511 = -1.08347328697249322858509316994e-4
a1512 = 3.82571090835658412954920192323e-4
a1513 = -3.40465008687404560802977114492e-4
a1514 = 1.41312443674632500278074618366e-1
a161 = -4.28896301583791923408573538692e-1
a166 = -4.69762141536116384314449447206e0
a167 = 7.68342119606259904184240953878e0
a168 = 4.06898981839711007970213554331e0
a169 = 3.56727187455281109270669543021e-1
a1613 = -1.39902416515901462129418009734e-3
a1614 = 2.9475147891527723389556272149e0
a1615 = -9.15095847217987001081870187138e0

# Final assembly coefficients
b1 = 5.42937341165687622380535766363e-2
b6 = 4.45031289275240888144113950566
b7 = 1.89151789931450038304281599044
b8 = -5.8012039600105847814672114227
b9 = 3.1116436695781989440891606237e-1
b10 = -1.52160949662516078556178806805e-1
b11 = 2.01365400804030348374776537501e-1
b12 = 4.47106157277725905176885569043e-2
bhh1 = 0.244094488188976377952755905512
bhh2 = 0.733846688281611857341361741547
bhh3 = 0.220588235294117647058823529412e-1

# Dense output coefficients
d41 = -0.84289382761090128651353491142e1
d46 = 0.56671495351937776962531783590
d47 = -0.30689499459498916912797304727e1
d48 = 0.23846676565120698287728149680e1
d49 = 0.21170345824450282767155149946e1
d410 = -0.87139158377797299206789907490
d411 = 0.22404374302607882758541771650e1
d412 = 0.63157877876946881815570249290
d413 = -0.88990336451333310820698117400e-1
d414 = 0.18148505520854727256656404962e2
d415 = -0.91946323924783554000451984436e1
d416 = -0.44360363875948939664310572000e1
d51 = 0.10427508642579134603413151009e2
d56 = 0.24228349177525818288430175319e3
d57 = 0.16520045171727028198505394887e3
d58 = -0.37454675472269020279518312152e3
d59 = -0.22113666853125306036270938578e2
d510 = 0.77334326684722638389603898808e1
d511 = -0.30674084731089398182061213626e2
d512 = -0.93321305264302278729567221706e1
d513 = 0.15697238121770843886131091075e2
d514 = -0.31139403219565177677282850411e2
d515 = -0.93529243588444783865713862664e1
d516 = 0.35816841486394083752465898540e2
d61 = 0.19985053242002433820987653617e2
d66 = -0.38703730874935176555105901742e3
d67 = -0.18917813819516756882830838328e3
d68 = 0.52780815920542364900561016686e3
d69 = -0.11573902539959630126141871134e2
d610 = 0.68812326946963000169666922661e1
d611 = -0.10006050966910838403183860980e1
d612 = 0.77771377980534432092869265740
d613 = -0.27782057523535084065932004339e1
d614 = -0.60196695231264120758267380846e2
d615 = 0.84320405506677161018159903784e2
d616 = 0.11992291136182789328035130030e2
d71 = -0.25693933462703749003312586129e2
d76 = -0.15418974869023643374053993627e3
d77 = -0.23152937917604549567536039109e3
d78 = 0.35763911791061412378285349910e3
d79 = 0.93405324183624310003907691704e2
d710 = -0.37458323136451633156875139351e2
d711 = 0.10409964950896230045147246184e3
d712 = 0.29840293426660503123344363579e2
d713 = -0.43533456590011143754432175058e2
d714 = 0.96324553959188282948394950600e2
d715 = -0.39177261675615439165231486172e2
d716 = -0.14972683625798562581422125276e3

# Error calculation coefficients
er1 = 0.1312004499419488073250102996e-1
er6 = -0.1225156446376204440720569753e1
er7 = -0.4957589496572501915214079952
er8 = 0.1664377182454986536961530415e1
er9 = -0.3503288487499736816886487290
er10 = 0.3341791187130174790297318841
er11 = 0.8192320648511571246570742613e-1
er12 = -0.2235530786388629525884427845e-1

# machine limit related info from numpy
unsigned_int_max = numpy.iinfo(numpy.int32).max
uround = numpy.finfo(numpy.float64).eps


def custom_sign(a, b):
    return numpy.fabs(a) if b > 0.0 else -numpy.fabs(a)


def hinit(func, x, t, pos_neg, f0, iord, hmax, rtol, atol, args):
    """
    Estimate initial step size
    """
    sk = atol + rtol * numpy.fabs(x)
    dnf = numpy.sum(numpy.square(f0 / sk), axis=0)
    dny = numpy.sum(numpy.square(x / sk), axis=0)

    h = numpy.sqrt(dny / dnf) * 0.01

    h = numpy.min([h, numpy.fabs(hmax)])
    h = custom_sign(h, pos_neg)

    # perform an explicit Euler step
    xx1 = x + h * f0
    f1 = numpy.array(func(xx1, t[0] + h, *args))

    # estimate the second derivative of the solution
    der2 = numpy.sum(numpy.square((f1 - f0) / sk), axis=0)
    der2 = numpy.sqrt(der2) / h

    # step size is computed such that h ** iord * max_d(norm(f0), norm(der2)) = 0.01
    der12 = numpy.max([numpy.fabs(der2), numpy.sqrt(dnf)])
    h1 = numpy.power(0.01 / der12, 1.0 / iord)

    h = numpy.min(
        [100.0 * numpy.fabs(h), numpy.min([numpy.fabs(h1), numpy.fabs(hmax)])]
    )

    return custom_sign(h, pos_neg), f0, f1, xx1


def dense_output(t_current, t_old, h_current, rcont):
    """
    Dense output function, basically extrapolating
    """
    # initialization
    s = (t_current - t_old) / h_current
    s1 = 1.0 - s

    return rcont[0] + s * (
        rcont[1]
        + s1
        * (
            rcont[2]
            + s
            * (
                rcont[3]
                + s1 * (rcont[4] + s * (rcont[5] + s1 * (rcont[6] + s * rcont[7])))
            )
        )
    )


def dopri853core(
    n, func, x, t, hmax, h, rtol, atol, nmax, safe, beta, fac1, fac2, pos_neg, args
):
    """
    Core of DOP8(5, 3) integration
    """
    # array to store the result
    result = numpy.zeros((len(t), n))

    # initial preparations
    facold = 1.0e-4
    expo1 = 1.0 / 8.0 - beta * 0.2
    facc1 = 1.0 / fac1
    facc2 = 1.0 / fac2

    k1 = numpy.array(func(x, t[0], *args))
    hmax = numpy.fabs(hmax)
    iord = 8

    if h == 0.0:  # estimate initial time step
        h, k1, k2, k3 = hinit(func, x, t, pos_neg, k1, iord, hmax, rtol, atol, args)

    reject = 0
    t_current = t[
        0
    ]  # store current integration time internally (not the current time wanted by user!!)
    t_old = t[0]
    finished_user_t_ii = 0  # times indices wanted by user

    result[0, :] = x

    # basic integration step
    while (
        finished_user_t_ii < len(t) - 1
    ):  # check if the current computed time indices less than total inices needed
        # keep time step not too small
        h = pos_neg * numpy.max([numpy.fabs(h), 1e3 * uround])

        # the twelve stages
        xx1 = x + h * a21 * k1
        k2 = numpy.array(func(xx1, t_current + c2 * h, *args))

        xx1 = x + h * (a31 * k1 + a32 * k2)
        k3 = numpy.array(func(xx1, t_current + c3 * h, *args))

        xx1 = x + h * (a41 * k1 + a43 * k3)
        k4 = numpy.array(func(xx1, t_current + c4 * h, *args))

        xx1 = x + h * (a51 * k1 + a53 * k3 + a54 * k4)
        k5 = numpy.array(func(xx1, t_current + c5 * h, *args))

        xx1 = x + h * (a61 * k1 + a64 * k4 + a65 * k5)
        k6 = numpy.array(func(xx1, t_current + c6 * h, *args))

        xx1 = x + h * (a71 * k1 + a74 * k4 + a75 * k5 + a76 * k6)
        k7 = numpy.array(func(xx1, t_current + c7 * h, *args))

        xx1 = x + h * (a81 * k1 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7)
        k8 = numpy.array(func(xx1, t_current + c8 * h, *args))

        xx1 = x + h * (a91 * k1 + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98 * k8)
        k9 = numpy.array(func(xx1, t_current + c9 * h, *args))

        xx1 = x + h * (
            a101 * k1
            + a104 * k4
            + a105 * k5
            + a106 * k6
            + a107 * k7
            + a108 * k8
            + a109 * k9
        )
        k10 = numpy.array(func(xx1, t_current + c10 * h, *args))

        xx1 = x + h * (
            a111 * k1
            + a114 * k4
            + a115 * k5
            + a116 * k6
            + a117 * k7
            + a118 * k8
            + a119 * k9
            + a1110 * k10
        )
        k2 = numpy.array(func(xx1, t_current + c11 * h, *args))

        xx1 = x + h * (
            a121 * k1
            + a124 * k4
            + a125 * k5
            + a126 * k6
            + a127 * k7
            + a128 * k8
            + a129 * k9
            + a1210 * k10
            + a1211 * k2
        )

        t_old_older = numpy.copy(t_old)
        t_old = numpy.copy(t_current)
        t_current += h

        k3 = numpy.array(func(xx1, t_current, *args))

        k4 = (
            b1 * k1
            + b6 * k6
            + b7 * k7
            + b8 * k8
            + b9 * k9
            + b10 * k10
            + b11 * k2
            + b12 * k3
        )
        k5 = x + h * k4

        # error estimation
        sk = atol + rtol * numpy.max([numpy.fabs(x), numpy.fabs(k5)], axis=0)
        erri = k4 - bhh1 * k1 - bhh2 * k9 - bhh3 * k3
        err2 = numpy.sum(numpy.square(erri / sk), axis=0)
        erri = (
            er1 * k1
            + er6 * k6
            + er7 * k7
            + er8 * k8
            + er9 * k9
            + er10 * k10
            + er11 * k2
            + er12 * k3
        )
        err = numpy.sum(numpy.square(erri / sk), axis=0)

        deno = err + 0.01 * err2
        deno = 1.0 if deno <= 0.0 else deno
        err = numpy.fabs(h) * err * numpy.sqrt(1.0 / (deno * n))

        # computation of hnew
        fac11 = numpy.power(err, expo1)

        # Lund-stabilization
        fac = fac11 / pow(facold, beta)

        # we require fac1 <= hnew / h <= fac2
        fac = numpy.max([facc2, numpy.min([facc1, fac / safe])])
        hnew = h / fac

        if err <= 1.0:
            # step accepted
            facold = numpy.max([err, 1.0e-4])
            k4 = numpy.array(func(k5, t_current, *args))

            # final preparation for dense output
            rcont1 = numpy.copy(x)
            xdiff = k5 - x
            rcont2 = xdiff
            bspl = h * k1 - xdiff
            rcont3 = numpy.copy(bspl)
            rcont4 = xdiff - h * k4 - bspl
            rcont5 = (
                d41 * k1
                + d46 * k6
                + d47 * k7
                + d48 * k8
                + d49 * k9
                + d410 * k10
                + d411 * k2
                + d412 * k3
            )
            rcont6 = (
                d51 * k1
                + d56 * k6
                + d57 * k7
                + d58 * k8
                + d59 * k9
                + d510 * k10
                + d511 * k2
                + d512 * k3
            )
            rcont7 = (
                d61 * k1
                + d66 * k6
                + d67 * k7
                + d68 * k8
                + d69 * k9
                + d610 * k10
                + d611 * k2
                + d612 * k3
            )
            rcont8 = (
                d71 * k1
                + d76 * k6
                + d77 * k7
                + d78 * k8
                + d79 * k9
                + d710 * k10
                + d711 * k2
                + d712 * k3
            )

            # the next three function evaluations
            xx1 = x + h * (
                a141 * k1
                + a147 * k7
                + a148 * k8
                + a149 * k9
                + a1410 * k10
                + a1411 * k2
                + a1412 * k3
                + a1413 * k4
            )
            k10 = numpy.array(func(xx1, t_old + c14 * h, *args))
            xx1 = x + h * (
                a151 * k1
                + a156 * k6
                + a157 * k7
                + a158 * k8
                + a1511 * k2
                + a1512 * k3
                + a1513 * k4
                + a1514 * k10
            )
            k2 = numpy.array(func(xx1, t_old + c15 * h, *args))
            xx1 = x + h * (
                a161 * k1
                + a166 * k6
                + a167 * k7
                + a168 * k8
                + a169 * k9
                + a1613 * k4
                + a1614 * k10
                + a1615 * k2
            )
            k3 = numpy.array(func(xx1, t_old + c16 * h, *args))

            # final preparation
            rcont5 = h * (rcont5 + d413 * k4 + d414 * k10 + d415 * k2 + d416 * k3)
            rcont6 = h * (rcont6 + d513 * k4 + d514 * k10 + d515 * k2 + d516 * k3)
            rcont7 = h * (rcont7 + d613 * k4 + d614 * k10 + d615 * k2 + d616 * k3)
            rcont8 = h * (rcont8 + d713 * k4 + d714 * k10 + d715 * k2 + d716 * k3)

            k1 = numpy.copy(k4)
            x = numpy.copy(k5)

            # loop for dense output in this time slot
            while (finished_user_t_ii < len(t) - 1) and (
                pos_neg * t[finished_user_t_ii + 1] < pos_neg * t_current
            ):
                result[finished_user_t_ii + 1, :] = dense_output(
                    t[finished_user_t_ii + 1],
                    t_old,
                    h,
                    [rcont1, rcont2, rcont3, rcont4, rcont5, rcont6, rcont7, rcont8],
                )
                finished_user_t_ii += 1

            if numpy.fabs(hnew) > hmax:
                hnew = pos_neg * hmax
            if reject:
                hnew = pos_neg * numpy.min([numpy.fabs(hnew), numpy.fabs(h)])

            reject = 0
        else:
            # step rejected since error too big
            hnew = h / numpy.min([facc1, fac11 / safe])
            reject = 1

            # reverse time increment since error rejected
            t_current = numpy.copy(t_old)
            t_old = numpy.copy(t_old_older)

        h = numpy.copy(hnew)  # current h

    return result


def dop853(
    func=None,
    x=None,
    t=None,
    hmax=0.0,
    h=0.0,
    rtol=1e-12,
    atol=1e-12,
    nmax=int(1e8),
    safe=0.9,
    beta=0.0,
    fac1=0.333,
    fac2=6.0,
    args=(),
):
    """
    Solve a system of ordinary differential equations using the DOP853 method, an explicit Runge-Kutta method of order 8(5,3) due to Dormand & Prince with step size control and dense output.

    Parameters
    ----------
    func : callable
        function of the differential equation, usually take func([position, velocity], time) and return velocity, acceleration
    x : float or ndarray
        initial x, usually is [position, velocity]
    t : ndarray
        set of times at which one wants the result
    hmax : float
        Maximal step size, default 0. which will be internally as t[-1]-t[0]
    h : float
        Initial step size, default 0. which will be computed by the function hinit()
    rtol : float or ndarray
        Relative error tolerances, default 1e-8
    atol : float or ndarray
        Absolute error tolerances, default 1e-8
    nmax : int
        Maximum number of steps, default 1e8
    safe : float
        Safety factor in the step size prediction, default 0.9
    beta : float
        The "beta" for stabilized step size control (see section IV.2 of the reference). Larger values for beta (e.g.<= 0.1) make the step size control more stable. Default beta=0.
    fac1, fac2 : float
        Parameters for step size selection; the new step size is chosen subject to the restriction fac1 <= hnew/hold <= fac2

    Returns
    -------
    ndarray
        integrated result

    Notes
    -----
    - 2018-11-23 - Written - Henry Leung (University of Toronto)

    References
    ----------
    - E. Hairer, S.P. Norsett and G. Wanner, Solving ordinary differential equations I, non-stiff problems, 2nd edition, Springer Series in Computational Mathematics, Springer-Verlag (1993).
    """
    # initialization
    n = len(x)

    # maximal step size, default a big one
    if hmax == 0.0:
        hmax = t[-1] - t[0]

    # see if integrate forward or integrate backward
    pos_neg = custom_sign(1.0, t[-1] - t[0])

    result = dopri853core(
        n, func, x, t, hmax, h, rtol, atol, nmax, safe, beta, fac1, fac2, pos_neg, args
    )

    return result
