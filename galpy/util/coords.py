###############################################################################
#
#   coords: module for coordinate transformations between the equatorial
#           and Galactic coordinate frame
#
#
#      Main included functions:
#            radec_to_lb
#            lb_to_radec
#            radec_to_custom
#            custom_to_radec
#            lbd_to_XYZ
#            XYZ_to_lbd
#            rectgal_to_sphergal
#            sphergal_to_rectgal
#            vrpmllpmbb_to_vxvyvz
#            vxvyvz_to_vrpmllpmbb
#            pmrapmdec_to_pmllpmbb
#            pmllpmbb_to_pmrapmdec
#            pmrapmdec_to_custom
#            custom_to_pmrapmdec
#            cov_pmrapmdec_to_pmllpmbb
#            cov_dvrpmllbb_to_vxyz
#            XYZ_to_galcenrect
#            XYZ_to_galcencyl
#            galcenrect_to_XYZ
#            galcencyl_to_XYZ
#            rect_to_cyl
#            cyl_to_rect
#            rect_to_cyl_vec
#            cyl_to_rect_vec
#            vxvyvz_to_galcenrect
#            vxvyvz_to_galcencyl
#            galcenrect_to_vxvyvz
#            galcencyl_to_vxvyvz
#            dl_to_rphi_2d
#            rphi_to_dl_2d
#            Rz_to_coshucosv
#            Rz_to_uv
#            uv_to_Rz
#            Rz_to_lambdanu
#            Rz_to_lambdanu_jac
#            Rz_to_lambdanu_hess
#            lambdanu_to_Rz
#
##############################################################################
#############################################################################
# Copyright (c) 2010 - 2020, Jo Bovy
# All rights reserved.
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
from functools import wraps

import numpy

from ..util import _rotate_to_arbitrary_vector
from ..util._optional_deps import _APY_LOADED
from ..util.config import __config__

_APY_COORDS = __config__.getboolean("astropy", "astropy-coords")
_APY_COORDS *= _APY_LOADED
_DEGTORAD = numpy.pi / 180.0
if _APY_LOADED:
    import astropy.coordinates as apycoords
    from astropy import units

    _K = (
        (1.0 * units.mas / units.yr)
        .to(units.km / units.s / units.kpc, equivalencies=units.dimensionless_angles())
        .value
    )
else:
    _K = 4.74047
# numpy 1.14 einsum bug causes astropy conversions to fail in py2.7 -> turn off
if _APY_COORDS:
    ra, dec = numpy.array([192.25 * _DEGTORAD]), numpy.array([27.4 * _DEGTORAD])
    c = apycoords.SkyCoord(
        ra * units.rad, dec * units.rad, equinox="B1950", frame="fk4"
    )
    # This conversion fails bc of einsum bug
    try:
        c = c.transform_to(apycoords.Galactic)
    except TypeError:  # pragma: no cover
        _APY_COORDS = False


def scalarDecorator(func):
    """Decorator to return scalar outputs as a set"""

    @wraps(func)
    def scalar_wrapper(*args, **kwargs):
        if numpy.array(args[0]).shape == ():
            scalarOut = True
            newargs = ()
            for ii in range(len(args)):
                newargs = newargs + (numpy.array([args[ii]]),)
            args = newargs
        else:
            scalarOut = False
        result = func(*args, **kwargs)
        if scalarOut:
            out = ()
            for ii in range(result.shape[1]):
                out = out + (result[0, ii],)
            return out
        else:
            return result

    return scalar_wrapper


def degreeDecorator(inDegrees, outDegrees):
    """
    Decorator to transform angles from and to degrees if necessary

    Parameters
    ----------
    inDegrees : list
        List specifying indices of angle arguments (e.g., [0,1])
    outDegrees : list
        Same as inDegrees, but for function return

    Returns
    -------
    function
        Wrapped function

    Notes
    -----
    - ____-__-__ - Written - Bovy
    - 2019-03-02 - speedup - Nathaniel Starkman (UofT)
    """

    # (modified) old degree decorator
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            isdeg = kwargs.get("degree", False)
            if isdeg:
                args = [
                    arg * numpy.pi / 180 if i in inDegrees else arg
                    for i, arg in enumerate(args)
                ]
            out = func(*args, **kwargs)
            if isdeg:
                for i in outDegrees:
                    out[:, i] *= 180.0 / numpy.pi
            return out

        return wrapped

    return wrapper


@scalarDecorator
@degreeDecorator([0, 1], [0, 1])
def radec_to_lb(ra, dec, degree=False, epoch=2000.0):
    """
    Transform from equatorial coordinates to Galactic coordinates

    Parameters
    ----------
    ra : float or numpy.ndarray
        Right ascension
    dec : float or numpy.ndarray
        Declination
    degree : bool, optional
        If True, ra and dec are given in degree and l and b will be as well
    epoch : float, optional
        Epoch of ra,dec (right now only 2000.0 and 1950.0 are supported when not using astropy's transformations internally; when internally using astropy's coordinate transformations, epoch can be None for ICRS, 'JXXXX' for FK5, and 'BXXXX' for FK4)

    Returns
    -------
    tuple or numpy.ndarray
        Galactic longitude and latitude

    Notes
    -----
    - 2009-11-12 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    - 2016-05-13 - Added support for using astropy's coordinate transformations and for non-standard epochs - Bovy (UofT)
    """
    if _APY_COORDS:
        epoch, frame = _parse_epoch_frame_apy(epoch)
        c = apycoords.SkyCoord(
            ra * units.rad, dec * units.rad, equinox=epoch, frame=frame
        )
        c = c.transform_to(apycoords.Galactic)
        return numpy.array([c.l.to(units.rad).value, c.b.to(units.rad).value]).T
    # First calculate the transformation matrix T
    theta, dec_ngp, ra_ngp = get_epoch_angles(epoch)
    T = numpy.dot(
        numpy.array(
            [
                [numpy.cos(theta), numpy.sin(theta), 0.0],
                [numpy.sin(theta), -numpy.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        numpy.dot(
            numpy.array(
                [
                    [-numpy.sin(dec_ngp), 0.0, numpy.cos(dec_ngp)],
                    [0.0, 1.0, 0.0],
                    [numpy.cos(dec_ngp), 0.0, numpy.sin(dec_ngp)],
                ]
            ),
            numpy.array(
                [
                    [numpy.cos(ra_ngp), numpy.sin(ra_ngp), 0.0],
                    [-numpy.sin(ra_ngp), numpy.cos(ra_ngp), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
    )
    # Whether to use degrees and scalar input is handled by decorators
    XYZ = numpy.array(
        [numpy.cos(dec) * numpy.cos(ra), numpy.cos(dec) * numpy.sin(ra), numpy.sin(dec)]
    )
    galXYZ = numpy.dot(T, XYZ)
    galXYZ[2][galXYZ[2] > 1.0] = 1.0
    galXYZ[2][galXYZ[2] < -1.0] = -1.0
    b = numpy.arcsin(galXYZ[2])
    l = numpy.arctan2(galXYZ[1] / numpy.cos(b), galXYZ[0] / numpy.cos(b))
    l[l < 0.0] += 2.0 * numpy.pi
    out = numpy.array([l, b])
    return out.T


@scalarDecorator
@degreeDecorator([0, 1], [0, 1])
def lb_to_radec(l, b, degree=False, epoch=2000.0):
    """
    Transform from Galactic coordinates to equatorial coordinates

    Parameters
    ----------
    l : float or numpy.ndarray
        Galactic longitude
    b : float or numpy.ndarray
        Galactic latitude
    degree : bool, optional
        If True, l and b are given in degree and ra and dec will be as well
    epoch : float, optional
        Epoch of ra,dec (right now only 2000.0 and 1950.0 are supported when not using astropy's transformations internally; when internally using astropy's coordinate transformations, epoch can be None for ICRS, 'JXXXX' for FK5, and 'BXXXX' for FK4)

    Returns
    -------
    tuple or numpy.ndarray
        Right ascension and declination

    Notes
    -----
    - 2010-04-07 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    - 2016-05-13 - Added support for using astropy's coordinate transformations and for non-standard epochs - Bovy (UofT)
    """
    if _APY_COORDS:
        epoch, frame = _parse_epoch_frame_apy(epoch)
        c = apycoords.SkyCoord(l * units.rad, b * units.rad, frame="galactic")
        if not epoch is None and "J" in epoch:
            c = c.transform_to(apycoords.FK5(equinox=epoch))
        elif not epoch is None and "B" in epoch:
            c = c.transform_to(apycoords.FK4(equinox=epoch))
        else:
            c = c.transform_to(apycoords.ICRS)
        return numpy.array([c.ra.to(units.rad).value, c.dec.to(units.rad).value]).T
    # First calculate the transformation matrix T'
    theta, dec_ngp, ra_ngp = get_epoch_angles(epoch)
    T = numpy.dot(
        numpy.array(
            [
                [numpy.cos(ra_ngp), -numpy.sin(ra_ngp), 0.0],
                [numpy.sin(ra_ngp), numpy.cos(ra_ngp), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        numpy.dot(
            numpy.array(
                [
                    [-numpy.sin(dec_ngp), 0.0, numpy.cos(dec_ngp)],
                    [0.0, 1.0, 0.0],
                    [numpy.cos(dec_ngp), 0.0, numpy.sin(dec_ngp)],
                ]
            ),
            numpy.array(
                [
                    [numpy.cos(theta), numpy.sin(theta), 0.0],
                    [numpy.sin(theta), -numpy.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
    )
    # Whether to use degrees and scalar input is handled by decorators
    XYZ = numpy.array(
        [numpy.cos(b) * numpy.cos(l), numpy.cos(b) * numpy.sin(l), numpy.sin(b)]
    )
    eqXYZ = numpy.dot(T, XYZ)
    dec = numpy.arcsin(eqXYZ[2])
    ra = numpy.arctan2(eqXYZ[1], eqXYZ[0])
    ra[ra < 0.0] += 2.0 * numpy.pi
    return numpy.array([ra, dec]).T


@scalarDecorator
@degreeDecorator([0, 1], [])
def lbd_to_XYZ(l, b, d, degree=False):
    """
    Transform from spherical Galactic coordinates to rectangular Galactic coordinates (works with vector inputs)

    Parameters
    ----------
    l : float or numpy.ndarray
        Galactic longitude (rad)
    b : float or numpy.ndarray
        Galactic latitude (rad)
    d : float or numpy.ndarray
        Distance (arbitrary units)
    degree : bool, optional
        If True, l and b are in degrees. Default is False.

    Returns
    -------
    tuple or numpy.ndarray
        [X,Y,Z] in whatever units d was in. For vector inputs [:,3]

    Notes
    -----
    - 2009-10-24 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)

    """
    # Whether to use degrees and scalar input is handled by decorators
    return numpy.array(
        [
            d * numpy.cos(b) * numpy.cos(l),
            d * numpy.cos(b) * numpy.sin(l),
            d * numpy.sin(b),
        ]
    ).T


def rectgal_to_sphergal(X, Y, Z, vx, vy, vz, degree=False):
    """
    Transform phase-space coordinates in rectangular Galactic coordinates to spherical Galactic coordinates (can take vector inputs)

    Parameters
    ----------
    X : float or numpy.ndarray
        Component towards the Galactic Center (kpc)
    Y : float or numpy.ndarray
        Component in the direction of Galactic rotation (kpc)
    Z : float or numpy.ndarray
        Component towards the North Galactic Pole (kpc)
    vx : float or numpy.ndarray
        Velocity towards the Galactic Center (km/s)
    vy : float or numpy.ndarray
        Velocity in the direction of Galactic rotation (km/s)
    vz : float or numpy.ndarray
        Velocity towards the North Galactic Pole (km/s)
    degree : bool, optional
        If True, return l and b in degrees. Default is False.

    Returns
    -------
    tuple or numpy.ndarray
        [l,b,d,vr,pmll x cos(b),pmbb] in (rad,rad,kpc,km/s,mas/yr,mas/yr). For vector inputs [:,6]

    Notes
    -----
    - 2009-10-25 - Written - Bovy (NYU)
    """
    lbd = XYZ_to_lbd(X, Y, Z, degree=degree)
    vrpmllpmbb = vxvyvz_to_vrpmllpmbb(vx, vy, vz, X, Y, Z, XYZ=True)
    if numpy.array(X).shape == ():
        return numpy.array(
            [lbd[0], lbd[1], lbd[2], vrpmllpmbb[0], vrpmllpmbb[1], vrpmllpmbb[2]]
        )
    else:
        out = numpy.zeros((len(X), 6))
        out[:, 0:3] = lbd
        out[:, 3:6] = vrpmllpmbb
        return out


def sphergal_to_rectgal(l, b, d, vr, pmll, pmbb, degree=False):
    """
    Transform phase-space coordinates in spherical Galactic coordinates to rectangular Galactic coordinates (can take vector inputs)

    Parameters
    ----------
    l : float or numpy.ndarray
        Galactic longitude
    b : float or numpy.ndarray
        Galactic latitude
    d : float or numpy.ndarray
        Distance (kpc)
    vr : float or numpy.ndarray
        Line-of-sight velocity (km/s)
    pmll : float or numpy.ndarray
        Proper motion in the Galactic longitude direction (mu_l * cos(b)) (mas/yr)
    pmbb : float or numpy.ndarray
        Proper motion in the Galactic latitude (mas/yr)
    degree : bool, optional
        If True, l and b are in degrees. Default is False.

    Returns
    -------
    tuple or numpy.ndarray
        [X,Y,Z,vx,vy,vz] in (kpc,kpc,kpc,km/s,km/s,km/s). For vector inputs [:,6]

    Notes
    -----
    - 2009-10-25 - Written - Bovy (NYU)
    """
    XYZ = lbd_to_XYZ(l, b, d, degree=degree)
    vxvyvz = vrpmllpmbb_to_vxvyvz(vr, pmll, pmbb, l, b, d, XYZ=False, degree=degree)
    if numpy.array(l).shape == ():
        return numpy.array([XYZ[0], XYZ[1], XYZ[2], vxvyvz[0], vxvyvz[1], vxvyvz[2]])
    else:
        out = numpy.zeros((len(l), 6))
        out[:, 0:3] = XYZ
        out[:, 3:6] = vxvyvz
        return out


@scalarDecorator
@degreeDecorator([3, 4], [])
def vrpmllpmbb_to_vxvyvz(vr, pmll, pmbb, l, b, d, XYZ=False, degree=False):
    """
    Transform velocities in the spherical Galactic coordinate frame to the rectangular Galactic coordinate frame (can take vector inputs)

    Parameters
    ----------
    vr : float or numpy.ndarray
        Line-of-sight velocity (km/s)
    pmll : float or numpy.ndarray
        Proper motion in the Galactic longitude direction (mu_l * cos(b)) (mas/yr)
    pmbb : float or numpy.ndarray
        Proper motion in the Galactic latitude (mas/yr)
    l : float or numpy.ndarray
        Galactic longitude
    b : float or numpy.ndarray
        Galactic latitude
    d : float or numpy.ndarray
        Distance (kpc)
    XYZ : bool, optional
        If True, then l,b,d is actually X,Y,Z (rectangular Galactic coordinates). Default is False.
    degree : bool, optional
        If True, l and b are in degrees. Default is False.

    Returns
    -------
    tuple or numpy.ndarray
        [vx,vy,vz] in (km/s,km/s,km/s). For vector inputs [:,3]

    Notes
    -----
    - 2009-10-24 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    """
    # Whether to use degrees and scalar input is handled by decorators
    if XYZ:  # undo the incorrect conversion that the decorator did
        if degree:
            l *= 180.0 / numpy.pi
            b *= 180.0 / numpy.pi
        lbd = XYZ_to_lbd(l, b, d, degree=False)
        l = lbd[:, 0]
        b = lbd[:, 1]
        d = lbd[:, 2]
    R = numpy.zeros((3, 3, len(l)))
    R[0, 0] = numpy.cos(l) * numpy.cos(b)
    R[1, 0] = -numpy.sin(l)
    R[2, 0] = -numpy.cos(l) * numpy.sin(b)
    R[0, 1] = numpy.sin(l) * numpy.cos(b)
    R[1, 1] = numpy.cos(l)
    R[2, 1] = -numpy.sin(l) * numpy.sin(b)
    R[0, 2] = numpy.sin(b)
    R[2, 2] = numpy.cos(b)
    invr = numpy.array(
        [
            [vr, vr, vr],
            [d * pmll * _K, d * pmll * _K, d * pmll * _K],
            [d * pmbb * _K, d * pmbb * _K, d * pmbb * _K],
        ]
    )
    return (R.T * invr.T).sum(-1)


@scalarDecorator
@degreeDecorator([3, 4], [])
def vxvyvz_to_vrpmllpmbb(vx, vy, vz, l, b, d, XYZ=False, degree=False):
    """
    Transform velocities in the rectangular Galactic coordinate frame to the spherical Galactic coordinate frame (can take vector inputs)

    Parameters
    ----------
    vx : float or numpy.ndarray
        Velocity towards the Galactic Center (km/s)
    vy : float or numpy.ndarray
        Velocity in the direction of Galactic rotation (km/s)
    vz : float or numpy.ndarray
        Velocity towards the North Galactic Pole (km/s)
    l : float or numpy.ndarray
        Galactic longitude
    b : float or numpy.ndarray
        Galactic latitude
    d : float or numpy.ndarray
        Distance (kpc)
    XYZ : bool, optional
        If True, then l,b,d is actually X,Y,Z (rectangular Galactic coordinates). Default is False.
    degree : bool, optional
        If True, l and b are in degrees. Default is False.

    Returns
    -------
    tuple or numpy.ndarray
        [vr,pmll x cos(b),pmbb] in (km/s,mas/yr,mas/yr). For vector inputs [:,3]

    Notes
    -----
    - 2009-10-24 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    """
    # Whether to use degrees and scalar input is handled by decorators
    if XYZ:  # undo the incorrect conversion that the decorator did
        if degree:
            l *= 180.0 / numpy.pi
            b *= 180.0 / numpy.pi
        lbd = XYZ_to_lbd(l, b, d, degree=False)
        l = lbd[:, 0]
        b = lbd[:, 1]
        d = lbd[:, 2]
    R = numpy.zeros((3, 3, len(l)))
    R[0, 0] = numpy.cos(l) * numpy.cos(b)
    R[0, 1] = -numpy.sin(l)
    R[0, 2] = -numpy.cos(l) * numpy.sin(b)
    R[1, 0] = numpy.sin(l) * numpy.cos(b)
    R[1, 1] = numpy.cos(l)
    R[1, 2] = -numpy.sin(l) * numpy.sin(b)
    R[2, 0] = numpy.sin(b)
    R[2, 2] = numpy.cos(b)
    invxyz = numpy.array([[vx, vx, vx], [vy, vy, vy], [vz, vz, vz]])
    vrvlvb = (R.T * invxyz.T).sum(-1)
    vrvlvb[:, 1] /= d * _K
    vrvlvb[:, 2] /= d * _K
    return vrvlvb


@scalarDecorator
@degreeDecorator([], [0, 1])
def XYZ_to_lbd(X, Y, Z, degree=False):
    """
    Transform from rectangular Galactic coordinates to spherical Galactic coordinates (works with vector inputs)

    Parameters
    ----------
    X : float or numpy.ndarray
        Component towards the Galactic Center (in kpc; though this obviously does not matter))
    Y : float or numpy.ndarray
        Component in the direction of Galactic rotation (in kpc)
    Z : float or numpy.ndarray
        Component towards the North Galactic Pole (kpc)
    degree : bool, optional
        If True, return l and b in degrees (default is False)

    Returns
    -------
    tuple or numpy.ndarray
        [l,b,d] in (rad or degree,rad or degree,kpc); for vector inputs [:,3]

    Notes
    -----
    - 2009-10-24 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    """
    # Whether to use degrees and scalar input is handled by decorators
    d = numpy.sqrt(X**2.0 + Y**2.0 + Z**2.0)
    b = numpy.arcsin(Z / d)
    l = numpy.arctan2(Y, X)
    l[l < 0.0] += 2.0 * numpy.pi
    out = numpy.empty((len(d), 3))
    out[:, 0] = l
    out[:, 1] = b
    out[:, 2] = d
    return out


@scalarDecorator
@degreeDecorator([2, 3], [])
def pmrapmdec_to_pmllpmbb(pmra, pmdec, ra, dec, degree=False, epoch=2000.0):
    """
    Rotate proper motions in (ra,dec) into proper motions in (l,b)

    Parameters
    ----------
    pmra : float or numpy.ndarray
        Proper motion in ra (multiplied with cos(dec)) (mas/yr)
    pmdec : float or numpy.ndarray
        Proper motion in dec (mas/yr)
    ra : float or numpy.ndarray
        Right ascension
    dec : float or numpy.ndarray
        Declination
    degree : bool, optional
        If True, ra and dec are given in degrees (default=False)
    epoch : float, optional
        Epoch of ra,dec (right now only 2000.0 and 1950.0 are supported when not using astropy's transformations internally; when internally using astropy's coordinate transformations, epoch can be None for ICRS, 'JXXXX' for FK5, and 'BXXXX' for FK4)

    Returns
    -------
    tuple or numpy.ndarray
        [pmll x cos(b),pmbb] in (mas/yr,mas/yr). For vector inputs [:,2]

    Notes
    -----
    - 2010-04-07 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    """
    theta, dec_ngp, ra_ngp = get_epoch_angles(epoch)
    # Whether to use degrees and scalar input is handled by decorators
    dec[dec == dec_ngp] += 10.0**-16  # deal w/ pole.
    sindec_ngp = numpy.sin(dec_ngp)
    cosdec_ngp = numpy.cos(dec_ngp)
    sindec = numpy.sin(dec)
    cosdec = numpy.cos(dec)
    sinrarangp = numpy.sin(ra - ra_ngp)
    cosrarangp = numpy.cos(ra - ra_ngp)
    # These were replaced by Poleski (2013)'s equivalent form that is better at the poles
    # cosphi= (sindec_ngp-sindec*sinb)/cosdec/cosb
    # sinphi= sinrarangp*cosdec_ngp/cosb
    cosphi = sindec_ngp * cosdec - cosdec_ngp * sindec * cosrarangp
    sinphi = sinrarangp * cosdec_ngp
    norm = numpy.sqrt(cosphi**2.0 + sinphi**2.0)
    cosphi /= norm
    sinphi /= norm
    return (
        numpy.array([[cosphi, -sinphi], [sinphi, cosphi]]).T
        * numpy.array([[pmra, pmra], [pmdec, pmdec]]).T
    ).sum(-1)


@scalarDecorator
@degreeDecorator([2, 3], [])
def pmllpmbb_to_pmrapmdec(pmll, pmbb, l, b, degree=False, epoch=2000.0):
    """
    Rotate proper motions in (l,b) into proper motions in (ra,dec)

    Parameters
    ----------
    pmll : float or numpy.ndarray
        Proper motion in l (multiplied with cos(b)) (mas/yr)
    pmbb : float or numpy.ndarray
        Proper motion in b (mas/yr)
    l : float or numpy.ndarray
        Galactic longitude
    b : float or numpy.ndarray
        Galactic latitude
    degree : bool, optional
        If True, l and b are given in degrees (default=False)
    epoch : float, optional
        Epoch of ra,dec (right now only 2000.0 and 1950.0 are supported when not using astropy's transformations internally; when internally using astropy's coordinate transformations, epoch can be None for ICRS, 'JXXXX' for FK5, and 'BXXXX' for FK4)

    Returns
    -------
    tuple or numpy.ndarray
        [pmra x cos(dec),pmdec] in (mas/yr,mas/yr). For vector inputs [:,2]

    Notes
    -----
    - 2010-04-07 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    """
    theta, dec_ngp, ra_ngp = get_epoch_angles(epoch)
    # Whether to use degrees and scalar input is handled by decorators
    radec = lb_to_radec(l, b, degree=False, epoch=epoch)
    ra = radec[:, 0]
    dec = radec[:, 1]
    dec[dec == dec_ngp] += 10.0**-16  # deal w/ pole.
    sindec_ngp = numpy.sin(dec_ngp)
    cosdec_ngp = numpy.cos(dec_ngp)
    sindec = numpy.sin(dec)
    cosdec = numpy.cos(dec)
    sinrarangp = numpy.sin(ra - ra_ngp)
    cosrarangp = numpy.cos(ra - ra_ngp)
    # These were replaced by Poleski (2013)'s equivalent form that is better at the poles
    # cosphi= (sindec_ngp-sindec*sinb)/cosdec/cosb
    # sinphi= sinrarangp*cosdec_ngp/cosb
    cosphi = sindec_ngp * cosdec - cosdec_ngp * sindec * cosrarangp
    sinphi = sinrarangp * cosdec_ngp
    norm = numpy.sqrt(cosphi**2.0 + sinphi**2.0)
    cosphi /= norm
    sinphi /= norm
    return (
        numpy.array([[cosphi, sinphi], [-sinphi, cosphi]]).T
        * numpy.array([[pmll, pmll], [pmbb, pmbb]]).T
    ).sum(-1)


def cov_pmrapmdec_to_pmllpmbb(cov_pmradec, ra, dec, degree=False, epoch=2000.0):
    """
    Propagate the proper motions errors through the rotation from (ra,dec) to (l,b)

    Parameters
    ----------
    cov_pmradec : numpy.ndarray
        Uncertainty covariance matrix of the proper motion in ra (multiplied with cos(dec)) and dec [2,2] or [:,2,2]
    ra : float or numpy.ndarray
        Right ascension
    dec : float or numpy.ndarray
        Declination
    degree : bool, optional
        If True, ra and dec are given in degrees (default=False)
    epoch : float, optional
        Epoch of ra,dec (right now only 2000.0 and 1950.0 are supported when not using astropy's transformations internally; when internally using astropy's coordinate transformations, epoch can be None for ICRS, 'JXXXX' for FK5, and 'BXXXX' for FK4)

    Returns
    -------
    numpy.ndarray
        covar_pmllbb [2,2] or [:,2,2] [pmll here is pmll x cos(b)]

    Notes
    -----
    - 2010-04-12 - Written - Bovy (NYU)
    - 2020-09-21 - Adapted for array input - Mackereth (UofT)
    """
    scalar = not hasattr(ra, "__iter__")
    if scalar:
        cov_pmradec = cov_pmradec[numpy.newaxis, :, :]
    theta, dec_ngp, ra_ngp = get_epoch_angles(epoch)
    if degree:
        sindec_ngp = numpy.sin(dec_ngp)
        cosdec_ngp = numpy.cos(dec_ngp)
        sindec = numpy.sin(dec * _DEGTORAD)
        cosdec = numpy.cos(dec * _DEGTORAD)
        sinrarangp = numpy.sin(ra * _DEGTORAD - ra_ngp)
        cosrarangp = numpy.cos(ra * _DEGTORAD - ra_ngp)
    else:
        sindec_ngp = numpy.sin(dec_ngp)
        cosdec_ngp = numpy.cos(dec_ngp)
        sindec = numpy.sin(dec)
        cosdec = numpy.cos(dec)
        sinrarangp = numpy.sin(ra - ra_ngp)
        cosrarangp = numpy.cos(ra - ra_ngp)
    # These were replaced by Poleski (2013)'s equivalent form that is better at the poles
    # cosphi= (sindec_ngp-sindec*sinb)/cosdec/cosb
    # sinphi= sinrarangp*cosdec_ngp/cosb
    cosphi = sindec_ngp * cosdec - cosdec_ngp * sindec * cosrarangp
    sinphi = sinrarangp * cosdec_ngp
    norm = numpy.sqrt(cosphi**2.0 + sinphi**2.0)
    cosphi /= norm
    sinphi /= norm
    P = numpy.zeros([len(cov_pmradec), 2, 2])
    P[:, 0, 0] = cosphi
    P[:, 0, 1] = sinphi
    P[:, 1, 0] = -sinphi
    P[:, 1, 1] = cosphi
    out = numpy.einsum(
        "aij,ajk->aik", P, numpy.einsum("aij,jka->aik", cov_pmradec, P.T)
    )
    if scalar:
        return out[0]
    else:
        return out


def cov_dvrpmllbb_to_vxyz(
    d, e_d, e_vr, pmll, pmbb, cov_pmllbb, l, b, plx=False, degree=False
):
    """
    Propagate distance, radial velocity, and proper motion uncertainties to Galactic coordinates

    Parameters
    ----------
    d : float or numpy.ndarray
        Distance [kpc, as/mas for plx]
    e_d : float or numpy.ndarray
        Distance uncertainty [kpc, [as/mas] for plx]
    e_vr : float or numpy.ndarray
        Low velocity uncertainty [km/s]
    pmll : float or numpy.ndarray
        Proper motion in l (*cos(b)) [ [as/mas]/yr ]
    pmbb : float or numpy.ndarray
        Proper motion in b [ [as/mas]/yr ]
    cov_pmllbb : numpy.ndarray
        Uncertainty covariance for proper motion [pmll is pmll x cos(b)] [:,2,2]
    l : float or numpy.ndarray
        Galactic longitude
    b : float or numpy.ndarray
        Galactic latitude
    plx : bool, optional
        If True, d is a parallax, and e_d is a parallax uncertainty (default=False)
    degree : bool, optional
        If True, l and b are given in degree (default=False)

    Returns
    -------
    numpy.ndarray
        cov(vx,vy,vz) [3,3] or [:,3,3]

    Notes
    -----
    - 2010-04-12 - Written - Bovy (NYU)
    - 2020-09-21 - Adapted for array input - Mackereth (UofT)
    """
    if plx:
        d = 1.0 / d
        e_d *= d**2.0
    if degree:
        l *= _DEGTORAD
        b *= _DEGTORAD
    scalar = not hasattr(d, "__iter__")
    if scalar:
        cov_pmllbb = cov_pmllbb[numpy.newaxis, :, :]
    ndata = len(cov_pmllbb)
    M = numpy.zeros((ndata, 2, 3))
    M[:, 0, 0] = pmll
    M[:, 1, 0] = pmbb
    M[:, 0, 1] = d
    M[:, 1, 2] = d
    M = _K * M
    cov_dpmllbb = numpy.zeros((ndata, 3, 3))
    cov_dpmllbb[:, 0, 0] = e_d**2.0
    cov_dpmllbb[:, 1:3, 1:3] = cov_pmllbb
    cov_vlvb = numpy.einsum(
        "aij,ajk->aik", M, numpy.einsum("aij,jka->aik", cov_dpmllbb, M.T)
    )
    if scalar:
        cov_vlvb = cov_vlvb[0]
    cov_vrvlvb = numpy.zeros((ndata, 3, 3))
    cov_vrvlvb[:, 0, 0] = e_vr**2.0
    cov_vrvlvb[:, 1:3, 1:3] = cov_vlvb
    R = numpy.zeros((ndata, 3, 3))
    R[:, 0, 0] = numpy.cos(l) * numpy.cos(b)
    R[:, 0, 1] = numpy.sin(l) * numpy.cos(b)
    R[:, 0, 2] = numpy.sin(b)
    R[:, 1, 0] = -numpy.sin(l)
    R[:, 1, 1] = numpy.cos(l)
    R[:, 2, 0] = -numpy.cos(l) * numpy.sin(b)
    R[:, 2, 1] = -numpy.sin(l) * numpy.sin(b)
    R[:, 2, 2] = numpy.cos(b)
    out = numpy.einsum("ija,ajk->aik", R.T, numpy.einsum("aij,ajk->aik", cov_vrvlvb, R))
    if scalar:
        return out[0]
    else:
        return out


def cov_vxyz_to_galcencyl(cov_vxyz, phi, Xsun=1.0, Zsun=0.0):
    """
    Propagate uncertainties in vxyz to galactocentric cylindrical coordinates

    Parameters
    ----------
    cov_vxyz : numpy.ndarray
        Uncertainty covariance in U,V,W [3,3] or [:,3,3]
    phi : float or numpy.ndarray
        Angular position of star in Galactocentric cylindrical coords [rad]
    Xsun : float, optional
        Cylindrical distance to the GC (can be array of same length as R) [kpc]
    Zsun : float, optional
        Sun's height above the midplane (can be array of same length as R) [kpc]

    Returns
    -------
    numpy.ndarray
        cov(vR,vT,vz) [3,3] or [:,3,3]

    Notes
    -----
    - 2018-03-22 - Written - Mackereth (LJMU)
    - 2020-09-21- Moved to coords.py - Mackereth (UofT)
    """
    cov_galcenrect = cov_vxyz_to_galcenrect(cov_vxyz, Xsun=Xsun, Zsun=Zsun)
    cov_galcencyl = cov_galcenrect_to_galcencyl(cov_galcenrect, phi)
    return cov_galcencyl


def cov_vxyz_to_galcenrect(cov_vxyz, Xsun=1.0, Zsun=0.0):
    """
    Propagate uncertainties in vxyz to galactocentric rectangular coordinates

    Parameters
    ----------
    cov_vxyz : numpy.ndarray
        Uncertainty covariance in U,V,W [3,3] or [:,3,3]
    Xsun : float, optional
        Cylindrical distance to the GC (can be array of same length as R) [kpc]
    Zsun : float, optional
        Sun's height above the midplane (can be array of same length as R) [kpc]

    Returns
    -------
    numpy.ndarray
        cov(vx,vy,vz) [3,3] or [:,3,3]

    Notes
    -----
    - 2018-03-22 - Written - Mackereth (LJMU)
    - 2020-09-21- Moved to coords.py - Mackereth (UofT)
    """
    scalar = cov_vxyz.ndim < 3
    if scalar:
        cov_vxyz = cov_vxyz[numpy.newaxis, :, :]
    dgc = numpy.sqrt(Xsun**2.0 + Zsun**2.0)
    costheta, sintheta = Xsun / dgc, Zsun / dgc
    R = numpy.array(
        [[costheta, 0.0, -sintheta], [0.0, 1.0, 0.0], [sintheta, 0.0, costheta]]
    )
    R = numpy.ones([len(cov_vxyz), 3, 3]) * R
    out = numpy.einsum("ija,ajk->aik", R.T, numpy.einsum("aij,ajk->aik", cov_vxyz, R))
    if scalar:
        return out[0]
    else:
        return out


def cov_galcenrect_to_galcencyl(cov_galcenrect, phi):
    """
    Propagate uncertainties in galactocentric rectangular to galactocentric cylindrical coordinates

    Parameters
    ----------
    cov_galcenrect : numpy.ndarray
        Uncertainty covariance in vx,vy,vz [3,3] or [:,3,3]
    phi : float or numpy.ndarray
        Angular position of star in Galactocentric cylindrical coords [rad]

    Returns
    -------
    numpy.ndarray
        cov(vR,vT,vz) [3,3] or [:,3,3]

    Notes
    -----
    - 2018-03-22 - Written - Mackereth (LJMU)
    - 2020-09-21- Moved to coords.py - Mackereth (UofT)
    """
    scalar = cov_galcenrect.ndim < 3
    if scalar:
        cov_galcenrect = cov_galcenrect[numpy.newaxis, :, :]
    cosphi = numpy.cos(phi)
    sinphi = numpy.sin(phi)
    R = numpy.zeros([len(cov_galcenrect), 3, 3])
    R[:, 0, 0] = cosphi
    R[:, 0, 1] = sinphi
    R[:, 1, 0] = -sinphi
    R[:, 1, 1] = cosphi
    R[:, 2, 2] = 1.0
    out = numpy.einsum(
        "aij,ajk->aik", R, numpy.einsum("aij,jka->aik", cov_galcenrect, R.T)
    )
    if scalar:
        return out[0]
    else:
        return out


@scalarDecorator
def XYZ_to_galcenrect(X, Y, Z, Xsun=1.0, Zsun=0.0, _extra_rot=True):
    """
    Transform XYZ coordinates (wrt Sun) to rectangular Galactocentric coordinates.

    Parameters
    ----------
    X : float or numpy.ndarray
        X coordinate.
    Y : float or numpy.ndarray
        Y coordinate.
    Z : float or numpy.ndarray
        Z coordinate.
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the Galactic center (default is 1.0).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (default is 0.0).
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition (default is True).

    Returns
    -------
    numpy.ndarray
        Containing:
            * Xg : X coordinate in Galactocentric frame.
            * Yg : Y coordinate in Galactocentric frame.
            * Zg : Z coordinate in Galactocentric frame.

    Notes
    -----
    - 2010-09-24 - Written - Bovy (NYU)
    - 2016-05-12 - Edited to properly take into account the Sun's vertical position; dropped Ysun keyword - Bovy (UofT)
    - 2018-04-18 - Tweaked to be consistent with astropy's Galactocentric frame - Bovy (UofT)
    - 2023-07-23 - Allowed Xsun/Zsun to be arrays - Bovy (UofT)
    """
    if _extra_rot:
        X, Y, Z = numpy.dot(galcen_extra_rot, numpy.array([X, Y, Z]))
    dgc = numpy.sqrt(Xsun**2.0 + Zsun**2.0)
    costheta, sintheta = Xsun / dgc, Zsun / dgc
    zero = numpy.zeros_like(costheta)
    one = numpy.ones_like(costheta)
    return numpy.einsum(
        (
            "ijk,jk->ik"
            if isinstance(costheta, numpy.ndarray) and costheta.ndim > 0
            else "ij,jk->ik"
        ),
        numpy.array(
            [[costheta, zero, -sintheta], [zero, one, zero], [sintheta, zero, costheta]]
        ),
        numpy.array([-X + dgc, Y, numpy.sign(Xsun) * Z]),
    ).T


@scalarDecorator
def galcenrect_to_XYZ(X, Y, Z, Xsun=1.0, Zsun=0.0, _extra_rot=True):
    """
    Transform rectangular Galactocentric to XYZ coordinates (wrt Sun) coordinates.

    Parameters
    ----------
    X : float or numpy.ndarray
        Galactocentric rectangular coordinate.
    Y : float or numpy.ndarray
        Galactocentric rectangular coordinate.
    Z : float or numpy.ndarray
        Galactocentric rectangular coordinate.
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as X).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as X).
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition.

    Returns
    -------
    numpy.ndarray
        Containing (X,Y,Z)

    Notes
    -----
    - 2011-02-23 - Written - Bovy (NYU)
    - 2016-05-12 - Edited to properly take into account the Sun's vertical position; dropped Ysun keyword - Bovy (UofT)
    - 2017-10-24 - Allowed Xsun/Zsun to be arrays - Bovy (UofT)
    - 2018-04-18 - Tweaked to be consistent with astropy's Galactocentric frame - Bovy (UofT)

    """
    dgc = numpy.sqrt(Xsun**2.0 + Zsun**2.0)
    costheta, sintheta = Xsun / dgc, Zsun / dgc
    zero = numpy.zeros_like(costheta)
    one = numpy.ones_like(costheta)
    out = (
        numpy.einsum(
            (
                "ijk,jk->ik"
                if isinstance(costheta, numpy.ndarray) and costheta.ndim > 0
                else "ij,jk->ik"
            ),
            numpy.array(
                [
                    [-costheta, zero, -sintheta],
                    [zero, one, zero],
                    [-numpy.sign(Xsun) * sintheta, zero, numpy.sign(Xsun) * costheta],
                ]
            ),
            numpy.array([X, Y, Z]),
        ).T
        + numpy.array([dgc, zero, zero]).T
    )
    if _extra_rot:
        return numpy.dot(galcen_extra_rot.T, out.T).T
    else:
        return out


def rect_to_cyl(X, Y, Z):
    """
    Convert from rectangular to cylindrical coordinates

    Parameters
    ----------
    X : float or numpy.ndarray
        X coordinate.
    Y : float or numpy.ndarray
        Y coordinate.
    Z : float or numpy.ndarray
        Z coordinate.

    Returns
    -------
    numpy.ndarray
        Containing (R,phi,z)

    Notes
    -----
    - 2010-09-24 - Written - Bovy (NYU)
    - 2019-06-21 - Changed such that phi in [-pi,pi] - Bovy (UofT)
    """
    return (numpy.sqrt(X**2.0 + Y**2.0), numpy.arctan2(Y, X), Z)


def cyl_to_rect(R, phi, Z):
    """
    Convert from cylindrical to rectangular coordinates

    Parameters
    ----------
    R : float or numpy.ndarray
        Cylindrical R coordinate.
    phi : float or numpy.ndarray
        Cylindrical phi coordinate.
    Z : float or numpy.ndarray
        Cylindrical Z coordinate.

    Returns
    -------
    numpy.ndarray
        Containing (X,Y,Z)

    Notes
    -----
    - 2011-02-23 - Written - Bovy (NYU)
    """
    return (R * numpy.cos(phi), R * numpy.sin(phi), Z)


def cyl_to_spher(R, Z, phi):
    """
    Convert from cylindrical to spherical coordinates

    Parameters
    ----------
    R : float or numpy.ndarray
        Cylindrical R coordinate.
    Z : float or numpy.ndarray
        Cylindrical Z coordinate.
    phi : float or numpy.ndarray
        Cylindrical phi coordinate.

    Returns
    -------
    numpy.ndarray
        Containing (r,theta,phi)

    Notes
    -----
    - 2016-05-16 - Written - Aladdin
    """
    theta = numpy.arctan2(R, Z)
    r = (R**2 + Z**2) ** 0.5
    return (r, theta, phi)


def spher_to_cyl(r, theta, phi):
    """
    Convert from spherical to cylindrical coordinates

    Parameters
    ----------
    r : float or numpy.ndarray
        Spherical r coordinate.
    theta : float or numpy.ndarray
        Spherical theta coordinate.
    phi : float or numpy.ndarray
        Spherical phi coordinate.

    Returns
    -------
    numpy.ndarray
        Containing (R,z,phi)

    Notes
    -----
    - 2016-05-20 - Written - Aladdin
    """
    R = r * numpy.sin(theta)
    z = r * numpy.cos(theta)
    return (R, z, phi)


@scalarDecorator
def XYZ_to_galcencyl(X, Y, Z, Xsun=1.0, Zsun=0.0, _extra_rot=True):
    """
    Transform XYZ coordinates (wrt Sun) to cylindrical Galactocentric coordinates

    Parameters
    ----------
    X : float or numpy.ndarray
        X coordinate.
    Y : float or numpy.ndarray
        Y coordinate.
    Z : float or numpy.ndarray
        Z coordinate.
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as R).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as R).
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition.

    Returns
    -------
    numpy.ndarray
        Containing (R,phi,z)

    Notes
    -----
    - 2010-09-24 - Written - Bovy (NYU)
    - 2023-07-19 - Allowed Xsun/Zsun to be arrays - Bovy (UofT)
    """
    XYZ = numpy.atleast_2d(
        XYZ_to_galcenrect(X, Y, Z, Xsun=Xsun, Zsun=Zsun, _extra_rot=_extra_rot)
    )
    return numpy.array(rect_to_cyl(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])).T


@scalarDecorator
def galcencyl_to_XYZ(R, phi, Z, Xsun=1.0, Zsun=0.0, _extra_rot=True):
    """
    Transform cylindrical Galactocentric coordinates to XYZ coordinates (wrt Sun)

    Parameters
    ----------
    R : float or numpy.ndarray
        Cylindrical R coordinate.
    phi : float or numpy.ndarray
        Cylindrical phi coordinate.
    Z : float or numpy.ndarray
        Cylindrical Z coordinate.
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as R).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as R).
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition.

    Returns
    -------
    numpy.ndarray
        Containing (X,Y,Z)

    Notes
    -----
    - 2011-02-23 - Written - Bovy (NYU)
    - 2017-10-24 - Allowed Xsun/Zsun to be arrays - Bovy (UofT)
    """
    Xr, Yr, Zr = cyl_to_rect(R, phi, Z)
    return galcenrect_to_XYZ(Xr, Yr, Zr, Xsun=Xsun, Zsun=Zsun, _extra_rot=_extra_rot)


@scalarDecorator
def vxvyvz_to_galcenrect(
    vx, vy, vz, vsun=[0.0, 1.0, 0.0], Xsun=1.0, Zsun=0.0, _extra_rot=True
):
    """
    Transform velocities in XYZ coordinates (wrt Sun) to rectangular Galactocentric coordinates for velocities

    Parameters
    ----------
    vx : float or numpy.ndarray
        U velocity.
    vy : float or numpy.ndarray
        V velocity.
    vz : float or numpy.ndarray
        W velocity.
    vsun : float or numpy.ndarray, optional
        Velocity of the Sun in the GC frame (can also be array of same length as vx; shape [3,N]).
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as vXg).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as vXg).
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition.

    Returns
    -------
    numpy.ndarray
        Containing (vx,vy,vz)

    Notes
    -----
    - 2010-09-24 - Written - Bovy (NYU)
    - 2016-05-12 - Edited to properly take into account the Sun's vertical position; dropped Ysun keyword - Bovy (UofT)
    - 2018-04-18 - Tweaked to be consistent with astropy's Galactocentric frame - Bovy (UofT)
    - 2023-07-23 - Allowed Xsun/Zsun/vsun to be arrays- Bovy (UofT)
    """
    if _extra_rot:
        vx, vy, vz = numpy.dot(galcen_extra_rot, numpy.array([vx, vy, vz]))
    dgc = numpy.sqrt(Xsun**2.0 + Zsun**2.0)
    costheta, sintheta = Xsun / dgc, Zsun / dgc
    zero = numpy.zeros_like(costheta)
    one = numpy.ones_like(costheta)
    return (
        numpy.einsum(
            (
                "ijk,jk->ik"
                if isinstance(costheta, numpy.ndarray) and costheta.ndim > 0
                else "ij,jk->ik"
            ),
            numpy.array(
                [
                    [costheta, zero, -sintheta],
                    [zero, one, zero],
                    [sintheta, zero, costheta],
                ]
            ),
            numpy.array([-vx, vy, numpy.sign(Xsun) * vz]),
        ).T
        + numpy.array(vsun).T
    )


@scalarDecorator
def vxvyvz_to_galcencyl(
    vx,
    vy,
    vz,
    X,
    Y,
    Z,
    vsun=[0.0, 1.0, 0.0],
    Xsun=1.0,
    Zsun=0.0,
    galcen=False,
    _extra_rot=True,
):
    """
    Transform velocities in XYZ coordinates (wrt Sun) to cylindrical Galactocentric coordinates for velocities

    Parameters
    ----------
    vx : float or numpy.ndarray
        U velocity.
    vy : float or numpy.ndarray
        V velocity.
    vz : float or numpy.ndarray
        W velocity.
    X : float or numpy.ndarray
        X in Galactocentric rectangular coordinates.
    Y : float or numpy.ndarray
        Y in Galactocentric rectangular coordinates.
    Z : float or numpy.ndarray
        Z in Galactocentric rectangular coordinates.
    vsun : float or numpy.ndarray, optional
        Velocity of the Sun in the GC frame (can also be array of same length as vx; shape [3,N]).
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as vXg).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as vXg).
    galcen : bool, optional
        If True, then X,Y,Z are in cylindrical Galactocentric coordinates rather than rectangular coordinates.
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition.

    Returns
    -------
    numpy.ndarray
        Containing (vR,vT,vz)

    Notes
    -----
    - 2010-09-24 - Written - Bovy (NYU)
    - 2023-07-19 - Allowed Xsun/Zsun/vsun to be arrays- Bovy (UofT)
    """
    vxyz = vxvyvz_to_galcenrect(
        vx, vy, vz, vsun=vsun, Xsun=Xsun, Zsun=Zsun, _extra_rot=_extra_rot
    )
    return numpy.array(
        rect_to_cyl_vec(vxyz[:, 0], vxyz[:, 1], vxyz[:, 2], X, Y, Z, cyl=galcen)
    ).T


@scalarDecorator
def galcenrect_to_vxvyvz(
    vXg, vYg, vZg, vsun=[0.0, 1.0, 0.0], Xsun=1.0, Zsun=0.0, _extra_rot=True
):
    """
    Transform rectangular Galactocentric coordinates to XYZ coordinates (wrt Sun) for velocities

    Parameters
    ----------
    vXg : float or numpy.ndarray
        Galactocentric x-velocity.
    vYg : float or numpy.ndarray
        Galactocentric y-velocity.
    vZg : float or numpy.ndarray
        Galactocentric z-velocity.
    vsun : float or numpy.ndarray, optional
        Velocity of the Sun in the GC frame (can also be array of same length as vXg; shape [3,N]).
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as vXg).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as vXg).
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition.

    Returns
    -------
    numpy.ndarray
        Containing (vx,vy,vz)

    Notes
    -----
    - 2011-02-24 - Written - Bovy (NYU)
    - 2016-05-12 - Edited to properly take into account the Sun's vertical position; dropped Ysun keyword - Bovy (UofT)
    - 2017-10-24 - Allowed Xsun/Zsun/vsun to be arrays - Bovy (UofT)
    - 2018-04-18 - Tweaked to be consistent with astropy's Galactocentric frame - Bovy (UofT)
    """
    dgc = numpy.sqrt(Xsun**2.0 + Zsun**2.0)
    costheta, sintheta = Xsun / dgc, Zsun / dgc
    zero = numpy.zeros_like(costheta)
    one = numpy.ones_like(costheta)
    out = numpy.einsum(
        (
            "ijk,jk->ik"
            if isinstance(costheta, numpy.ndarray) and costheta.ndim > 0
            else "ij,jk->ik"
        ),
        numpy.array(
            [
                [-costheta, zero, -sintheta],
                [zero, one, zero],
                [-numpy.sign(Xsun) * sintheta, zero, numpy.sign(Xsun) * costheta],
            ]
        ),
        numpy.array([vXg - vsun[0], vYg - vsun[1], vZg - vsun[2]]),
    ).T
    if _extra_rot:
        return numpy.dot(galcen_extra_rot.T, out.T).T
    else:
        return out


@scalarDecorator
def galcencyl_to_vxvyvz(
    vR, vT, vZ, phi, vsun=[0.0, 1.0, 0.0], Xsun=1.0, Zsun=0.0, _extra_rot=True
):
    """
    Transform cylindrical Galactocentric coordinates to XYZ coordinates (wrt Sun) for velocities

    Parameters
    ----------
    vR : float or numpy.ndarray
        Galactocentric radial velocity.
    vT : float or numpy.ndarray
        Galactocentric rotational velocity.
    vZ : float or numpy.ndarray
        Galactocentric vertical velocity.
    phi : float or numpy.ndarray
        Galactocentric azimuth.
    vsun : float or numpy.ndarray, optional
        Velocity of the Sun in the GC frame (can also be array of same length as vR; shape [3,N]).
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as vR).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as vR).
    _extra_rot : bool, optional
        If True, perform an extra tiny rotation to align the Galactocentric coordinate frame with astropy's definition.

    Parameters
    ----------
    numpy.ndarray
        Containing (vx,vy,vz)

    Notes
    -----
    - 2011-02-24 - Written - Bovy (NYU)
    - 2017-10-24 - Allowed vsun/Xsun/Zsun to be arrays - Bovy (NYU)

    """
    vXg, vYg, vZg = cyl_to_rect_vec(vR, vT, vZ, phi)
    return galcenrect_to_vxvyvz(
        vXg, vYg, vZg, vsun=vsun, Xsun=Xsun, Zsun=Zsun, _extra_rot=_extra_rot
    )


def cyl_to_spher_vec(vR, vT, vz, R, z):
    """
    Transform vectors from cylindrical to spherical coordinates. vtheta is positive from pole towards equator.

    Parameters
    ----------
    vR : float or numpy.ndarray
        Galactocentric cylindrical radial velocity
    vT : float or numpy.ndarray
        Galactocentric cylindrical rotational velocity
    vz : float or numpy.ndarray
        Galactocentric cylindrical vertical velocity
    R : float or numpy.ndarray
        Galactocentric cylindrical radius
    z : float or numpy.ndarray
        Galactocentric cylindrical height

    Returns
    -------
    tuple
        Containing:
            * vr : Galactocentric spherical radial velocity
            * vT : Galactocentric spherical rotational velocity
            * vtheta : Galactocentric spherical polar velocity

    Notes
    -----
    - 2020-07-01 - Written - James Lane (UofT)
    """
    r = numpy.sqrt(R**2.0 + z**2.0)
    vr = (R * vR + z * vz) / r
    vtheta = (z * vR - R * vz) / r
    return (vr, vT, vtheta)


def spher_to_cyl_vec(vr, vT, vtheta, theta):
    """
    Transform vectors from spherical polar to cylindrical coordinates.

    Parameters
    ----------
    vr : float or numpy.ndarray
        Galactocentric spherical radial velocity
    vT : float or numpy.ndarray
        Galactocentric spherical azimuthal velocity
    vtheta : float or numpy.ndarray
        Galactocentric spherical polar velocity
    theta : float or numpy.ndarray
        Galactocentric spherical polar angle

    Returns
    -------
    tuple
        A tuple containing the Galactocentric cylindrical radial velocity, azimuthal velocity, and vertical velocity in that order.

    Notes
    -----
    - 2020-07-01 - Written - James Lane (UofT)
    """
    vR = vr * numpy.sin(theta) + vtheta * numpy.cos(theta)
    vz = vr * numpy.cos(theta) - vtheta * numpy.sin(theta)
    return (vR, vT, vz)


def rect_to_cyl_vec(vx, vy, vz, X, Y, Z, cyl=False):
    """
    Transform vectors from rectangular to cylindrical coordinates vectors.

    Parameters
    ----------
    vx : float or numpy.ndarray
        Velocity in the x-direction.
    vy : float or numpy.ndarray
        Velocity in the y-direction.
    vz : float or numpy.ndarray
        Velocity in the z-direction.
    X : float or numpy.ndarray
        X-coordinate.
    Y : float or numpy.ndarray
        Y-coordinate.
    Z : float or numpy.ndarray
        Z-coordinate.
    cyl : bool, optional
        If True, X, Y, Z are already cylindrical (i.e., [X,Y,Z] == [R,phi,Z]), by default False.

    Returns
    -------
    tuple
        Tuple containing vR, vT, vz.

    Notes
    -----
    - 2010-09-24 - Written - Bovy (NYU)

    """
    if not cyl:
        R, phi, Z = rect_to_cyl(X, Y, Z)
    else:
        phi = Y
    vr = +vx * numpy.cos(phi) + vy * numpy.sin(phi)
    vt = -vx * numpy.sin(phi) + vy * numpy.cos(phi)
    return (vr, vt, vz)


def cyl_to_rect_vec(vr, vt, vz, phi):
    """
    Transform vectors from cylindrical to rectangular coordinate vectors.

    Parameters
    ----------
    vr : float or numpy.ndarray
        Radial velocity.
    vt : float or numpy.ndarray
        Rotational velocity.
    vz : float or numpy.ndarray
        Vertical velocity.
    phi : float or numpy.ndarray
        Azimuth.

    Returns
    -------
    tuple
        Rectangular coordinate vectors (vx, vy, vz).

    Notes
    -----
    - 2011-02-24 - Written - Bovy (NYU)
    """
    vx = vr * numpy.cos(phi) - vt * numpy.sin(phi)
    vy = vr * numpy.sin(phi) + vt * numpy.cos(phi)
    return (vx, vy, vz)


def cyl_to_rect_jac(*args):
    """
    Calculate the Jacobian of the cylindrical to rectangular conversion

    Parameters
    ----------
    R : float or numpy.ndarray
        Cylindrical R coordinate.
    phi : float or numpy.ndarray
        Cylindrical phi coordinate.
    Z : float or numpy.ndarray
        Cylindrical Z coordinate.
    vR : float or numpy.ndarray, optional
        Cylindrical vR coordinate.
    vT : float or numpy.ndarray, optional
        Cylindrical vT coordinate.
    vz : float or numpy.ndarray, optional
        Cylindrical vz coordinate.

    Returns
    -------
    numpy.ndarray
        Jacobian of the cylindrical to rectangular conversion either just spatial or spatial+velocity.

    Notes
    -----
    - 2013-12-09 - Written - Bovy (IAS)
    """
    out = numpy.zeros((6, 6))
    if len(args) == 3:
        R, phi, Z = args
        vR, vT, vZ = 0.0, 0.0, 0.0
        outIndx = numpy.array([True, False, False, True, False, True], dtype="bool")
    elif len(args) == 6:
        R, vR, vT, Z, vZ, phi = args
        outIndx = numpy.ones(6, dtype="bool")
    cp = numpy.cos(phi)
    sp = numpy.sin(phi)
    out[0, 0] = cp
    out[0, 5] = -R * sp
    out[1, 0] = sp
    out[1, 5] = R * cp
    out[2, 3] = 1.0
    out[3, 1] = cp
    out[3, 2] = -sp
    out[3, 5] = -vT * cp - vR * sp
    out[4, 1] = sp
    out[4, 2] = cp
    out[4, 5] = -vT * sp + vR * cp
    out[5, 4] = 1.0
    if len(args) == 3:
        out = out[:3, outIndx]
        out[:, [1, 2]] = out[:, [2, 1]]
    return out


def galcenrect_to_XYZ_jac(*args, **kwargs):
    """
    Calculate the Jacobian of the Galactocentric rectangular to Galactic coordinates

    Parameters
    ----------
    X : float or numpy.ndarray
        Galactocentric rectangular coordinate.
    Y : float or numpy.ndarray
        Galactocentric rectangular coordinate.
    Z : float or numpy.ndarray
        Galactocentric rectangular coordinate.
    vX : float or numpy.ndarray, optional
        Galactocentric rectangular velocity.
    vY : float or numpy.ndarray, optional
        Galactocentric rectangular velocity.
    vZ : float or numpy.ndarray, optional
        Galactocentric rectangular velocity.
    Xsun : float or numpy.ndarray, optional
        Cylindrical distance to the GC (can be array of same length as X).
    Zsun : float or numpy.ndarray, optional
        Sun's height above the midplane (can be array of same length as X).

    Returns
    -------
    numpy.ndarray
        Jacobian of the Galactocentric rectangular to Galactic coordinates either just spatial or spatial+velocity.

    Notes
    -----
    - 2013-12-09 - Written - Bovy (IAS)

    """
    Xsun = kwargs.get("Xsun", 1.0)
    dgc = numpy.sqrt(Xsun**2.0 + kwargs.get("Zsun", 0.0) ** 2.0)
    costheta, sintheta = Xsun / dgc, kwargs.get("Zsun", 0.0) / dgc
    out = numpy.zeros((6, 6))
    out[0, 0] = -costheta
    out[0, 2] = -sintheta
    out[1, 1] = 1.0
    out[2, 0] = -numpy.sign(Xsun) * sintheta
    out[2, 2] = numpy.sign(Xsun) * costheta
    out[3, 3] = -costheta
    out[3, 5] = -sintheta
    out[4, 4] = 1.0
    out[5, 3] = -numpy.sign(Xsun) * sintheta
    out[5, 5] = numpy.sign(Xsun) * costheta
    return out[: len(args), : len(args)]


def lbd_to_XYZ_jac(*args, **kwargs):
    """
    Calculate the Jacobian of the Galactic spherical coordinates to Galactic rectangular coordinates transformation

    Parameters
    ----------
    l : float or numpy.ndarray
        Galactic longitude.
    b : float or numpy.ndarray
        Galactic latitude.
    D : float or numpy.ndarray
        Distance.
    vlos : float or numpy.ndarray, optional
        Line-of-sight velocity.
    pmll : float or numpy.ndarray, optional
        Proper motion in Galactic longitude.
    pmbb : float or numpy.ndarray, optional
        Proper motion in Galactic latitude.
    degree : bool, optional
        If True, l and b are in degrees rather than rad.

    Returns
    -------
    numpy.ndarray
        Jacobian of the Galactic spherical coordinates to Galactic rectangular coordinates transformation either just spatial or spatial+velocity.

    Notes
    -----
    - 2013-12-09 - Written - Bovy (IAS)
    """
    out = numpy.zeros((6, 6))
    if len(args) == 3:
        l, b, D = args
        vlos, pmll, pmbb = 0.0, 0.0, 0.0
    elif len(args) == 6:
        l, b, D, vlos, pmll, pmbb = args
    if kwargs.get("degree", False):
        l *= _DEGTORAD
        b *= _DEGTORAD
    cl = numpy.cos(l)
    sl = numpy.sin(l)
    cb = numpy.cos(b)
    sb = numpy.sin(b)
    out[0, 0] = -D * cb * sl
    out[0, 1] = -D * sb * cl
    out[0, 2] = cb * cl
    out[1, 0] = D * cb * cl
    out[1, 1] = -D * sb * sl
    out[1, 2] = cb * sl
    out[2, 1] = D * cb
    out[2, 2] = sb
    if len(args) == 3:
        if kwargs.get("degree", False):
            out[:, 0] *= _DEGTORAD
            out[:, 1] *= _DEGTORAD
        return out[:3, :3]
    out[3, 0] = -sl * cb * vlos - cl * _K * D * pmll + sb * sl * _K * D * pmbb
    out[3, 1] = -cl * sb * vlos - cb * cl * _K * D * pmbb
    out[3, 2] = -sl * _K * pmll - sb * cl * _K * pmbb
    out[3, 3] = cl * cb
    out[3, 4] = -sl * _K * D
    out[3, 5] = -cl * sb * _K * D
    out[4, 0] = cl * cb * vlos - sl * _K * D * pmll - cl * sb * _K * D * pmbb
    out[4, 1] = -sl * sb * vlos - sl * cb * _K * D * pmbb
    out[4, 2] = cl * _K * pmll - sl * sb * _K * pmbb
    out[4, 3] = sl * cb
    out[4, 4] = cl * _K * D
    out[4, 5] = -sl * sb * _K * D
    out[5, 1] = cb * vlos - sb * _K * D * pmbb
    out[5, 2] = cb * _K * pmbb
    out[5, 3] = sb
    out[5, 5] = cb * _K * D
    if kwargs.get("degree", False):
        out[:, 0] *= _DEGTORAD
        out[:, 1] *= _DEGTORAD
    return out


def dl_to_rphi_2d(d, l, degree=False, ro=1.0, phio=0.0):
    """
    Convert Galactic longitude and distance to Galactocentric radius and azimuth

    Parameters
    ----------
    d : float or numpy.ndarray
        Distance.
    l : float or numpy.ndarray
        Galactic longitude.
    degree : bool, optional
        If True, l is in degrees rather than rad, by default False.
    ro : float, optional
        Galactocentric radius of the observer, by default 1.0.
    phio : float, optional
        Galactocentric azimuth of the observer, by default 0.0.

    Returns
    -------
    tuple
        (R, phi); phi in degree if degree

    Notes
    -----
    - 2012-01-04 - Written - Bovy (IAS)
    """
    scalarOut, listOut = False, False
    if isinstance(d, (int, float)):
        d = numpy.array([d])
        scalarOut = True
    elif isinstance(d, list):
        d = numpy.array(d)
        listOut = True
    if isinstance(l, (int, float)):
        l = numpy.array([l])
    elif isinstance(l, list):
        l = numpy.array(l)
    if degree:
        l *= _DEGTORAD
    R = numpy.sqrt(ro**2.0 + d**2.0 - 2.0 * d * ro * numpy.cos(l))
    phi = numpy.arcsin(d / R * numpy.sin(l))
    indx = (ro / numpy.cos(l) < d) * (numpy.cos(l) > 0.0)
    phi[indx] = numpy.pi - numpy.arcsin(d[indx] / R[indx] * numpy.sin(l[indx]))
    if degree:
        phi /= _DEGTORAD
    phi += phio
    if scalarOut:
        return (R[0], phi[0])
    elif listOut:
        return (list(R), list(phi))
    else:
        return (R, phi)


def rphi_to_dl_2d(R, phi, degree=False, ro=1.0, phio=0.0):
    """
    Convert Galactocentric radius and azimuth to distance and Galactic longitude

    Parameters
    ----------
    R : float or numpy.ndarray
        Galactocentric radius.
    phi : float or numpy.ndarray
        Galactocentric azimuth.
    degree : bool, optional
        If True, phi is in degrees rather than rad, by default False.
    ro : float, optional
        Galactocentric radius of the observer, by default 1.0.
    phio : float, optional
        Galactocentric azimuth of the observer, by default 0.0.

    Returns
    -------
    tuple
        (d, l); phi in degree if degree

    Notes
    -----
    - 2012-01-04 - Written - Bovy (IAS)
    """
    scalarOut, listOut = False, False
    if isinstance(R, (int, float)):
        R = numpy.array([R])
        scalarOut = True
    elif isinstance(R, list):
        R = numpy.array(R)
        listOut = True
    if isinstance(phi, (int, float)):
        phi = numpy.array([phi])
    elif isinstance(phi, list):
        phi = numpy.array(phi)
    phi -= phio
    if degree:
        phi *= _DEGTORAD
    d = numpy.sqrt(R**2.0 + ro**2.0 - 2.0 * R * ro * numpy.cos(phi))
    l = numpy.arcsin(R / d * numpy.sin(phi))
    indx = (ro / numpy.cos(phi) < R) * (numpy.cos(phi) > 0.0)
    l[indx] = numpy.pi - numpy.arcsin(R[indx] / d[indx] * numpy.sin(phi[indx]))
    if degree:
        l /= _DEGTORAD
    if scalarOut:
        return (d[0], l[0])
    elif listOut:
        return (list(d), list(l))
    else:
        return (d, l)


def Rz_to_coshucosv(R, z, delta=1.0, oblate=False):
    """
    Calculate prolate confocal cosh(u) and cos(v) coordinates from R,z, and delta.

    Parameters
    ----------
    R : float or numpy.ndarray
        Radius.
    z : float or numpy.ndarray
        Height.
    delta : float or numpy.ndarray, optional
        Focus. Default is 1.0.
    oblate : bool, optional
        If True, compute oblate confocal coordinates instead of prolate. Default is False.

    Returns
    -------
    tuple
        Tuple containing cosh(u) and cos(v).

    Notes
    -----
    - 2012-11-27 - Written - Bovy (IAS)
    - 2017-10-11 - Added oblate coordinates - Bovy (UofT)
    """
    if oblate:
        d12 = (R + delta) ** 2.0 + z**2.0
        d22 = (R - delta) ** 2.0 + z**2.0
    else:
        d12 = (z + delta) ** 2.0 + R**2.0
        d22 = (z - delta) ** 2.0 + R**2.0
    coshu = 0.5 / delta * (numpy.sqrt(d12) + numpy.sqrt(d22))
    cosv = 0.5 / delta * (numpy.sqrt(d12) - numpy.sqrt(d22))
    if oblate:  # cosv is currently really sinv
        cosv = numpy.sqrt(1.0 - cosv**2.0)
    return (coshu, cosv)


def Rz_to_uv(R, z, delta=1.0, oblate=False):
    """
    Calculate prolate or oblate confocal u and v coordinates from R, z, and delta.

    Parameters
    ----------
    R : float or numpy.ndarray
        Radius.
    z : float or numpy.ndarray
        Height.
    delta : float or numpy.ndarray, optional
        Focus. Default is 1.0.
    oblate : bool, optional
        If True, compute oblate confocal coordinates instead of prolate. Default is False.

    Returns
    -------
    tuple
        (u, v)

    Notes
    -----
    - 2012-11-27 - Written - Bovy (IAS)
    - 2017-10-11 - Added oblate coordinates - Bovy (UofT)

    """
    coshu, cosv = Rz_to_coshucosv(R, z, delta, oblate=oblate)
    u = numpy.arccosh(coshu)
    v = numpy.arccos(cosv)
    return (u, v)


def uv_to_Rz(u, v, delta=1.0, oblate=False):
    """
    Calculate R and z from prolate confocal u and v coordinates.

    Parameters
    ----------
    u : float or numpy.ndarray
        Confocal u.
    v : float or numpy.ndarray
        Confocal v.
    delta : float or numpy.ndarray, optional
        Focus. Default is 1.0.
    oblate : bool, optional
        If True, compute oblate confocal coordinates instead of prolate. Default is False.

    Returns
    -------
    tuple
        (R, z)

    Notes
    -----
    - 2012-11-27 - Written - Bovy (IAS)
    - 2017-10-11 - Added oblate coordinates - Bovy (UofT)

    """
    if oblate:
        R = delta * numpy.cosh(u) * numpy.sin(v)
        z = delta * numpy.sinh(u) * numpy.cos(v)
    else:
        R = delta * numpy.sinh(u) * numpy.sin(v)
        z = delta * numpy.cosh(u) * numpy.cos(v)
    return (R, z)


def vRvz_to_pupv(vR, vz, R, z, delta=1.0, oblate=False, uv=False):
    """
    Calculate momenta in prolate or oblate confocal u and v coordinates from cylindrical velocities vR,vz for a given focal length delta.

    Parameters
    ----------
    vR : float or numpy.ndarray
        Radial velocity in cylindrical coordinates.
    vz : float or numpy.ndarray
        Vertical velocity in cylindrical coordinates.
    R : float or numpy.ndarray
        Radius.
    z : float or numpy.ndarray
        Height.
    delta : float or numpy.ndarray, optional
        Focus. Default is 1.0.
    oblate : bool, optional
        If True, compute oblate confocal coordinates instead of prolate. Default is False.
    uv : bool, optional
        If True, the given R,z are actually u,v. Default is False.

    Returns
    -------
    tuple
        (pu, pv)

    Notes
    -----
    - 2017-11-28 - Written - Bovy (UofT)
    """
    if not uv:
        u, v = Rz_to_uv(R, z, delta, oblate=oblate)
    else:
        u, v = R, z
    if oblate:
        pu = delta * (
            vR * numpy.sinh(u) * numpy.sin(v) + vz * numpy.cosh(u) * numpy.cos(v)
        )
        pv = delta * (
            vR * numpy.cosh(u) * numpy.cos(v) - vz * numpy.sinh(u) * numpy.sin(v)
        )
    else:
        pu = delta * (
            vR * numpy.cosh(u) * numpy.sin(v) + vz * numpy.sinh(u) * numpy.cos(v)
        )
        pv = delta * (
            vR * numpy.sinh(u) * numpy.cos(v) - vz * numpy.cosh(u) * numpy.sin(v)
        )
    return (pu, pv)


def pupv_to_vRvz(pu, pv, u, v, delta=1.0, oblate=False):
    """
    Calculate cylindrical vR and vz from momenta in prolate or oblate confocal u and v coordinates for a given focal length delta.

    Parameters
    ----------
    pu : float or numpy.ndarray
        Momentum in u.
    pv : float or numpy.ndarray
        Momentum in v.
    u : float or numpy.ndarray
        Confocal u.
    v : float or numpy.ndarray
        Confocal v.
    delta : float or numpy.ndarray, optional
        Focus. Default is 1.0.
    oblate : bool, optional
        If True, compute oblate confocal coordinates instead of prolate. Default is False.

    Returns
    -------
    tuple
        (vR, vz)

    Notes
    -----
    - 2017-12-04 - Written - Bovy (UofT)
    """
    if oblate:
        denom = delta * (numpy.sinh(u) ** 2.0 + numpy.cos(v) ** 2.0)
        vR = (
            pu * numpy.sinh(u) * numpy.sin(v) + pv * numpy.cosh(u) * numpy.cos(v)
        ) / denom
        vz = (
            pu * numpy.cosh(u) * numpy.cos(v) - pv * numpy.sinh(u) * numpy.sin(v)
        ) / denom
    else:
        denom = delta * (numpy.sinh(u) ** 2.0 + numpy.sin(v) ** 2.0)
        vR = (
            pu * numpy.cosh(u) * numpy.sin(v) + pv * numpy.sinh(u) * numpy.cos(v)
        ) / denom
        vz = (
            pu * numpy.sinh(u) * numpy.cos(v) - pv * numpy.cosh(u) * numpy.sin(v)
        ) / denom
    return (vR, vz)


def Rz_to_lambdanu(R, z, ac=5.0, Delta=1.0):
    """
    Calculate the prolate spheroidal coordinates (lambda,nu) from galactocentric cylindrical coordinates (R,z) by solving eq. (2.2) in Dejonghe & de Zeeuw (1988a) for (lambda,nu).

    Parameters
    ----------
    R : float
        Galactocentric cylindrical radius
    z : float
        Vertical height
    ac : float, optional
        Axis ratio of the coordinate surfaces (a/c) = sqrt(-a) / sqrt(-g) (default: 5.)
    Delta : float, optional
        Focal distance that defines the spheroidal coordinate system (default: 1.)
        Delta=sqrt(g-a)

    Returns
    -------
    tuple
        (lambda,nu)

    Notes
    -----
    - 2015-02-13 - Written - Trick (MPIA)
    """
    g = Delta**2 / (1.0 - ac**2)
    a = g - Delta**2
    term = R**2 + z**2 - a - g
    discr = (R**2 + z**2 - Delta**2) ** 2 + (4.0 * Delta**2 * R**2)
    l = 0.5 * (term + numpy.sqrt(discr))
    n = 0.5 * (term - numpy.sqrt(discr))
    if isinstance(z, float) and z == 0.0:
        l = R**2 - a
        n = -g
    elif isinstance(z, numpy.ndarray) and numpy.sum(z == 0.0) > 0:
        R = numpy.atleast_1d(R)
        l[z == 0.0] = R[z == 0.0] ** 2 - a
        n[z == 0.0] = -g
    return (l, n)


def Rz_to_lambdanu_jac(R, z, Delta=1.0):
    """
    Calculate the Jacobian of the cylindrical (R,z) to prolate spheroidal (lambda,nu) conversion.

    Parameters
    ----------
    R : float or numpy.ndarray
        Galactocentric cylindrical radius.
    z : float or numpy.ndarray
        Vertical height.
    Delta : float, optional
        Focal distance that defines the spheroidal coordinate system (default: 1.). Delta=sqrt(g-a).

    Returns
    -------
    ndarray
        Jacobian d((lambda,nu))/d((R,z)).

    Notes
    -----
    - 2015-02-13 - Written - Trick (MPIA).
    """

    discr = (R**2 + z**2 - Delta**2) ** 2 + (4.0 * Delta**2 * R**2)
    dldR = R * (1.0 + (R**2 + z**2 + Delta**2) / numpy.sqrt(discr))
    dndR = R * (1.0 - (R**2 + z**2 + Delta**2) / numpy.sqrt(discr))
    dldz = z * (1.0 + (R**2 + z**2 - Delta**2) / numpy.sqrt(discr))
    dndz = z * (1.0 - (R**2 + z**2 - Delta**2) / numpy.sqrt(discr))
    dim = numpy.amax([len(numpy.atleast_1d(R)), len(numpy.atleast_1d(z))])
    jac = numpy.zeros((2, 2, dim))
    jac[0, 0, :] = dldR
    jac[0, 1, :] = dldz
    jac[1, 0, :] = dndR
    jac[1, 1, :] = dndz
    if dim == 1:
        return jac[:, :, 0]
    else:
        return jac


def Rz_to_lambdanu_hess(R, z, Delta=1.0):
    """
    Calculate the Hessian of the cylindrical (R,z) to prolate spheroidal (lambda,nu) conversion.

    Parameters
    ----------
    R : float
        Galactocentric cylindrical radius.
    z : float
        Vertical height.
    Delta : float, optional
        Focal distance that defines the spheroidal coordinate system (default: 1.).
        Delta=sqrt(g-a)

    Returns
    -------
    ndarray
        Hessian [d^2(lambda)/d(R,z)^2 , d^2(nu)/d(R,z)^2].

    Notes
    -----
    - 2015-02-13 - Written - Trick (MPIA).
    """
    D = Delta
    R2 = R**2
    z2 = z**2
    D2 = D**2
    discr = (R2 + z2 - D2) ** 2 + (4.0 * D2 * R2)
    d2ldR2 = (
        1.0
        + (3.0 * R2 + z2 + D2) / discr**0.5
        - (2.0 * R2 * (R2 + z2 + D2) ** 2) / discr**1.5
    )
    d2ndR2 = (
        1.0
        - (3.0 * R2 + z2 + D2) / discr**0.5
        + (2.0 * R2 * (R2 + z2 + D2) ** 2) / discr**1.5
    )
    d2ldz2 = (
        1.0
        + (R2 + 3.0 * z2 - D2) / discr**0.5
        - (2.0 * z2 * (R2 + z2 - D2) ** 2) / discr**1.5
    )
    d2ndz2 = (
        1.0
        - (R2 + 3.0 * z2 - D2) / discr**0.5
        + (2.0 * z2 * (R2 + z2 - D2) ** 2) / discr**1.5
    )
    d2ldRdz = 2.0 * R * z / discr**0.5 * (1.0 - ((R2 + z2) ** 2 - D**4) / discr)
    d2ndRdz = 2.0 * R * z / discr**0.5 * (-1.0 + ((R2 + z2) ** 2 - D**4) / discr)
    dim = numpy.amax([len(numpy.atleast_1d(R)), len(numpy.atleast_1d(z))])
    hess = numpy.zeros((2, 2, 2, dim))
    # Hessian for lambda:
    hess[0, 0, 0, :] = d2ldR2
    hess[0, 0, 1, :] = d2ldRdz
    hess[0, 1, 0, :] = d2ldRdz
    hess[0, 1, 1, :] = d2ldz2
    # Hessian for nu:
    hess[1, 0, 0, :] = d2ndR2
    hess[1, 0, 1, :] = d2ndRdz
    hess[1, 1, 0, :] = d2ndRdz
    hess[1, 1, 1, :] = d2ndz2
    if dim == 1:
        return hess[:, :, :, 0]
    else:
        return hess


def lambdanu_to_Rz(l, n, ac=5.0, Delta=1.0):
    """
    Calculate galactocentric cylindrical coordinates (R,z) from prolate spheroidal coordinates (lambda,nu).

    Parameters
    ----------
    l : float
        Prolate spheroidal coordinate lambda.
    n : float
        Prolate spheroidal coordinate nu.
    ac : float, optional
        Axis ratio of the coordinate surfaces (a/c) = sqrt(-a) / sqrt(-g) (default: 5.).
    Delta : float, optional
        Focal distance that defines the spheroidal coordinate system (default: 1.). Delta=sqrt(g-a)

    Returns
    -------
    tuple
        (R,z)

    Notes
    -----
    - 2015-02-13 - Written - Trick (MPIA)
    """
    g = Delta**2 / (1.0 - ac**2)
    a = g - Delta**2
    r2 = (l + a) * (n + a) / (a - g)
    z2 = (l + g) * (n + g) / (g - a)
    index = (r2 < 0.0) * ((n + a) > 0.0) * ((n + a) < 1e-10)
    if numpy.any(index):
        if isinstance(r2, numpy.ndarray):
            r2[index] = 0.0
        else:
            r2 = 0.0
    index = (z2 < 0.0) * ((n + g) < 0.0) * ((n + g) > -1e-10)
    if numpy.any(index):
        if isinstance(z2, numpy.ndarray):
            z2[index] = 0.0
        else:
            z2 = 0.0
    return (numpy.sqrt(r2), numpy.sqrt(z2))


@scalarDecorator
@degreeDecorator([0, 1], [0, 1])
def radec_to_custom(ra, dec, T=None, degree=False):
    """
    Transform from equatorial coordinates to a custom set of sky coordinates

    Parameters
    ----------
    ra : float or numpy.ndarray
        Right ascension.
    dec : float or numpy.ndarray
        Declination.
    T : numpy.ndarray, optional
        Matrix defining the transformation: new_rect= T dot old_rect, where old_rect = [cos(dec)cos(ra),cos(dec)sin(ra),sin(dec)] and similar for new_rect.
    degree : bool, optional
        If True, ra and dec are given in degree and l and b will be as well.

    Returns
    -------
    tuple
        (custom longitude, custom latitude) (with longitude -180 to 180)

    Notes
    -----
    - 2009-11-12 - Written - Bovy (NYU)
    - 2014-06-14 - Re-written w/ numpy functions for speed and w/ decorators for beauty - Bovy (IAS)
    - 2019-03-02 - adjusted angle ranges - Nathaniel (UofT)
    """
    if T is None:
        raise ValueError("Must set T= for radec_to_custom")
    # Whether to use degrees and scalar input is handled by decorators
    XYZ = numpy.array(
        [numpy.cos(dec) * numpy.cos(ra), numpy.cos(dec) * numpy.sin(ra), numpy.sin(dec)]
    )
    galXYZ = numpy.dot(T, XYZ)
    b = numpy.arcsin(galXYZ[2])  # [-pi/2, pi/2]
    l = numpy.arctan2(galXYZ[1], galXYZ[0])
    l[l < 0] += 2 * numpy.pi  # fix range to [0, 2 pi]
    out = numpy.array([l, b])
    return out.T


@scalarDecorator
@degreeDecorator([2, 3], [])
def pmrapmdec_to_custom(pmra, pmdec, ra, dec, T=None, degree=False):
    """
    Rotate proper motions in (ra,dec) to proper motions in a custom set of sky coordinates (phi1,phi2)

    Parameters
    ----------
    pmra : float or numpy.ndarray
        Proper motion in ra (multiplied with cos(dec)) [mas/yr].
    pmdec : float or numpy.ndarray
        Proper motion in dec [mas/yr].
    ra : float or numpy.ndarray
        Right ascension.
    dec : float or numpy.ndarray
        Declination.
    T : numpy.ndarray, optional
        Matrix defining the transformation: new_rect= T dot old_rect, where old_rect = [cos(dec)cos(ra),cos(dec)sin(ra),sin(dec)] and similar for new_rect.
    degree : bool, optional
        If True, ra and dec are given in degrees (default=False).

    Returns
    -------
    tuple
        (pmphi1 x cos(phi2),pmph2) for vector inputs [:,2]

    Notes
    -----
    - 2016-10-24 - Written - Bovy (UofT/CCA)
    - 2019-03-09 - uses custom_to_radec - Nathaniel Starkman (UofT)
    """
    if T is None:
        raise ValueError("Must set T= for pmrapmdec_to_custom")
    # Need to figure out ra_ngp and dec_ngp for this custom set of sky coords
    ra_ngp, dec_ngp = custom_to_radec(0.0, numpy.pi / 2, T=T)
    # Whether to use degrees and scalar input is handled by decorators
    dec[dec == dec_ngp] += 10.0**-16  # deal w/ pole.
    sindec_ngp = numpy.sin(dec_ngp)
    cosdec_ngp = numpy.cos(dec_ngp)
    sindec = numpy.sin(dec)
    cosdec = numpy.cos(dec)
    sinrarangp = numpy.sin(ra - ra_ngp)
    cosrarangp = numpy.cos(ra - ra_ngp)
    cosphi = sindec_ngp * cosdec - cosdec_ngp * sindec * cosrarangp
    sinphi = sinrarangp * cosdec_ngp
    norm = numpy.sqrt(cosphi**2.0 + sinphi**2.0)
    cosphi /= norm
    sinphi /= norm
    return (
        numpy.array([[cosphi, -sinphi], [sinphi, cosphi]]).T
        * numpy.array([[pmra, pmra], [pmdec, pmdec]]).T
    ).sum(-1)


def custom_to_radec(phi1, phi2, T=None, degree=False):
    """
    Rotate a custom set of sky coordinates (phi1,phi2) to (ra,dec) given the rotation matrix T for (ra,dec) -> (phi1,phi2)

    Parameters
    ----------
    phi1 : float or numpy.ndarray
        Custom sky coord.
    phi2 : float or numpy.ndarray
        Custom sky coord.
    T : numpy.ndarray, optional
        Matrix defining the transformation: new_rect= T dot old_rect, where old_rect = [cos(dec)cos(ra),cos(dec)sin(ra),sin(dec)] and similar for new_rect.
    degree : bool, optional
        If True, phi1 and phi2 are given in degrees (default=False).

    Returns
    -------
    tuple
        (ra,dec) for vector inputs [:,2]

    Notes
    -----
    - 2018-10-23 - Written - Starkman (UofT)
    """
    if T is None:
        raise ValueError("Must set T= for custom_to_radec")
    return radec_to_custom(
        phi1,
        phi2,
        T=numpy.transpose(T),
        degree=degree,  # T.T = inv(T)
    )


def custom_to_pmrapmdec(pmphi1, pmphi2, phi1, phi2, T=None, degree=False):
    """
    Rotate proper motions in a custom set of sky coordinates (phi1,phi2) to (ra,dec) given the rotation matrix T for (ra,dec) -> (phi1,phi2)

    Parameters
    ----------
    pmphi1 : float or numpy.ndarray
        Proper motion in phi1 (multiplied with cos(phi2)) [mas/yr].
    pmphi2 : float or numpy.ndarray
        Proper motion in phi2 [mas/yr].
    phi1 : float or numpy.ndarray
        Custom sky coord.
    phi2 : float or numpy.ndarray
        Custom sky coord.
    T : numpy.ndarray, optional
        Matrix defining the transformation: new_rect= T dot old_rect, where old_rect = [cos(dec)cos(ra),cos(dec)sin(ra),sin(dec)] and similar for new_rect.
    degree : bool, optional
        If True, phi1 and phi2 are given in degrees (default=False).

    Returns
    -------
    tuple
        (pmra x cos(dec), dec) for vector inputs [:,2]

    Notes
    -----
    - 2019-03-02 - Written - Nathaniel Starkman (UofT)
    """
    if T is None:
        raise ValueError("Must set T= for custom_to_pmrapmdec")
    return pmrapmdec_to_custom(
        pmphi1,
        pmphi2,
        phi1,
        phi2,
        T=numpy.transpose(T),
        degree=degree,  # T.T = inv(T)
    )


def get_epoch_angles(epoch=2000.0):
    """
    Get the angles relevant for the transformation from ra, dec to l,b for the given epoch.

    Parameters
    ----------
    epoch : float or str, optional
        Epoch of ra,dec. Default is 2000.0. When internally using astropy's coordinate transformations, epoch can be None for ICRS, 'JXXXX' for FK5, and 'BXXXX' for FK4. For B1950 FK4 with no E aberration terms is assumed.

    Returns
    -------
    tuple
        Set of angles.

    Notes
    -----
    - 2010-04-07 - Written - Bovy (NYU)
    - 2016-05-13 - Added support for using astropy's coordinate transformations and for non-standard epochs - Bovy (UofT)
    - 2018-04-18 - Edited J2000 angles to be fully consistent with astropy - Bovy (UofT)
    """
    if epoch == 2000.0:
        # Following astropy's definition here
        theta = 122.9319185680026 / 180.0 * numpy.pi
        dec_ngp = 27.12825118085622 / 180.0 * numpy.pi
        ra_ngp = 192.8594812065348 / 180.0 * numpy.pi
    elif epoch == 1950.0:
        theta = 123.0 / 180.0 * numpy.pi
        dec_ngp = 27.4 / 180.0 * numpy.pi
        ra_ngp = 192.25 / 180.0 * numpy.pi
    elif epoch == None:  # obtained below
        theta = theta_icrs
        dec_ngp = dec_ngp_icrs
        ra_ngp = ra_ngp_icrs
    elif _APY_LOADED:
        # Use astropy to get the angles
        epoch, frame = _parse_epoch_frame_apy(epoch)
        c = apycoords.SkyCoord(
            180.0 * units.deg, 90.0 * units.deg, frame=frame, equinox=epoch
        )
        c = c.transform_to(apycoords.Galactic)
        theta = c.l.to(units.rad).value
        c = apycoords.SkyCoord(180.0 * units.deg, 90.0 * units.deg, frame="galactic")
        if not epoch is None and "J" in epoch:
            c = c.transform_to(apycoords.FK5(equinox=epoch))
        elif not epoch is None and "B" in epoch:
            c = c.transform_to(apycoords.FK4(equinox=epoch))
        else:  # pragma: no cover
            raise ValueError(
                "epoch input not understood; should be None for ICRS, JXXXX, or BXXXX"
            )
        dec_ngp = c.dec.to(units.rad).value
        ra_ngp = c.ra.to(units.rad).value
    else:
        raise OSError(
            "Only epochs 1950 and 2000 are supported if you don't have astropy"
        )
    return (theta, dec_ngp, ra_ngp)


# Get ICRS angles once when astropy is installed
if _APY_LOADED:
    c = apycoords.SkyCoord(180.0 * units.deg, 90.0 * units.deg, frame="icrs")
    c = c.transform_to(apycoords.Galactic)
    theta_icrs = c.l.to(units.rad).value
    c = apycoords.SkyCoord(180.0 * units.deg, 90.0 * units.deg, frame="galactic")
    c = c.transform_to(apycoords.ICRS)
    dec_ngp_icrs = c.dec.to(units.rad).value
    ra_ngp_icrs = c.ra.to(units.rad).value
else:
    theta_icrs = 2.1455668515225916
    dec_ngp_icrs = 0.4734773249532947
    ra_ngp_icrs = 3.366032882941063


def _parse_epoch_frame_apy(epoch):
    if epoch == 2000.0 or epoch == "2000":
        epoch = "J2000"
    elif epoch == 1950.0 or epoch == "1950":
        epoch = "B1950"
    if not epoch is None and "J" in epoch:
        frame = "fk5"
    elif not epoch is None and "B" in epoch:
        frame = "fk4"
    else:
        frame = "icrs"
    return (epoch, frame)


# Matrix to rotate to the astropy Galactocentric frame: astropy's
# Galactocentric frame is slightly off from the one that we get by simply
# taking Galactic coordinates and transforming them: transformation from
# Bovy (2011) maps NGP --> (0,0,1), astropy to (v. small, v. small, 1-v. small)
# so we rotate Bovy (2011) such that we agree; for that we compute what NGP
# goes to using astropy's transformations
theta, dec_ngp, ra_ngp = get_epoch_angles(None)  # None = ICRS, basis for astropy
dec_gc, ra_gc = (
    -28.936175 / 180.0 * numpy.pi,
    266.4051 / 180.0 * numpy.pi,
)  # from apy def.
eta = 58.5986320306 / 180.0 * numpy.pi  # astropy 'roll' angle
gc_vec = numpy.array(
    [
        numpy.cos(theta)
        * (
            -numpy.sin(dec_ngp) * numpy.cos(dec_gc) * numpy.cos(ra_gc - ra_ngp)
            + numpy.cos(dec_ngp) * numpy.sin(dec_gc)
        )
        + numpy.sin(theta) * numpy.cos(dec_gc) * numpy.sin(ra_gc - ra_ngp),
        numpy.sin(theta)
        * (
            -numpy.sin(dec_ngp) * numpy.cos(dec_gc) * numpy.cos(ra_gc - ra_ngp)
            + numpy.cos(dec_ngp) * numpy.sin(dec_gc)
        )
        - numpy.cos(theta) * numpy.cos(dec_gc) * numpy.sin(ra_gc - ra_ngp),
        numpy.cos(dec_ngp) * numpy.cos(dec_gc) * numpy.cos(ra_gc - ra_ngp)
        + numpy.sin(dec_ngp) * numpy.sin(dec_gc),
    ]
)
galcen_extra_rot1 = _rotate_to_arbitrary_vector(
    numpy.atleast_2d(gc_vec),
    numpy.array([1.0, 0.0, 0.0]),
    inv=False,
    _dontcutsmall=True,
)[0]
ngp_vec = numpy.dot(
    galcen_extra_rot1,
    numpy.array(
        [
            -numpy.cos(dec_gc) * numpy.cos(dec_ngp) * numpy.cos(ra_ngp - ra_gc)
            - numpy.sin(dec_gc) * numpy.sin(dec_ngp),
            numpy.cos(eta) * numpy.cos(dec_ngp) * numpy.sin(ra_ngp - ra_gc)
            + numpy.sin(eta)
            * (
                -numpy.sin(dec_gc) * numpy.cos(dec_ngp) * numpy.cos(ra_ngp - ra_gc)
                + numpy.cos(dec_gc) * numpy.sin(dec_ngp)
            ),
            -numpy.sin(eta) * numpy.cos(dec_ngp) * numpy.sin(ra_ngp - ra_gc)
            + numpy.cos(eta)
            * (
                -numpy.sin(dec_gc) * numpy.cos(dec_ngp) * numpy.cos(ra_ngp - ra_gc)
                + numpy.cos(dec_gc) * numpy.sin(dec_ngp)
            ),
        ]
    ),
)
galcen_extra_rot2 = _rotate_to_arbitrary_vector(
    numpy.atleast_2d(ngp_vec),
    numpy.array([0.0, 0.0, 1.0]),
    inv=True,
    _dontcutsmall=True,
)[0]
# Leave x axis alone, because in place by rot1
galcen_extra_rot2[0, 0] = 1.0
galcen_extra_rot2[0, 1:] = 0.0
galcen_extra_rot2[1:, 0] = 0.0
galcen_extra_rot = numpy.dot(galcen_extra_rot2, galcen_extra_rot1)
