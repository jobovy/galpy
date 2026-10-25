import numpy
import pytest

from galpy.util import conversion


@pytest.mark.parametrize("scale", ["ro", "vo"])
@pytest.mark.parametrize("shape", [(), (0,), (2,), (2, 1)])
@pytest.mark.parametrize(
    "parser",
    [
        conversion.parse_length,
        conversion.parse_velocity,
        conversion.parse_time,
        conversion.parse_mass,
        conversion.parse_energy,
        conversion.parse_angmom,
        conversion.parse_frequency,
        conversion.parse_force,
        conversion.parse_dens,
        conversion.parse_surfdens,
        conversion.parse_numdens,
    ],
)
def test_numeric_array_scales(parser, scale, shape):
    value = numpy.array([0.5, 1.0])
    scales = numpy.full(shape, 8.0 if scale == "ro" else 220.0)
    scales.flags.writeable = False
    # Numeric inputs already use internal units and must pass through unchanged.
    assert parser(value, **{scale: scales}) is value


@pytest.mark.parametrize("scale", ["ro", "vo"])
@pytest.mark.parametrize("value", ["invalid", [8.0, 9.0], numpy.array(["invalid"])])
def test_invalid_array_scales(scale, value):
    with pytest.raises(RuntimeError, match=f"{scale}="):
        conversion.parse_frequency(1.0, **{scale: value})


def test_quantity_array_scales():
    units = pytest.importorskip("astropy.units")
    ro = numpy.array([8.0, 9.0])
    vo = numpy.array([200.0, 230.0])
    numpy.testing.assert_allclose(
        conversion.parse_length(2.0 * units.kpc, ro=ro, vo=vo), 2.0 / ro
    )
    numpy.testing.assert_allclose(
        conversion.parse_velocity(100.0 * units.km / units.s, ro=ro, vo=vo),
        100.0 / vo,
    )
    numpy.testing.assert_allclose(
        conversion.parse_frequency(30.0 * units.km / units.s / units.kpc, ro=ro, vo=vo),
        30.0 * ro / vo,
    )


def test_jacobi_with_array_scales():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    orbit = Orbit(
        [[1.0, 0.1, 1.0, 0.1, 0.1, 0.0], [1.2, -0.1, 0.9, 0.2, -0.1, 0.5]],
        ro=numpy.array([8.0, 9.0]),
        vo=numpy.array([200.0, 230.0]),
    )
    expected = orbit.E(pot=MWPotential2014) - orbit.Lz() * orbit._vo / orbit._ro
    numpy.testing.assert_allclose(orbit.Jacobi(pot=MWPotential2014), expected)


@pytest.mark.parametrize("size", [2, 3])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("integrated", [False, True])
def test_jacobi_with_quantity_pattern_and_array_scales(size, vector, integrated):
    units = pytest.importorskip("astropy.units")
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    orbit = Orbit(
        [[1.0, 0.1, 1.0, 0.1, 0.1, 0.0]] * size,
        ro=numpy.linspace(8.0, 9.0, size),
        vo=numpy.linspace(200.0, 230.0, size),
    )
    omega = numpy.array([10.0, -5.0, 30.0]) if vector else 30.0
    omega = omega * units.km / units.s / units.kpc
    times = numpy.linspace(0.0, 0.1, 5) if integrated else 0.0
    if integrated:
        orbit.integrate(
            times, MWPotential2014, method="odeint", numcores=1, progressbar=False
        )
    expected = numpy.array(
        [orbit[i].Jacobi(times, pot=MWPotential2014, OmegaP=omega) for i in range(size)]
    )
    actual = orbit.Jacobi(times, pot=MWPotential2014, OmegaP=omega)
    numpy.testing.assert_allclose(actual, expected)
