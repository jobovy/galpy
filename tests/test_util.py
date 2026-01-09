# Test the functions in galpy/util/__init__.py
import numpy


def test_save_pickles():
    import os
    import pickle
    import tempfile

    from galpy.util import save_pickles

    savethis = numpy.linspace(0.0, 100.0, 1001)
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        save_pickles(tmp_savefilename, savethis)
        savefile = open(tmp_savefilename, "rb")
        restorethis = pickle.load(savefile)
        savefile.close()
        assert numpy.all(numpy.fabs(restorethis - savethis) < 10.0**-10.0), (
            "save_pickles did not work as expected"
        )
    finally:
        os.remove(tmp_savefilename)
    # Also test the handling of KeyboardInterrupt
    try:
        save_pickles(tmp_savefilename, savethis, testKeyboardInterrupt=True)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError(
            "save_pickles with testKeyboardInterrupt=True did not raise KeyboardInterrupt"
        )
    savefile = open(tmp_savefilename, "rb")
    restorethis = pickle.load(savefile)
    savefile.close()
    assert numpy.all(numpy.fabs(restorethis - savethis) < 10.0**-10.0), (
        "save_pickles did not work as expected when KeyboardInterrupted"
    )
    if os.path.exists(tmp_savefilename):
        os.remove(tmp_savefilename)
    return None


def test_logsumexp():
    from galpy.util import logsumexp

    sumthis = numpy.array([[0.0, 1.0]])
    sum = numpy.log(numpy.exp(0.0) + numpy.exp(1.0))
    assert numpy.all(numpy.fabs(logsumexp(sumthis, axis=0) - sumthis) < 10.0**-10.0), (
        "galpy.util.logsumexp did not work as expected"
    )
    assert numpy.fabs(logsumexp(sumthis, axis=1) - sum) < 10.0**-10.0, (
        "galpy.util.logsumexp did not work as expected"
    )
    assert numpy.fabs(logsumexp(sumthis, axis=None) - sum) < 10.0**-10.0, (
        "galpy.util.logsumexp did not work as expected"
    )
    return None


def test_fast_cholesky_invert():
    from galpy.util import fast_cholesky_invert

    matrix = numpy.array([[2.0, 1.0], [1.0, 4.0]])
    invmatrix = fast_cholesky_invert(matrix)
    unit = numpy.dot(invmatrix, matrix)
    assert numpy.all(numpy.fabs(numpy.diag(unit) - 1.0) < 10.0**-8.0), (
        "fast_cholesky_invert did not work as expected"
    )
    assert numpy.fabs(unit[0, 1] - 0.0) < 10.0**-8.0, (
        "fast_cholesky_invert did not work as expected"
    )
    assert numpy.fabs(unit[1, 0] - 0.0) < 10.0**-8.0, (
        "fast_cholesky_invert did not work as expected"
    )
    # Check the other way around
    unit = numpy.dot(matrix, invmatrix)
    assert numpy.all(numpy.fabs(numpy.diag(unit) - 1.0) < 10.0**-8.0), (
        "fast_cholesky_invert did not work as expected"
    )
    assert numpy.fabs(unit[0, 1] - 0.0) < 10.0**-8.0, (
        "fast_cholesky_invert did not work as expected"
    )
    assert numpy.fabs(unit[1, 0] - 0.0) < 10.0**-8.0, (
        "fast_cholesky_invert did not work as expected"
    )
    # Also check determinant
    invmatrix, logdet = fast_cholesky_invert(matrix, logdet=True)
    assert numpy.fabs(logdet - numpy.log(7.0)) < 10.0**-8.0, (
        "fast_cholesky_invert's determinant did not work as expected"
    )
    return None


def test_quadpack():
    from galpy.util.quadpack import dblquad

    int = dblquad(lambda y, x: 4.0 * x * y, 0.0, 1.0, lambda z: 0.0, lambda z: 1.0)
    assert numpy.fabs(int[0] - 1.0) < int[1], (
        "galpy.util.quadpack.dblquad did not work as expected"
    )
    return None


# Tests for galpy.util.config
def test_check_config_valid():
    """Test check_config returns True for a valid complete configuration."""
    import configparser

    from galpy.util.config import check_config, default_configuration

    config = configparser.ConfigParser()
    for sec_key in default_configuration.keys():
        config.add_section(sec_key)
        for key in default_configuration[sec_key]:
            config.set(sec_key, key, default_configuration[sec_key][key])
    assert check_config(config), "check_config should return True for valid config"
    return None


def test_check_config_missing_section():
    """Test check_config returns False when a section is missing."""
    import configparser

    from galpy.util.config import check_config, default_configuration

    config = configparser.ConfigParser()
    # Only add the first section, skip the rest
    sections = list(default_configuration.keys())
    config.add_section(sections[0])
    for key in default_configuration[sections[0]]:
        config.set(sections[0], key, default_configuration[sections[0]][key])
    assert not check_config(config), (
        "check_config should return False when sections are missing"
    )
    return None


def test_check_config_missing_key():
    """Test check_config returns False when a key is missing."""
    import configparser

    from galpy.util.config import check_config, default_configuration

    config = configparser.ConfigParser()
    for sec_key in default_configuration.keys():
        config.add_section(sec_key)
        keys = list(default_configuration[sec_key].keys())
        # Skip the first key in each section
        for key in keys[1:]:
            config.set(sec_key, key, default_configuration[sec_key][key])
    assert not check_config(config), (
        "check_config should return False when keys are missing"
    )
    return None


def test_fix_config_none():
    """Test fix_config returns default configuration when input is None."""
    from galpy.util.config import check_config, default_configuration, fix_config

    fixed = fix_config(None)
    assert check_config(fixed), "fix_config(None) should produce a valid config"
    # Verify all default values are present
    for sec_key in default_configuration.keys():
        for key in default_configuration[sec_key]:
            assert fixed.get(sec_key, key) == default_configuration[sec_key][key], (
                f"fix_config(None) should have default value for {sec_key}/{key}"
            )
    return None


def test_fix_config_partial():
    """Test fix_config fills in missing values from defaults."""
    import configparser

    from galpy.util.config import check_config, default_configuration, fix_config

    # Create a partial config with only normalization section
    partial = configparser.ConfigParser()
    partial.add_section("normalization")
    partial.set("normalization", "ro", "10.")
    partial.set("normalization", "vo", "250.")

    fixed = fix_config(partial)
    assert check_config(fixed), "fix_config should produce a valid config"
    # Custom values should be preserved
    assert fixed.get("normalization", "ro") == "10.", (
        "fix_config should preserve existing ro value"
    )
    assert fixed.get("normalization", "vo") == "250.", (
        "fix_config should preserve existing vo value"
    )
    # Missing sections should get defaults
    assert (
        fixed.get("astropy", "astropy-units")
        == default_configuration["astropy"]["astropy-units"]
    ), "fix_config should fill in missing astropy section with defaults"
    return None


def test_fix_config_preserves_extra_values():
    """Test fix_config preserves custom values that exist in the input."""
    import configparser

    from galpy.util.config import default_configuration, fix_config

    # Create config with all sections but one modified value
    config = configparser.ConfigParser()
    for sec_key in default_configuration.keys():
        config.add_section(sec_key)
        for key in default_configuration[sec_key]:
            config.set(sec_key, key, default_configuration[sec_key][key])
    # Modify one value
    config.set("normalization", "ro", "7.5")
    config.set("warnings", "verbose", "True")

    fixed = fix_config(config)
    assert fixed.get("normalization", "ro") == "7.5", (
        "fix_config should preserve custom ro value"
    )
    assert fixed.get("warnings", "verbose") == "True", (
        "fix_config should preserve custom verbose value"
    )
    return None


def test_fix_config_missing_key_in_section():
    """Test fix_config fills in a missing key within an existing section."""
    import configparser

    from galpy.util.config import check_config, default_configuration, fix_config

    # Create config with normalization section but missing 'vo' key
    config = configparser.ConfigParser()
    for sec_key in default_configuration.keys():
        config.add_section(sec_key)
        for key in default_configuration[sec_key]:
            if sec_key == "normalization" and key == "vo":
                continue  # Skip vo
            config.set(sec_key, key, default_configuration[sec_key][key])

    fixed = fix_config(config)
    assert check_config(fixed), "fix_config should produce a valid config"
    assert (
        fixed.get("normalization", "vo") == default_configuration["normalization"]["vo"]
    ), "fix_config should fill in missing vo with default"
    return None


def test_set_ro_float():
    """Test set_ro with a float value."""
    from galpy.util.config import __config__, set_ro

    original = __config__.get("normalization", "ro")
    try:
        set_ro(9.5)
        assert __config__.get("normalization", "ro") == "9.5", (
            "set_ro should update ro in config"
        )
    finally:
        __config__.set("normalization", "ro", original)
    return None


def test_set_vo_float():
    """Test set_vo with a float value."""
    from galpy.util.config import __config__, set_vo

    original = __config__.get("normalization", "vo")
    try:
        set_vo(200.0)
        assert __config__.get("normalization", "vo") == "200.0", (
            "set_vo should update vo in config"
        )
    finally:
        __config__.set("normalization", "vo", original)
    return None


def test_set_ro_quantity():
    """Test set_ro with an astropy Quantity."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.config import __config__, set_ro

    original = __config__.get("normalization", "ro")
    try:
        set_ro(8500.0 * units.pc)  # 8.5 kpc in parsecs
        assert numpy.fabs(float(__config__.get("normalization", "ro")) - 8.5) < 1e-10, (
            "set_ro should convert Quantity to kpc"
        )
    finally:
        __config__.set("normalization", "ro", original)
    return None


def test_set_vo_quantity():
    """Test set_vo with an astropy Quantity."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.config import __config__, set_vo

    original = __config__.get("normalization", "vo")
    try:
        set_vo(220000.0 * units.m / units.s)  # 220 km/s in m/s
        assert (
            numpy.fabs(float(__config__.get("normalization", "vo")) - 220.0) < 1e-10
        ), "set_vo should convert Quantity to km/s"
    finally:
        __config__.set("normalization", "vo", original)
    return None


# Tests for galpy.potential._repr_utils
# These use lightweight mock objects for pure string tests and real potentials
# for introspection-based tests


class _MockPotentialPhysical:
    """Mock potential with physical output attributes for testing."""

    def __init__(self, roSet=False, voSet=False, ro=8.0, vo=220.0):
        self._roSet = roSet
        self._voSet = voSet
        self._ro = ro
        self._vo = vo


def test_build_params_string_basic():
    """Test _build_params_string with a real potential object."""
    from galpy.potential import PlummerPotential
    from galpy.potential._repr_utils import _build_params_string

    pot = PlummerPotential(amp=2.0, b=1.5)
    params = _build_params_string(pot)
    assert "amp=2.0" in params, f"Expected 'amp=2.0' in params, got {params}"
    assert "b=1.5" in params, f"Expected 'b=1.5' in params, got {params}"
    return None


def test_build_params_string_exclude():
    """Test _build_params_string with custom exclusions."""
    from galpy.potential import PlummerPotential
    from galpy.potential._repr_utils import _build_params_string

    pot = PlummerPotential(amp=2.0, b=1.5)
    # Exclude 'amp' from output
    params = _build_params_string(pot, exclude_params=["self", "amp", "ro", "vo"])
    assert "amp=2.0" not in params, f"Did not expect 'amp=2.0' in params, got {params}"
    assert "b=1.5" in params, f"Expected 'b=1.5' in params, got {params}"
    return None


def test_build_params_string_default_excludes_ro_vo():
    """Test _build_params_string excludes ro and vo by default."""
    from galpy.potential import PlummerPotential
    from galpy.potential._repr_utils import _build_params_string

    pot = PlummerPotential(amp=1.0, b=0.5, ro=8.0, vo=220.0)
    params = _build_params_string(pot)
    # ro and vo should be excluded by default
    param_str = " ".join(params)
    assert "ro=" not in param_str, f"Did not expect 'ro=' in params, got {params}"
    assert "vo=" not in param_str, f"Did not expect 'vo=' in params, got {params}"
    assert "amp=" in param_str, f"Expected 'amp=' in params, got {params}"
    return None


def test_build_params_string_no_matching_attrs():
    """Test _build_params_string when object has no matching attributes."""
    from galpy.potential._repr_utils import _build_params_string

    class _MockNoAttrs:
        def __init__(self, x=1):
            # Don't store x as _x or x, so no match
            self.something_else = x

    obj = _MockNoAttrs(x=5)
    params = _build_params_string(obj)
    assert params == [], f"Expected empty list, got {params}"
    return None


def test_build_physical_output_string_off():
    """Test _build_physical_output_string when physical outputs are off."""
    from galpy.potential._repr_utils import _build_physical_output_string

    obj = _MockPotentialPhysical(roSet=False, voSet=False)
    result = _build_physical_output_string(obj)
    assert "physical outputs off" in result, (
        f"Expected 'physical outputs off', got '{result}'"
    )
    assert "ro=" not in result, f"Did not expect 'ro=' in result, got '{result}'"
    assert "vo=" not in result, f"Did not expect 'vo=' in result, got '{result}'"
    return None


def test_build_physical_output_string_fully_on():
    """Test _build_physical_output_string when both ro and vo are set."""
    from galpy.potential._repr_utils import _build_physical_output_string

    obj = _MockPotentialPhysical(roSet=True, voSet=True, ro=8.5, vo=230.0)
    result = _build_physical_output_string(obj)
    assert "physical outputs fully on" in result, (
        f"Expected 'physical outputs fully on', got '{result}'"
    )
    assert "ro=8.5 kpc" in result, f"Expected 'ro=8.5 kpc', got '{result}'"
    assert "vo=230.0 km/s" in result, f"Expected 'vo=230.0 km/s', got '{result}'"
    return None


def test_build_physical_output_string_ro_only():
    """Test _build_physical_output_string when only ro is set."""
    from galpy.potential._repr_utils import _build_physical_output_string

    obj = _MockPotentialPhysical(roSet=True, voSet=False, ro=7.5)
    result = _build_physical_output_string(obj)
    assert "partially on (ro only)" in result, (
        f"Expected 'partially on (ro only)', got '{result}'"
    )
    assert "ro=7.5 kpc" in result, f"Expected 'ro=7.5 kpc', got '{result}'"
    assert "vo=" not in result, f"Did not expect 'vo=' in result, got '{result}'"
    return None


def test_build_physical_output_string_vo_only():
    """Test _build_physical_output_string when only vo is set."""
    from galpy.potential._repr_utils import _build_physical_output_string

    obj = _MockPotentialPhysical(roSet=False, voSet=True, vo=200.0)
    result = _build_physical_output_string(obj)
    assert "partially on (vo only)" in result, (
        f"Expected 'partially on (vo only)', got '{result}'"
    )
    assert "vo=200.0 km/s" in result, f"Expected 'vo=200.0 km/s', got '{result}'"
    assert "ro=" not in result, f"Did not expect 'ro=' in result, got '{result}'"
    return None


def test_build_repr_basic():
    """Test _build_repr with a real potential object."""
    from galpy.potential import PlummerPotential
    from galpy.potential._repr_utils import _build_repr

    pot = PlummerPotential(amp=2.0, b=1.5)
    result = _build_repr(pot)
    assert "PlummerPotential" in result, f"Expected class name in repr, got '{result}'"
    assert "internal parameters:" in result, (
        f"Expected 'internal parameters:' in repr, got '{result}'"
    )
    assert "amp=2.0" in result, f"Expected 'amp=2.0' in repr, got '{result}'"
    assert "physical outputs off" in result, (
        f"Expected 'physical outputs off' in repr, got '{result}'"
    )
    return None


def test_build_repr_with_physical():
    """Test _build_repr with physical outputs enabled."""
    from galpy.potential import PlummerPotential
    from galpy.potential._repr_utils import _build_repr

    pot = PlummerPotential(amp=1.0, b=0.5, ro=8.0, vo=220.0)
    result = _build_repr(pot)
    assert "PlummerPotential" in result, f"Expected class name in repr, got '{result}'"
    assert "physical outputs fully on" in result, (
        f"Expected 'physical outputs fully on', got '{result}'"
    )
    assert "ro=8.0 kpc" in result, f"Expected 'ro=8.0 kpc', got '{result}'"
    return None


def test_build_repr_custom_class_name():
    """Test _build_repr with a custom class name."""
    from galpy.potential import PlummerPotential
    from galpy.potential._repr_utils import _build_repr

    pot = PlummerPotential(amp=1.0, b=0.5)
    result = _build_repr(pot, class_name="CustomPotential")
    assert "CustomPotential" in result, (
        f"Expected 'CustomPotential' in repr, got '{result}'"
    )
    assert "PlummerPotential" not in result, (
        f"Did not expect 'PlummerPotential' in repr, got '{result}'"
    )
    return None


def test_strip_physical_output_info_fully_on():
    """Test _strip_physical_output_info removes 'fully on' info."""
    from galpy.potential._repr_utils import _strip_physical_output_info

    input_str = "PlummerPotential with internal parameters: amp=1.0 and physical outputs fully on, using ro=8.0 kpc and vo=220.0 km/s"
    result = _strip_physical_output_info(input_str)
    assert "physical outputs" not in result, (
        f"Expected physical output info to be stripped, got '{result}'"
    )
    assert "PlummerPotential" in result, (
        f"Expected class name preserved, got '{result}'"
    )
    assert "amp=1.0" in result, f"Expected params preserved, got '{result}'"
    return None


def test_strip_physical_output_info_off():
    """Test _strip_physical_output_info removes 'off' info."""
    from galpy.potential._repr_utils import _strip_physical_output_info

    input_str = "NFWPotential with internal parameters: a=5.0 and physical outputs off"
    result = _strip_physical_output_info(input_str)
    assert "physical outputs" not in result, (
        f"Expected physical output info to be stripped, got '{result}'"
    )
    assert "NFWPotential" in result, f"Expected class name preserved, got '{result}'"
    return None


def test_strip_physical_output_info_partial():
    """Test _strip_physical_output_info removes partial physical info."""
    from galpy.potential._repr_utils import _strip_physical_output_info

    input_str = "LogPotential with internal parameters: q=0.9 and physical outputs partially on (ro only), using ro=8.0 kpc"
    result = _strip_physical_output_info(input_str)
    assert "physical outputs" not in result, (
        f"Expected physical output info to be stripped, got '{result}'"
    )
    assert "LogPotential" in result, f"Expected class name preserved, got '{result}'"
    return None


def test_strip_physical_output_info_no_match():
    """Test _strip_physical_output_info returns unchanged string when no match."""
    from galpy.potential._repr_utils import _strip_physical_output_info

    input_str = "SimplePotential with amp=1.0"
    result = _strip_physical_output_info(input_str)
    assert result == input_str, f"Expected unchanged string, got '{result}'"
    return None
