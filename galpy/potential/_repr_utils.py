"""Utility functions for building string representations of potential objects."""

import inspect
import re


def _build_params_string(obj, exclude_params=None):
    """
    Build a string of internal parameters for a potential object.

    Parameters
    ----------
    obj : object
        The potential object
    exclude_params : list, optional
        List of parameter names to exclude (default: ["self", "ro", "vo"])

    Returns
    -------
    list
        List of "param=value" strings

    """
    if exclude_params is None:
        exclude_params = ["self", "ro", "vo"]

    params = []

    # Get __init__ signature to find parameter names
    try:
        # First get the right __class__.__init__ function: for wrappers, we want the
        # one of the actual class, not the _class used in the parentWrapperPotential
        if obj.__class__.__name__.startswith("_"):
            class_name = obj.__class__.__name__[1:]
            for cls in obj.__class__.__mro__:
                if cls.__name__ == class_name:
                    init_func = cls.__init__
                    break
        else:
            init_func = obj.__class__.__init__
        sig = inspect.signature(init_func)
        init_params = [p for p in sig.parameters.keys() if p not in exclude_params]

        # Look for corresponding attributes in __dict__
        for param in init_params:
            # Try common attribute naming conventions
            attr_candidates = [
                f"_{param}",  # _b, _amp, etc.
                param,  # direct name
            ]

            for attr in attr_candidates:
                if attr in obj.__dict__:
                    value = obj.__dict__[attr]
                    if value is not None:
                        params.append(f"{param}={value}")
                    break
    except Exception:  # pragma: no cover
        # If anything goes wrong with introspection, just continue
        pass

    return params


def _build_physical_output_string(obj):
    """
    Build a string describing the physical output status.

    Parameters
    ----------
    obj : object
        The potential object with _roSet, _voSet, _ro, and _vo attributes

    Returns
    -------
    str
        String describing physical output status, or empty string if not applicable

    """
    # Build physical output status string
    physical_parts = []
    if obj._roSet and obj._voSet:
        physical_parts.append("physical outputs fully on")
    elif obj._roSet:
        physical_parts.append("physical outputs partially on (ro only)")
    elif obj._voSet:
        physical_parts.append("physical outputs partially on (vo only)")
    else:
        physical_parts.append("physical outputs off")

    # Add ro and vo values only when they are set
    ro_vo_parts = []
    if obj._roSet and hasattr(obj, "_ro"):
        ro_vo_parts.append(f"ro={obj._ro} kpc")
    if obj._voSet and hasattr(obj, "_vo"):
        ro_vo_parts.append(f"vo={obj._vo} km/s")

    return physical_parts[0] + (
        (", using " + " and ".join(ro_vo_parts)) if len(ro_vo_parts) > 0 else ""
    )


def _build_repr(obj, class_name=None):
    """
    Build a standard string representation for a potential object.

    Parameters
    ----------
    obj : object
        The potential object
    class_name : str, optional
        Class name to use (default: type(obj).__name__)

    Returns
    -------
    str
        String representation

    """
    if class_name is None:
        class_name = type(obj).__name__

    params = _build_params_string(obj)

    # Build components
    components = []

    # Add internal parameters if any
    if params:
        components.append(f"internal parameters: {', '.join(params)}")

    # Add physical output status
    physical_str = _build_physical_output_string(obj)
    if physical_str:
        components.append(physical_str)

    # Combine everything
    return f"{class_name} with {' and '.join(components)}"


def _strip_physical_output_info(repr_string):
    """
    Strip physical output information from a representation string.

    This is used to avoid duplication when a potential is nested within another
    (e.g., in WrapperPotentials or conversion classes).

    Parameters
    ----------
    repr_string : str
        The representation string to process

    Returns
    -------
    str
        The string with physical output information removed

    """
    return re.sub(
        r" and physical outputs (fully on|partially on \([^)]+\)|off)(, using .+)?$",
        "",
        repr_string,
    )
