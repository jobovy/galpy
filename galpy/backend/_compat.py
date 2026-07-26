###############################################################################
#   galpy.backend._compat: object-level backend compatibility.
#
#   ONE question, asked the same way of every kind of galpy object: are this
#   object's compute methods backend-aware (jax/torch), or do they only handle
#   numpy? Potentials, forces, wrappers, composites and lists thereof answer it
#   recursively (a container is only as backend-aware as its members); every
#   other galpy object -- a df, and in the future an actionAngle or an Orbit --
#   answers it with the same ``_backend_compatible`` flag its ``__init__`` sets,
#   exactly as ``hasC`` is set for the C layer.
#
#   Lives here rather than in galpy.potential because it is not a potential
#   question: the coercion boundary in ``_input`` asks it of dfs too, and the
#   potential-specific recursion is just how potentials happen to answer it.
###############################################################################


def is_backend_compatible(obj):
    """
    Check whether a galpy object's compute methods are backend-aware (jax/torch).

    Gates the coordinate coercion at galpy's backend entry points (see
    ``galpy.backend.backend_input``): only an object that can do arithmetic on
    backend arrays may be handed them.

    Answered recursively for potentials, mirroring ``_check_c``: a list iff every
    member is; a composite through its component list; a wrapper iff it is itself
    backend-aware AND the potential it wraps is. Any other galpy object (a df, an
    actionAngle, an Orbit) reads its own ``_backend_compatible`` flag, which
    defaults to False and is set in ``__init__`` by whatever has been migrated.

    Parameters
    ----------
    obj : object
        Any galpy object: a Potential/Force/planarForce/linearPotential, a list
        of them (possibly nested), a combined potential formed using addition
        (pot1+pot2+…), a wrapper, a df instance, ...

    Returns
    -------
    bool
        True iff the object's compute methods are backend-aware.

    Notes
    -----
    - 2026-06-15 - Written as potential._check_backend_compatible - Bovy (UofT)
    - 2026-07-24 - Generalized to all galpy objects and moved to galpy.backend - Bovy (UofT)

    """
    # Deferred: galpy.backend is imported before galpy.potential, so this cannot
    # be a module-level import. Reached only off the numpy path, so the numpy hot
    # path pays neither the import nor the flatten.
    from ..potential import flatten
    from ..potential.baseCompositePotential import baseCompositePotential
    from ..potential.WrapperPotential import (
        WrapperPotential,
        parentWrapperPotential,
        planarWrapperPotential,
    )

    obj = flatten(obj)
    if isinstance(obj, list):
        return all(is_backend_compatible(o) for o in obj)
    elif isinstance(obj, baseCompositePotential):
        # A (planar/linear)CompositePotential delegates to its members through
        # the no-decorator internal path (no per-member coercion), so coercion
        # must fire at the OUTER boundary; it is backend-compatible iff every
        # component is (mirrors the list branch). flatten leaves a composite
        # as-is, so this must precede the generic Force branch (a composite IS
        # a Force).
        return is_backend_compatible(obj._potlist)
    elif isinstance(
        obj, (parentWrapperPotential, WrapperPotential, planarWrapperPotential)
    ):
        # A wrapper modulates what it wraps, so both must be backend-aware.
        return bool(
            getattr(obj, "_backend_compatible", False)
            and is_backend_compatible(obj._pot)
        )
    # Leaf: a Force/planarForce/linearPotential, or any other galpy object that
    # opts in with the same flag (galpy.df.sphericaldf does). Anything without
    # the flag -- a plain object, an unmigrated class -- is False.
    return bool(getattr(obj, "_backend_compatible", False))
