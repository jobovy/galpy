###############################################################################
#   galpy.backend._jax.diffrax_adjoint: the diffrax DirectAdjoint galpy uses.
#
#   diffrax.DirectAdjoint runs the solve in equinox's bounded while loop, which
#   nests scans of 16 steps and checkpoints every level above the lowest. Once an
#   ADAPTIVE solve takes more than 16 steps, reverse mode through those
#   checkpointed levels is wrong (jax.grad off by up to tens of percent; forward
#   mode and RecursiveCheckpointAdjoint are right), on jax 0.10 and worse on 0.12.
#   Running the loop as one level of max_steps steps (no checkpointing) is exact.
###############################################################################
import functools


@functools.cache
def _single_level_direct_adjoint():
    import diffrax

    class SingleLevelDirectAdjoint(diffrax.DirectAdjoint):
        """``diffrax.DirectAdjoint`` with its step loop as a single scan of
        ``max_steps`` steps: reverse mode stores every step instead of
        recomputing checkpointed blocks."""

        @property
        def _loop(self):
            loop = super()._loop

            def single_level_loop(*, outer_while_loop, max_steps, **kwargs):
                if max_steps is not None:
                    outer_while_loop = functools.partial(
                        outer_while_loop, base=max(int(max_steps), 2)
                    )
                return loop(
                    outer_while_loop=outer_while_loop, max_steps=max_steps, **kwargs
                )

            return single_level_loop

    return SingleLevelDirectAdjoint


def direct_adjoint():
    """A ``diffrax.DirectAdjoint`` whose reverse-mode derivative is exact for
    adaptive solves (see the module header)."""
    return _single_level_direct_adjoint()()
