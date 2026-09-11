"""Own a short-lived ROS observation node without masking its failure."""

from contextlib import contextmanager
import logging


@contextmanager
def observation_node(runtime, factory, *args, **kwargs):
    """Initialize, collect through, and clean up an observation-only node.

    ``try_shutdown`` tolerates a context already stopped by a signal handler.
    Both cleanup steps run even if destruction fails. A collection/construction
    exception remains primary; otherwise cleanup failures propagate normally.
    Runtime injection keeps this lifecycle testable without importing ROS.
    """

    runtime.init(args=None)
    node = None
    primary_error = None
    try:
        node = factory(*args, **kwargs)
        yield node
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors = []
        for cleanup in (
            *((node.destroy_node,) if node is not None else ()),
            runtime.try_shutdown,
        ):
            try:
                cleanup()
            except Exception as error:
                cleanup_errors.append(error)
                if primary_error is not None:
                    logging.getLogger(__name__).error(
                        "ROS observation cleanup failed while preserving %s: %s",
                        type(primary_error).__name__, primary_error, exc_info=True,
                    )
        if cleanup_errors and primary_error is None:
            raise cleanup_errors[0]
