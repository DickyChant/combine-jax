"""Throttled optimizer progress records sent from device loops to the host.

Fits stay inside one compiled while loop, so the host cannot see iterations
unless the loop sends them. ``emit`` is called by the optimizer with a small
dict of scalars. It sends nothing unless the record is due, and under ``vmap``
the whole batch is sent as one record of arrays (one callback per update, not
one per lane). Records go to the innermost ``sink`` on the host, or are
appended as JSON lines to ``$COMBINE_JAX_PROGRESS`` when that is set. The
current ``task`` name is attached on the host; it never enters the trace.
Progress is a diagnostic channel only and never changes a fit result.
"""
import contextlib
import json
import os
import threading


# Process-wide, not thread-local: XLA runs host callbacks on its own threads.
_sinks = []
_task = None
_file_lock = threading.Lock()


@contextlib.contextmanager
def sink(function):
    """Deliver records to ``function(record)`` inside this context."""
    _sinks.append(function)
    try:
        yield function
    finally:
        _sinks.remove(function)


@contextlib.contextmanager
def task(name, **fields):
    """Attach ``task=name`` and ``fields`` to records produced in this context."""
    global _task
    previous, _task = _task, dict(fields, task=name)
    try:
        yield
    finally:
        _task = previous


def _write_file(record):
    path = os.environ.get('COMBINE_JAX_PROGRESS')
    if not path:
        return
    with _file_lock, open(path, 'a') as stream:
        stream.write(json.dumps(record)+'\n')


def deliver(record):
    """Host side of ``emit``: convert to plain Python and pass to the sink."""
    import numpy as np
    record = {name: np.asarray(value).tolist() for name, value in record.items()}
    record.update(_task or {})
    (_sinks[-1] if _sinks else _write_file)(record)


def _make_emit():
    import jax
    import jax.numpy as jnp
    from jax.custom_batching import custom_vmap

    @custom_vmap
    def emit(due, record):
        jax.lax.cond(due, lambda: jax.debug.callback(deliver, record), lambda: None)
        return due

    @emit.def_vmap
    def emit_batched(axis_size, in_batched, due, record):
        # A batched predicate would turn lax.cond into a select and run the
        # callback every iteration. Send the whole batch if any lane is due.
        due_batched, record_batched = in_batched
        if not due_batched:
            due = jnp.broadcast_to(due, (axis_size,))
        record = {name: value if record_batched[name] else
                  jnp.broadcast_to(value, (axis_size,)+jnp.shape(value))
                  for name, value in record.items()}
        jax.lax.cond(jnp.any(due), lambda: jax.debug.callback(deliver, record), lambda: None)
        return due, True
    return emit


_emit = None


def emit(due, record):
    """Send ``record`` (a dict of scalars) to the host when ``due`` is true."""
    global _emit
    if _emit is None:
        _emit = _make_emit()
    return _emit(due, record)
