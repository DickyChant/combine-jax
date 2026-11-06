"""Terminal progress for fits: convergence bar, capped-iteration bar, batch lanes.

Fed with records from ``combine_jax.tensor.progress``. One record describes one
fit, or a batch of fits when its fields are lists (one entry per lane).

The convergence bar is the fraction of the way, in log(EDM), from the first
EDM seen to the tolerance. It fills to the best EDM reached so far, so it does
not jitter; the current EDM is printed next to it. It stops at 99% while the
estimate has passed but the guard has not ("validating"), and reaches 100%
only when the fit reports convergence. "stalled" means the best EDM has not
improved over the last ``stall_updates`` updates. The iteration bar is shown
only for capped fits; its time is an upper bound (time to the cap).

In a terminal the lines are redrawn in place. Otherwise (batch logs) one plain
snapshot is printed every ``plain_interval`` seconds and one line per finished fit.
"""
import math
import sys
import time

BLOCKS = ' ▏▎▍▌▋▊▉█'
UNCAPPED = 10**7
STATUS = {0: 'converged', 1: 'iteration limit', 2: 'line search failed',
          3: 'non-finite', 4: 'invalid start', 5: 'EDM validation failed'}


def bar(fraction, width=12):
    fraction = min(max(fraction, 0.), 1.)
    eighths = int(round(fraction*width*8))
    full, part = divmod(eighths, 8)
    return ('█'*full+(BLOCKS[part] if part else '')).ljust(width, '░')[:width]


def duration(seconds):
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    return f'{h}:{rest//60:02d}:{rest%60:02d}' if h else f'{rest//60}:{rest%60:02d}'


class Fit:
    """Host-side history of one fit (one lane)."""

    def __init__(self, record, now, stall_updates):
        self.started = now
        self.stall_updates = stall_updates
        self.first = None
        self.best = math.inf
        self.since_best = 0
        self.previous_value = None
        self.record = None
        self.rate = None
        self.update(record, now)

    def measure(self, r):
        return r['edm'] if r['edm_criterion'] else r['max_gradient']

    def update(self, r, now):
        if self.record is not None:
            self.previous_value = self.record['value']
            if r['iteration'] > self.record['iteration'] and now > self.updated:
                rate = (r['iteration']-self.record['iteration'])/(now-self.updated)
                self.rate = rate if self.rate is None else .7*self.rate+.3*rate
        self.record, self.updated = r, now
        m = self.measure(r)
        if math.isfinite(m) and m > 0:
            if self.first is None:
                self.first = m
            if m < self.best*(1-1e-9):
                self.best, self.since_best = m, 0
            else:
                self.since_best += 1
        elif m == 0:
            self.best = 0.

    @property
    def done(self):
        return self.record['final'] or not self.record['active']

    @property
    def converged(self):
        r = self.record
        return r['status'] == 0 if r['final'] else (not r['active'] and not r['validating'] and self.best < r['tolerance'])

    @property
    def stalled(self):
        return not self.done and self.since_best >= self.stall_updates

    def fraction(self):
        r, tol = self.record, self.record['tolerance']
        if self.converged:
            return 1.
        if self.best < tol or r['validating']:
            return .99
        if self.first is None or self.first <= tol:
            return 0.
        return min(.99, max(0., math.log(self.first/self.best)/math.log(self.first/tol)))

    def flags(self):
        r = self.record
        if r['final']:
            return '✓' if r['status'] == 0 else '✗ '+STATUS.get(r['status'], str(r['status']))
        out = []
        if r['validating']:
            out.append('validating')
        if self.stalled:
            out.append('stalled')
        return ' '.join(out)

    def line(self, name, now, compact=False):
        r = self.record
        capped = r['budget'] < UNCAPPED
        label = 'EDM' if r['edm_criterion'] else '|g|'
        parts = [name]
        if capped:
            parts.append(f"it {bar(r['iteration']/max(r['budget'], 1), 8)} {r['iteration']}/{r['budget']}")
        else:
            parts.append(f"it {r['iteration']}")
        parts.append(f"converge {bar(self.fraction())} {100*self.fraction():3.0f}%")
        parts.append(f"{label} {self.measure(r):.1e}→{r['tolerance']:.0e}")
        if not compact:
            delta = '' if self.previous_value is None else f" (Δ {r['value']-self.previous_value:+.1e})"
            parts.append(f"NLL {r['value']:.4f}{delta}")
            if self.rate and not self.done:
                parts.append(f'{self.rate:.1f} it/s')
                if capped:
                    parts.append('≤'+duration((r['budget']-r['iteration'])/self.rate))
            parts.append(duration(now-self.started))
        if r['repairs']:
            parts.append(f"repairs {r['repairs']}")
        flags = self.flags()
        if flags:
            parts.append(flags)
        return '  '.join(parts)


def add_progress_options(parser):
    """Non-native flags shared by every method."""
    parser.add_argument('--show-progress', nargs='?', const='summary', choices=('summary', 'lanes'),
                        help='Non-native: live fit progress (convergence bar; iteration bar when capped). '
                             '"lanes" shows one line per fit of a batch')
    parser.add_argument('--progress-every', type=int, default=10,
                        help='Non-native: iterations between progress updates (with --show-progress)')


def progress_every(args):
    """Iterations between updates requested on the command line; 0 means off."""
    if not getattr(args, 'show_progress', None):
        return 0
    if args.progress_every < 1:
        raise ValueError('--progress-every must be positive')
    return args.progress_every


class ProgressDisplay:
    """Render progress records. Call ``update(record)``; ``close()`` at the end.

    ``lanes`` shows one line per lane of a batched fit instead of a summary.
    ``total`` is the number of fits expected (for example scan points), if known.
    """

    def __init__(self, stream=None, *, mode='auto', lanes=False, total=None,
                 plain_interval=30., redraw_interval=.1, stall_updates=20, clock=time.monotonic):
        self.stream = stream or sys.stderr
        if mode == 'auto':
            mode = 'tty' if getattr(self.stream, 'isatty', lambda: False)() else 'plain'
        if mode not in ('tty', 'plain'):
            raise ValueError('mode must be auto, tty or plain')
        self.mode, self.lanes, self.total = mode, lanes, total
        self.plain_interval, self.redraw_interval = plain_interval, redraw_interval
        self.stall_updates, self.clock = stall_updates, clock
        self.fits, self.finished, self.drawn = [], 0, 0
        self.task = None
        self.last_draw = self.last_plain = -math.inf

    def update(self, record):
        now = self.clock()
        batched = isinstance(record['iteration'], list)
        width = len(record['iteration']) if batched else 1
        rows = ([{n: (v[i] if isinstance(v, list) else v) for n, v in record.items()} for i in range(width)]
                if batched else [record])
        task = record.get('task', 'fit')
        new = (task != self.task or len(self.fits) != width
               or any(r['start'] or r['iteration'] < f.record['iteration'] for r, f in zip(rows, self.fits)))
        if new:
            self._finish(now)
            self.task = task
            self.fits = [Fit(r, now, self.stall_updates) for r in rows]
        else:
            for fit, r in zip(self.fits, rows):
                fit.update(r, now)
        if all(f.done for f in self.fits) and rows[0]['final']:
            self._finish(now)
        elif self.mode == 'tty' and now-self.last_draw >= self.redraw_interval:
            self._draw(self._lines(now))
            self.last_draw = now
        elif self.mode == 'plain' and now-self.last_plain >= self.plain_interval:
            self._print(self._lines(now))
            self.last_plain = now

    def _header(self):
        if self.total:
            return f'fits {bar(self.finished/self.total, 16)} {self.finished}/{self.total}'
        return None

    def _lines(self, now, header=True):
        lines = []
        header = self._header() if header else None
        if len(self.fits) == 1:
            lines.append(self.fits[0].line(self.task, now))
        elif self.lanes:
            lines.append(f'{self.task}  batch of {len(self.fits)}')
            lines.extend(fit.line(f'  lane {i}', now, compact=True) for i, fit in enumerate(self.fits))
        else:
            done = sum(f.done for f in self.fits)
            running = [f for f in self.fits if not f.done]
            # While running: the lane furthest from converging. At the end:
            # the lane that took the most iterations (it held the batch).
            slowest = (min(running, key=lambda f: f.fraction()) if running
                       else max(self.fits, key=lambda f: f.record['iteration']))
            iterations = sorted(f.record['iteration'] for f in self.fits)
            lines.append(f'{self.task}  done {done}/{len(self.fits)}  median it {iterations[len(iterations)//2]}'
                         f'  slowest lane {self.fits.index(slowest)}: '+slowest.line('', now, compact=True).strip())
        return ([header] if header else [])+lines

    def _finish(self, now):
        if not self.fits:
            return
        self.finished += len(self.fits)
        # The live header (fits k/N) is redrawn with the next fit, not kept.
        lines = self._lines(now, header=False)
        if self.total:
            lines[0] = f'[{self.finished}/{self.total}] '+lines[0]
        if self.mode == 'tty':
            self._draw(lines)
            # Keep the finished fit on screen; the next fit draws below it.
            self.drawn = 0
        else:
            self._print(lines)
        self.fits = []

    def _draw(self, lines):
        out = self.stream
        if self.drawn:
            out.write(f'\x1b[{self.drawn}F')
        for line in lines:
            out.write('\x1b[2K'+line+'\n')
        for _ in range(self.drawn-len(lines)):
            out.write('\x1b[2K\n')
        extra = max(self.drawn-len(lines), 0)
        if extra:
            out.write(f'\x1b[{extra}F')
        self.drawn = len(lines)
        out.flush()

    def _print(self, lines):
        for line in lines:
            self.stream.write(line+'\n')
        self.stream.flush()

    def close(self):
        self._finish(self.clock())
