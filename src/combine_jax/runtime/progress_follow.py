"""Follow the fit subprocess's progress file and render it in this terminal."""
import json

from ..progress_display import ProgressDisplay


class ProgressFollower:
    """Read whole JSON lines appended to ``path`` since the last poll."""

    def __init__(self, path, **display_options):
        self.path, self.offset, self.partial = path, 0, ''
        self.display = ProgressDisplay(**display_options)

    def poll(self):
        try:
            with open(self.path) as stream:
                stream.seek(self.offset)
                text = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return
        lines = (self.partial+text).split('\n')
        self.partial = lines.pop()
        for line in lines:
            if line:
                self.display.update(json.loads(line))

    def close(self):
        self.poll()
        self.display.close()
