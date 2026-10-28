"""Verify benchmark producers against current files or an explicit Git commit."""
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = 'src/combine_jax/tensor'


def verified_source_record(report, producer, revision=None):
    """A historical revision changes provenance lookup, never numerical gates."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args])

    commit = None
    if revision is None:
        paths = sorted(str(p.relative_to(ROOT))
                       for p in (ROOT/SOURCE_DIRECTORY).glob('*.py'))
        def contents(path):
            return (ROOT/path).read_bytes()
    else:
        commit = git('rev-parse', '--verify', '--end-of-options',
                     revision+'^{commit}').decode().strip()
        candidates = git('ls-tree', '-r', '--name-only', commit, '--',
                         SOURCE_DIRECTORY).decode().splitlines()
        paths = sorted(p for p in candidates if Path(p).parent.as_posix() == SOURCE_DIRECTORY
                       and p.endswith('.py'))
        def contents(path):
            return git('cat-file', 'blob', commit+':'+path)
    assert paths and set(paths) == set(report['sources']), 'Producer source file set differs'
    expected = {**report['sources'], producer: report['script_sha256'],
                'scripts/tensor_benchmark_common.py': report['common_script_sha256']}
    for path, digest in expected.items():
        assert hashlib.sha256(contents(path)).hexdigest() == digest, \
            f'Producer source hash mismatch: {path}'
    return dict(source_commit=commit, lookup='git_commit' if commit else 'current_files',
                producer=producer, verified_sha256=expected)
