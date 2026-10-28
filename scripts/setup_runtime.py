"""Build a relocatable worker runtime from this checkout and public dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

IMPORTER = 'git+https://github.com/DickyChant/combine_pypi.git@4d7101c7318394f7da6e849a642bf72c77558972'
VIEW = '/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--native-cmssw', type=Path, required=True)
    parser.add_argument('--native-version', choices=('10.3.3',), default='10.3.3',
                        help='Declared native release; its library hash is verified on workers')
    parser.add_argument('--refresh-code', action='store_true', help='Reuse installed dependencies')
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    source = Path(__file__).resolve().parents[1]
    if not (args.native_cmssw/'src/HiggsAnalysis/CombinedLimit').is_dir():
        parser.error('--native-cmssw must contain an installed Combine v10.3.3')
    if not args.refresh_code and prefix.exists():
        parser.error('Prefix already exists; choose a new directory or --refresh-code')
    prefix.mkdir(parents=True, exist_ok=True)
    clean = {k: os.environ[k] for k in ('HOME','USER','TMPDIR','LANG') if k in os.environ}
    clean.update(PATH='/usr/bin:/bin', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                 CMAKE_BUILD_PARALLEL_LEVEL='4', CXX='/usr/bin/g++', CC='/usr/bin/gcc')
    venv = prefix/'venv-import'
    python = str(venv/'bin/python')
    if not args.refresh_code:
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], env=clean, check=True)
        clean['PATH'] = str(venv/'bin')+':/usr/bin:/bin'
        subprocess.run([python, '-m', 'pip', 'install', '--upgrade', 'pip', 'cmake>=3.26',
                        'ninja', 'jax==0.12.0', 'iminuit==2.32.0', 'uproot>=5,<6', IMPORTER],
                       env=clean, check=True)
    if (prefix/'code').exists():
        shutil.rmtree(prefix/'code')
    shutil.copytree(source/'src', prefix/'code/src',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git','-C',str(source),'status','--porcelain'], text=True))
    libraries = list((args.native_cmssw/'lib').glob('*/libHiggsAnalysisCombinedLimit.so'))
    if len(libraries) != 1:
        raise RuntimeError('Expected exactly one native Combine library')
    manifest = dict(schema=1, source_repository='https://github.com/DickyChant/combine-jax',
        revision=revision, source_dirty=dirty, importer_source=IMPORTER,
        native_cmssw=str(args.native_cmssw.resolve()),
        native_version=args.native_version,
        native_library_sha256=hashlib.sha256(libraries[0].read_bytes()).hexdigest(),
        lcg_view=os.environ.get('COMBINE_JAX_LCG_VIEW',VIEW),
        files={str(p.relative_to(prefix)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((prefix/'code/src').rglob('*')) if p.is_file()})
    (prefix/'runtime.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (prefix/'setup.sh').write_text('export DHI_COMBINE_JAX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
                                   'export PATH="$DHI_COMBINE_JAX/bin:$PATH"\n')
    (prefix/'bin').mkdir(exist_ok=True)
    launcher = prefix/'bin/combine-jax'
    launcher.write_text('#!/usr/bin/env bash\nset -eo pipefail\n'
        'prefix="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"\n'
        'export DHI_COMBINE_JAX="$prefix"\n'
        'view="${COMBINE_JAX_LCG_VIEW:-'+manifest['lcg_view']+'}"\n'
        'exec bash "$prefix/code/src/combine_jax/runtime/run_lcg.sh" "$prefix" "$view" -m combine_jax.cli "$@"\n')
    launcher.chmod(0o755)
    # Probe the installed source build without inheriting LCG ROOT paths.
    clean['PATH'] = str(venv/'bin')+':/usr/bin:/bin'
    subprocess.run([python,'-c',
        'import ROOT,jax,cms_combine; from importlib.metadata import version; '
        'assert ROOT.gROOT.GetVersion()=="6.40.00"; '
        'assert version("higgs-combine-tool")=="11.0.0"; '
        'print("Importer verified",ROOT.gROOT.GetVersion(),jax.__version__)'], env=clean, check=True)
    archive = prefix/'runtime.tar.gz'
    with tarfile.open(archive.with_suffix('.tmp'),'w:gz',compresslevel=1) as output:
        for name in ('code','bin','setup.sh','runtime.json','venv-import'):
            output.add(prefix/name,arcname=name,filter=lambda info: None if
                '__pycache__' in info.name or info.name.endswith('.pyc') else info)
    archive.with_suffix('.tmp').replace(archive)
    checksum = hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    (prefix/'runtime.sha256').write_text(checksum+'\n')
    print('Runtime ready:',prefix,'archive SHA256:',checksum,flush=True)


if __name__ == '__main__':
    main()
