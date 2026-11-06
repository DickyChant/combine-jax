"""The packaged launcher must isolate ROOT import from the CUDA environment."""
import ast
import os
from pathlib import Path
import subprocess


def test_generated_launcher_selects_isolated_importer(tmp_path):
    source=Path(__file__).parents[1]/'scripts/setup_runtime.py'
    tree=ast.parse(source.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='write_text' and isinstance(n.func.value,ast.Name) and n.func.value.id=='launcher']
    assert len(calls)==1
    text=eval(compile(ast.Expression(calls[0].args[0]),str(source),'eval'),
        {'manifest':{'lcg_view':'/chosen/cuda/view/setup.sh'}})
    (tmp_path/'bin').mkdir();(tmp_path/'venv-import/bin').mkdir(parents=True)
    launcher=tmp_path/'bin/combine-jax';launcher.write_text(text);launcher.chmod(0o755)
    python=tmp_path/'venv-import/bin/python'
    python.write_text('#!/usr/bin/env bash\nset -eu\n'
        'test -z "${ROOTSYS:-}${LD_LIBRARY_PATH:-}${PYTHONHOME:-}"\n'
        'test "$JAX_PLATFORMS" = cpu\n'
        'test "$JAX_ENABLE_X64" = true\n'
        'test "$1" = -m && test "$2" = combine_jax.cli && test "$3" = import\n'
        'test "$PYTHONPATH" = "$DHI_COMBINE_JAX/code/src"\n')
    python.chmod(0o755)
    env={**os.environ,'ROOTSYS':'/wrong/root','PYTHONHOME':'/wrong/python',
        'PYTHONPATH':'/wrong/root/modules','LD_LIBRARY_PATH':'/wrong/root/lib'}
    result=subprocess.run([str(launcher),'import','input.root','--data','toy_asimov'],
        env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
