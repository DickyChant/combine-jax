"""Inspection and validation entry points."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import sys

from .graph import inspect_workspace
from .root_io import open_workspace, load_root


def main():
    parser = argparse.ArgumentParser(prog="combine-jax")
    argv = sys.argv[1:]
    if '-M' in argv:
        index = argv.index('-M')
        if index+1 >= len(argv) or argv[index+1] != 'AsymptoticLimits':
            parser.error('Implemented method dispatch: -M AsymptoticLimits; other methods await native validation')
        argv = ['limit', *argv[:index], *argv[index+2:]]
    commands = parser.add_subparsers(dest="command", required=True)
    from .methods.cli import add_limit_command, run_limit
    add_limit_command(commands)
    from .runtime.options import add_limits_command
    add_limits_command(commands)
    inspect = commands.add_parser("inspect", help="Inspect a workspace without fitting or modifying it")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--workspace", default="w")
    inspect.add_argument("--data", default="data_obs")
    inspect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'limit':
        return run_limit(args)
    if args.command == 'limits':
        from .runtime.controller import run_limits
        return run_limits(args)
    if args.command == "inspect":
        start = time.perf_counter()
        root = load_root()
        with open_workspace(args.path, args.workspace) as workspace:
            report = inspect_workspace(workspace, data=args.data)
        report.update(source=str(args.path), root_version=str(root.gROOT.GetVersion()),
                      combine_version=version("higgs-combine-tool"),
                      elapsed_seconds=time.perf_counter() - start)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"{report['node_count']} nodes, {len(report['channels'])} channels; {args.output}")


if __name__ == "__main__":
    main()
