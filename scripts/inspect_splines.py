"""Record HHVV spline algorithms and domains before/after benchmark mass setup."""

from collections import Counter
import json
from pathlib import Path

from hhvv_setup import configure_hhvv
from combine_jax.graph import walk
from combine_jax.root_io import load_root, open_workspace


def main():
    root = load_root()
    report = {"source": "inputs/workspace_MX2000.root", "nodes": []}
    with open_workspace(report["source"]) as workspace:
        mx = workspace.var("MX")
        report["saved_MX"] = {"value": float(mx.getVal()), "constant": bool(mx.isConstant())}
        splines = [n for n in walk(workspace.pdf("model_s")).values() if n.ClassName() == "RooSpline1D"]
        for node in splines:
            knots = json.loads(str(root.TBufferJSON.ConvertToJSON(node, 3, "x_")))
            algorithm = json.loads(str(root.TBufferJSON.ConvertToJSON(node, 3, "type_")))
            servers = tuple(node.servers())
            if len(servers) != 1:
                raise ValueError(f"{node.GetName()}: expected one spline argument")
            argument = servers[0]
            value = float(argument.getVal())
            report["nodes"].append(dict(name=str(node.GetName()), algorithm=algorithm, knots=len(knots),
                                        domain=[knots[0], knots[-1]], argument=str(argument.GetName()),
                                        saved_argument=value, saved_in_domain=knots[0] <= value <= knots[-1]))
        report["configured"] = configure_hhvv(workspace)
        for node, entry in zip(splines, report["nodes"], strict=True):
            value = float(tuple(node.servers())[0].getVal())
            entry.update(configured_argument=value,
                         configured_in_domain=entry["domain"][0] <= value <= entry["domain"][1])
    report["algorithms"] = dict(Counter(n["algorithm"] for n in report["nodes"]))
    report["saved_outside_domain"] = sum(not n["saved_in_domain"] for n in report["nodes"])
    report["configured_outside_domain"] = sum(not n["configured_in_domain"] for n in report["nodes"])
    Path("reports/spline_inventory.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k: v for k, v in report.items() if k != "nodes"})


if __name__ == "__main__":
    main()
