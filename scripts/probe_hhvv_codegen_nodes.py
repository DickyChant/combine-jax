"""Probe value-only source generation for representative actual HHVV nodes."""

import json
from pathlib import Path
import re
import resource
import time
import traceback

from combine_jax.root_io import load_caching_nll, open_workspace
from hhvv_setup import configure_hhvv


def main():
    started = time.perf_counter()
    root = load_caching_nll()
    assert root.gSystem.Load("libRooFitCodegen") >= 0
    assert root.gInterpreter.Declare('#include "RooFit/CodegenContext.h"')
    manifest = json.loads(Path("build/gpu_exports/manifest.json").read_text())
    report = dict(status="running", root_version=str(root.gROOT.GetVersion()),
                  combine_version=manifest["combine_version"], source=manifest["models"]["hhvv"]["source"],
                  scope="CodegenContext.buildFunction on actual workspace nodes; source-generation coverage only, no JIT execution, AD or NLL equivalence claim",
                  cases=[])
    output = Path("reports/hhvv_codegen_nodes.json")
    code_dir = Path("build/clad_node_sources")
    code_dir.mkdir(exist_ok=True)

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    with open_workspace("inputs/workspace_MX2000.root") as workspace:
        configure_hhvv(workspace)
        objects = {str(n.GetName()): n for collection in (workspace.allFunctions(), workspace.allPdfs()) for n in collection}
        for cls in ("AsymPow", "ProcessNormalization", "FastVerticalInterpHistPdf2", "FastVerticalInterpHistPdf2D2",
                    "RooParametricHist2D", "RooParametricStepFunction", "CMSHistFunc", "CMSHistErrorPropagator",
                    "RooFormulaVar", "RooDoubleCB", "RooErfExpPdf"):
            node = next((n for n in objects.values() if n.ClassName() == cls), None)
            case = dict(requested_class=cls)
            if node is None:
                case["status"] = "not_found_in_workspace_collections"
            else:
                case["name"] = str(node.GetName())
                start = time.perf_counter()
                messages_path = code_dir/(cls+".messages.log")
                messages_path.write_text("")
                messages = root.RooMsgService.instance()
                messages.saveState()
                for i in range(messages.numStreams()):
                    messages.setStreamStatus(i, False)
                stream = messages.addStream(root.RooFit.ERROR, root.RooFit.OutputFile(str(messages_path)))
                try:
                    context = root.RooFit.Experimental.CodegenContext()
                    function_name = str(context.buildFunction(node))
                    code = str(context.collectedCode())
                    if not code.strip():
                        raise ValueError("Codegen returned no collected C++ source")
                    artifact = code_dir/(cls+".cxx")
                    artifact.write_text(code)
                    case.update(status="source_generated", function_name=function_name,
                                source_file=str(artifact), source_bytes=len(code.encode()))
                except Exception:
                    case.update(status="source_generation_failed", exception=traceback.format_exc())
                finally:
                    messages.deleteStream(stream)
                    messages.restoreState()
                    case["messages"] = messages_path.read_text()
                    unsupported = re.findall(r'Translate function for class "([^"]+)" has not yet been implemented', case["messages"])
                    if unsupported:
                        case.update(status="unsupported_translation", unsupported_classes=unsupported,
                                    source_discarded=True,
                                    reason="ROOT emits an error but can still return source containing a 1.0 placeholder; generated text is not evidence of support")
                case["seconds"] = time.perf_counter()-start
            report["cases"].append(case)
            save()
            print(cls, case["status"], case.get("messages", ""), case.get("exception", ""), flush=True)
    report["status"] = "probe_complete"
    save()


if __name__ == "__main__":
    main()
