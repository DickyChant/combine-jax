"""Check v11 discrete penalties, active parameters and fixed channel masks."""

import json
from pathlib import Path

from build_multipdf_fixture import build_multipdf_fixture
from combine_jax.root_io import open_workspace
from combine_jax.validation import validate_workspace


def main():
    path=build_multipdf_fixture("build/multipdf_fixture")
    cases=[]
    for index,choice_mask,control_mask in [(0,0,0),(1,0,0),(1,1,0),(1,0,1),(1,1,1)]:
        with open_workspace(path) as workspace:
            workspace.cat("pdf_index").setIndex(index)
            workspace.var("mask_choice").setVal(choice_mask);workspace.var("mask_control").setVal(control_mask)
            for name,error in (("multi_slope",.005),("multi_c1",.1),("multi_c2",.1)):workspace.var(name).setError(error)
            report=validate_workspace(workspace,random_points=25,seed=7714)
        assert report["max_nll_absolute_error"]<1e-6
        assert report["max_channel_absolute_error"]<1e-6
        assert report["max_gradient_relative_error"]<1e-4
        assert report["max_hessian_scaled_error"]<1e-6
        cases.append(dict(index=index,choice_mask=choice_mask,control_mask=control_mask,validation=report))
        print(index,choice_mask,control_mask,{k:v for k,v in report.items() if k.startswith("max_")},flush=True)
    Path("reports/multipdf.json").write_text(json.dumps(dict(workspace=str(path),cases=cases),indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
