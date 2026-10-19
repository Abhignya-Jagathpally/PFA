"""Run DeepGO-SE's official predict.py (bio-ontology-research-group/deepgo2) unchanged.

dgl is imported at module level by deepgo/models.py (GATConv, used only by DeepGOGATModel) and deepgo/data.py
(PPI graph loading); predict.py uses neither, and dgl has no wheel for the installed torch, so both imports are
stubbed. torch >= 2.6 defaults torch.load to weights_only=True, which the fair-esm checkpoints do not support;
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 when calling this.

Usage: python run_deepgose.py DEEPGO2_REPO -- <predict.py arguments>
"""
import runpy
import sys
import types

repo = sys.argv[1]
dgl = types.ModuleType("dgl")
dgl.nn = types.ModuleType("dgl.nn")
dgl.nn.GATConv = None
sys.modules["dgl"], sys.modules["dgl.nn"] = dgl, dgl.nn
sys.path.insert(0, repo)
sys.argv = [f"{repo}/predict.py"] + sys.argv[sys.argv.index("--") + 1:]
runpy.run_path(f"{repo}/predict.py", run_name="__main__")
