"""Run DeepFRI's official predict.py (flatironinstitute/DeepFRI) unchanged, sequence-only path (--fasta_fn ->
the DeepCNN-MERGED models of trained_models/model_config.json). deepfrier.utils.load_FASTA opens files with mode
'rU', which Python >= 3.11 rejects; the 'U' flag is dropped (it only meant universal newlines). Run from the
DeepFRI directory with TF_USE_LEGACY_KERAS=1 (the .hdf5 models are Keras 2 files).

Usage: python run_deepfri.py DEEPFRI_REPO -- <predict.py arguments>
"""
import builtins
import runpy
import sys

repo = sys.argv[1]
sys.path.insert(0, repo)
import deepfrier.utils as du  # noqa: E402

du.open = lambda f, mode="r", *a, **k: builtins.open(f, mode.replace("U", ""), *a, **k)
sys.argv = [f"{repo}/predict.py"] + sys.argv[sys.argv.index("--") + 1:]
runpy.run_path(f"{repo}/predict.py", run_name="__main__")
