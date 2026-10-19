"""Run SPROF-GO's official script/predict.py (biomed-AI/SPROF-GO) unchanged apart from the documented
ProtTrans_path setting. transformers 5 removed PreTrainedTokenizer.batch_encode_plus; it is mapped to
__call__, which returns the same input_ids / attention_mask for a list of strings.

Usage: python run_sprof.py SPROF_REPO -- <predict.py arguments>
"""
import runpy
import sys

from transformers import T5Tokenizer

repo = sys.argv[1]
if not hasattr(T5Tokenizer, "batch_encode_plus"):
    T5Tokenizer.batch_encode_plus = lambda self, *a, **k: self(*a, **k)
sys.path.insert(0, f"{repo}/script")
sys.argv = [f"{repo}/script/predict.py"] + sys.argv[sys.argv.index("--") + 1:]
runpy.run_path(f"{repo}/script/predict.py", run_name="__main__")
