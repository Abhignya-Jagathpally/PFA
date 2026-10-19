"""ProteInfer GO (random-split CNN ensemble elements) on a FASTA file -> TSV (protein, GO id, confidence >=
threshold). MODEL_DIR may be a comma-separated list; confidences are then averaged over the elements as in
get_preds_at_or_above_threshold (proteinfer.py allows at most 5 elements, default 1).

Same computation as inference.Inferrer / get_preds_at_or_above_threshold of google-research/proteinfer:
one-hot residues from utils.residues_to_one_hot, the SavedModel's serving signature (inputs `sequence`,
`sequence_length`), sigmoid confidences over `label_vocab:0`. tensorflow_hub 0.16 no longer has the TF1
hub.Module used by in_graph_inferrer, so the SavedModel is loaded with tf.compat.v1.saved_model.loader.

Usage: tfenv/bin/python run_proteinfer.py MODEL_DIR PROTEINFER_REPO IN.fasta OUT.tsv [threshold]
"""
import sys
import time
import types

import numpy as np

model_dir, repo, fasta, out = sys.argv[1:5]
thr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01

import tensorflow.compat.v1 as tf  # noqa: E402

tf.disable_eager_execution()
contrib = types.ModuleType("tensorflow.contrib")
contrib.lookup = types.ModuleType("tensorflow.contrib.lookup")
sys.modules["tensorflow.contrib"] = contrib
sys.modules["tensorflow.contrib.lookup"] = contrib.lookup
sys.path.insert(0, repo)
import utils  # noqa: E402  (proteinfer utils)

ids, seqs = [], []
for line in open(fasta):
    line = line.strip()
    if line.startswith(">"):
        ids.append(line[1:].split()[0]); seqs.append("")
    else:
        seqs[-1] += line

models = []  # (session, input, length, output); several dirs (comma-separated) = ensemble mean, as the CLI
for d in model_dir.split(","):
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=4))
        mg = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], d)
        sig = mg.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        assert set(sig.inputs) == {"sequence", "sequence_length"} and len(sig.outputs) == 1, sig
        v = sess.run("label_vocab:0").astype(str)
        if models:
            assert (v == vocab).all(), "ensemble elements disagree on label_vocab"
        vocab = v
        models.append((sess, sig.inputs["sequence"].name, sig.inputs["sequence_length"].name,
                       list(sig.outputs.values())[0].name))
print(f"{len(ids)} sequences, {len(vocab)} labels, {len(models)} ensemble elements", flush=True)

order = np.argsort([-len(s) for s in seqs])
t0 = time.time()
n_rows = 0
with open(out, "w") as f:
    i = 0
    while i < len(order):
        L = len(seqs[order[i]])
        bs = max(1, min(64, 64000 // max(L, 1)))
        b = order[i:i + bs]
        oh = [utils.residues_to_one_hot(seqs[j]) for j in b]
        X = np.zeros((len(b), max(len(o) for o in oh), 20), np.float32)
        for k, o in enumerate(oh):
            X[k, :len(o)] = o
        lens = np.array([len(o) for o in oh], np.int32)
        P = np.mean([s.run(y_t, {x_t: X, l_t: lens}) for s, x_t, l_t, y_t in models], axis=0)
        for k, j in enumerate(b):
            for c in np.where(P[k] >= thr)[0]:
                if vocab[c].startswith("GO:"):
                    f.write(f"{ids[j]}\t{vocab[c]}\t{P[k, c]:.4f}\n"); n_rows += 1
        i += len(b)
        if (i // len(b)) % 20 == 0:
            print(f"{i}/{len(order)} done, {time.time() - t0:.0f}s", flush=True)
print(f"wrote {n_rows} rows in {time.time() - t0:.0f}s", flush=True)
