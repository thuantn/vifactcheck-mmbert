"""Leakage / label sanity checks on ViFactCheck. python data/sanity.py data"""
import sys, os, json
from collections import Counter
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoConfig
d = os.path.join(sys.argv[1], "vifactcheck", "data")
T = {f.split("-")[0]: pq.read_table(os.path.join(d, f)).to_pydict() for f in os.listdir(d)}
tr, te = T["train"], T["test"]
ut, ue = set(tr["Url"]), set(te["Url"])
print("articles: train", len(ut), "test", len(ue), "test articles also in train:", len(ue & ut),
      "test claims on such articles:", sum(u in ut for u in te["Url"]))
for name, t in T.items():
    ev_empty = Counter((l, not (e or "").strip()) for l, e in zip(t["labels"], t["Evidence"]))
    found = Counter((l, (e or "")[:60] in (c or "")) for l, e, c in zip(t["labels"], t["Evidence"], t["Context"]) if (e or "").strip())
    print(name, "empty evidence by label", sorted(ev_empty.items()), "| evidence found in article by label", sorted(found.items()))
print("label by Author (train):", {a: dict(Counter(l for l, b in zip(tr["labels"], tr["Author"]) if b == a)) for a in set(tr["Author"])})
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"); cfg = AutoConfig.from_pretrained("jhu-clsp/mmBERT-base")
print("sep", repr(tok.sep_token), "cls", repr(tok.cls_token), "pooling", getattr(cfg, "classifier_pooling", None), "layers", cfg.num_hidden_layers, "vocab", cfg.vocab_size)
print("pair example:", tok.decode(tok("claim text", "article text")["input_ids"]))
