"""Where the gold evidence sits in the article (mmBERT tokens before its first character). python data/evidence_pos.py data"""
import sys, os
import pyarrow.parquet as pq
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
d = os.path.join(sys.argv[1], "vifactcheck", "data")
for f in sorted(os.listdir(d)):
    t = pq.read_table(os.path.join(d, f)).to_pydict()
    pos, miss = [], 0
    for ctx, ev in zip(t["Context"], t["Evidence"]):
        ctx, ev = ctx or "", (ev or "").strip()
        j = ctx.find(ev[:60]) if ev else -1
        if j < 0:
            miss += 1; continue
        pos.append(len(tok(ctx[:j], add_special_tokens=False)["input_ids"]))
    n = len(pos); pos.sort()
    print(f"{f:30s} found {n}/{n + miss}  evidence starts after token 512: {sum(p > 512 for p in pos) / n:.1%}  after 448: {sum(p > 448 for p in pos) / n:.1%}  median start {pos[n // 2]}")
