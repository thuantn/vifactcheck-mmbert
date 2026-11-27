"""Download ViFactCheck (MIT, HF tranthaihoa/vifactcheck) to data/ and print token-length stats with the mmBERT tokenizer.
python data/fetch_stats.py <out_dir>"""
import sys, os, statistics as st
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import pyarrow.parquet as pq
out = sys.argv[1]
d = snapshot_download("tranthaihoa/vifactcheck", repo_type="dataset", local_dir=os.path.join(out, "vifactcheck"))
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
for f in sorted(os.listdir(os.path.join(d, "data"))):
    df = pq.read_table(os.path.join(d, "data", f)).to_pydict()
    n = lambda col: [len(tok(x or "")["input_ids"]) for x in df[col]]
    for col in ("Statement", "Evidence", "Context"):
        L = sorted(n(col))
        q = lambda p: L[min(len(L) - 1, int(p * len(L)))]
        print(f"{f:40s} {col:9s} n={len(L)} median {q(.5)} p90 {q(.9)} max {L[-1]} >512: {sum(x > 512 for x in L) / len(L):.1%} >2048: {sum(x > 2048 for x in L) / len(L):.1%}")
    from collections import Counter; print(f"{f:40s} labels {sorted(Counter(df['labels']).items())}")
