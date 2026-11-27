"""5-fold article-disjoint splits for the follow-up (docs/PREREG_cv.md). Test folds use only the articles of the old
article-disjoint TRAIN split (never used for scoring); the old dev rows only ever train; the old test is not used.
Writes data/split_cv.json = {"cv0".."cv4": {"train"|"dev"|"test": [[official_split, row], ...]}}.
python data/make_cv.py"""
import json, os, random
from collections import defaultdict

import pyarrow.parquet as pq

D = "data/vifactcheck/data"
url = {}
for f in sorted(x for x in os.listdir(D) if x.endswith(".parquet")):
    url[f.split("-")[0]] = pq.read_table(os.path.join(D, f), columns=["Url"]).column("Url").to_pylist()
old = json.load(open("data/split_article.json"))
by_url = defaultdict(list)
for s, i in old["train"]:
    by_url[url[s][i]].append([s, i])
urls = sorted(by_url)
random.Random(1).shuffle(urls)
out = {}
for k in range(5):
    test = urls[k::5]
    rest = [u for u in urls if u not in set(test)]
    random.Random(100 + k).shuffle(rest)
    n = len(rest) // 8
    out[f"cv{k}"] = {"train": [r for u in rest[n:] for r in by_url[u]] + old["dev"],
                     "dev": [r for u in rest[:n] for r in by_url[u]],
                     "test": [r for u in test for r in by_url[u]]}
arts = {k: {p: {url[s][i] for s, i in v[p]} for p in v} for k, v in out.items()}
for k, a in arts.items():
    assert not (a["train"] & a["dev"]) and not (a["train"] & a["test"]) and not (a["dev"] & a["test"]), k
    assert not ((a["train"] | a["dev"] | a["test"]) & {url[s][i] for s, i in old["test"]}), k
assert sorted(map(tuple, (r for v in out.values() for r in v["test"]))) == sorted(map(tuple, old["train"]))
json.dump(out, open("data/split_cv.json", "w", newline="\n"), separators=(",", ":"))
for k, v in out.items():
    print(k, {p: len(r) for p, r in v.items()}, "test articles", len(arts[k]["test"]))
