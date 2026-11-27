"""Article-disjoint split of ViFactCheck (docs/PREREG.md): group all claims by article URL, shuffle the articles with
seed 0, cut 70/10/20. Writes data/split_article.json = {"train"|"dev"|"test": [[official_split, row], ...]}.
python data/make_split.py"""
import json, os, random
from collections import defaultdict

import pyarrow.parquet as pq

D = "data/vifactcheck/data"
by_url = defaultdict(list)
for f in sorted(x for x in os.listdir(D) if x.endswith(".parquet")):  # one file per split: train-, dev-, test-
    name = f.split("-")[0]
    for row, url in enumerate(pq.read_table(os.path.join(D, f), columns=["Url"]).column("Url").to_pylist()):
        by_url[url].append([name, row])
urls = sorted(by_url)
random.Random(0).shuffle(urls)
a, b = int(0.7 * len(urls)), int(0.8 * len(urls))
split = {k: [r for u in us for r in by_url[u]] for k, us in (("train", urls[:a]), ("dev", urls[a:b]), ("test", urls[b:]))}
assert not ({u for u in urls[:a]} & {u for u in urls[b:]})
json.dump(split, open("data/split_article.json", "w"), separators=(",", ":"))
print({k: len(v) for k, v in split.items()}, "articles", len(urls))
