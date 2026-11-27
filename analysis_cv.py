"""Decision rule of docs/PREREG_cv.md. python analysis_cv.py [results/cv]
Units: dF1 = macro F1(A) - macro F1(E) on each fold's test, for each seed (10 units). Interval: the envelope of
(a) a 95% bootstrap over test articles, resampled within folds, of the seed-mean pooled out-of-fold dF1, and
(b) the 95% t interval of the 5 fold means (each the average of its 2 seeds, which share the fold: not independent)."""
import json, sys

import numpy as np

from analysis import by_article, cells, f1_conf
from train import load_split

DRAWS, FOLDS, SEEDS, T4 = 10000, 5, 2, 2.776
WANT = {"model": "jhu-clsp/mmBERT-base", "input": "article", "pooling": "cls", "lr": 2e-5, "epochs": 6, "bs": 4,
        "accum": 4, "warmup": 0.06, "limit": 0, "pilot_steps": 0, "no_test": False}
ARMS = {"cvA": {"max_len": 2048, "select": ""}, "cvE": {"max_len": 512, "select": "bm25"}}


def load_run(D, arm, k, s, y):
    name = f"{arm}_f{k}_s{s}"
    m = json.load(open(f"{D}/{name}/metrics.json"))
    want = {**WANT, **ARMS[arm], "split": f"cv{k}", "seed": s}
    bad = {key: (m["args"][key], v) for key, v in want.items() if m["args"][key] != v}
    assert not bad and len(m["dev"]) == 6, (name, "not the registered run", bad)
    assert m["args"]["fp32"] or (m["nan_losses_after_warmup"] == 0 and m["skipped_after_warmup"] <= 0.01 * (m["steps"] - m["warmup_steps"])), (
        name, "fails the fp16 rule: redo all 20 runs in fp32")
    L = ARMS[arm]["max_len"]
    rows = [json.loads(line) for line in open(f"{D}/{name}/test_L{L}.jsonl", encoding="utf-8")]
    assert [r["i"] for r in rows] == list(range(len(y))) and [r["y"] for r in rows] == list(y), name
    p = np.array([r["p"] for r in rows]).argmax(1)
    assert abs(f1_conf(cells(y, p).sum(0)) - m["test"][str(L)]["f1"]) < 0.2, (name, "preds disagree with metrics.json")
    return p, m


def decide(units, boot):
    """units: (folds, seeds) dF1. boot: bootstrap draws of the pooled statistic. -> mean, (lo, hi), verdict"""
    u = units.mean(1)  # one value per fold
    half = T4 * u.std(ddof=1) / np.sqrt(len(u))
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    lo, hi = min(blo, u.mean() - half), max(bhi, u.mean() + half)
    folds_up = int((u > 0).sum())
    if u.mean() >= 1.5 and folds_up >= 4 and lo > 0:
        v = "FULL ARTICLE WINS"
    elif hi < 1.5:
        v = "RETRIEVAL IS ENOUGH"
    else:
        v = "INCONCLUSIVE"
    return u.mean(), (lo, hi), (blo, bhi), folds_up, v


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "results/cv"
    rng = np.random.default_rng(0)
    units = np.zeros((FOLDS, SEEDS))
    ys, preds, boot_w, arts = [], {(a, s): [] for a in ARMS for s in range(SEEDS)}, [], []
    metrics, fp = {a: [] for a in ARMS}, set()
    for k in range(FOLDS):
        te = load_split("test", f"cv{k}")
        y = np.array(te["labels"])
        ids = {u: i for i, u in enumerate(sorted(set(te["Url"])))}
        art = np.array([ids[u] for u in te["Url"]])
        n = len(ids)
        boot_w.append(rng.multinomial(n, np.full(n, 1 / n), size=DRAWS).astype(float))  # within the fold
        ys.append(y); arts.append(art)
        for s in range(SEEDS):
            f = {}
            for a in ARMS:
                p, m = load_run(D, a, k, s, y)
                preds[a, s].append(p); metrics[a].append(m); fp.add(m["args"]["fp32"])
                f[a] = f1_conf(cells(y, p).sum(0))
            units[k, s] = f["cvA"] - f["cvE"]
    assert len(fp) == 1, "runs mix fp16 and fp32"
    boot = np.zeros(DRAWS)
    for s in range(SEEDS):
        conf = {a: sum(w @ by_article(art, cells(y, preds[a, s][k]))
                       for k, (y, art, w) in enumerate(zip(ys, arts, boot_w))) for a in ARMS}
        boot += (f1_conf(conf["cvA"]) - f1_conf(conf["cvE"])) / SEEDS
    yall = np.concatenate(ys)
    print(f"pooled out-of-fold: {len(yall)} claims, {sum(a.max() + 1 for a in arts)} articles")
    for a in ARMS:
        f = [f1_conf(cells(yall, np.concatenate(preds[a, s])).sum(0)) for s in range(SEEDS)]
        curve = np.mean([[d["f1"] for d in m["dev"]] for m in metrics[a]], 0)
        L = str(ARMS[a]["max_len"])
        print(f"{a}: pooled macro F1 per seed {[round(float(x), 2) for x in f]}\n   mean val F1 by epoch {np.round(curve, 2).tolist()}"
              f"\n   best epochs {[m['test'][L]['epoch'] for m in metrics[a]]}")
        print("   fp16 after warmup: " + ", ".join(
            f"{m['nan_losses_after_warmup']} nan {m['skipped_after_warmup']}/{m['steps'] - m['warmup_steps']}"
            + (" fp32" if m["args"]["fp32"] else "") for m in metrics[a]))
    mean, (lo, hi), (blo, bhi), up, v = decide(units, boot)
    print(f"\ndF1 (A - E) per fold and seed: {np.round(units, 2).tolist()}")
    print(f"mean {mean:+.2f}  folds > 0: {up}/5  article bootstrap [{blo:+.2f}, {bhi:+.2f}]  envelope with t(4) [{lo:+.2f}, {hi:+.2f}]  -> {v}")


if __name__ == "__main__":
    main()
