"""Decision rules of docs/PREREG.md over the finished runs. python analysis.py [results/main]
H1 (A vs B) and H3 (A vs E): per-seed dF1, 95% two-level bootstrap (seed indices and test articles, both with
replacement). H2: accuracy gain in S (evidence only A can see) and R (evidence B already sees), from B's real truncation.
Reference runs (official split) are reported, never decided on."""
import json, os, sys
from collections import Counter

import numpy as np

from train import load_split, macro_f1

DRAWS = 10000
BASE = {"model": "jhu-clsp/mmBERT-base", "input": "article", "max_len": 2048, "split": "article", "select": "",
        "pooling": "cls", "lr": 2e-5, "epochs": 3, "bs": 4, "accum": 4, "warmup": 0.06, "limit": 0, "pilot_steps": 0,
        "no_test": False}
ARMS = {  # run prefix: (seeds, what differs from BASE); D's LR is chosen on dev, so it is not pinned
    "A": (5, {}), "B": (5, {"max_len": 512}), "E": (5, {"max_len": 512, "select": "bm25"}),
    "offA": (3, {"split": "official"}), "offB": (3, {"split": "official", "max_len": 512}),
    "offC": (3, {"split": "official", "input": "evidence", "max_len": 1024}),
    "offQ": (1, {"split": "official", "input": "claim", "max_len": 512}),
    "offD": (3, {"split": "official", "model": "FacebookAI/xlm-roberta-large", "max_len": 512, "pooling": "", "lr": None}),
}


def cells(y, p):
    """(n, 9) one-hot of the confusion cell y*3+p."""
    return np.eye(9)[3 * y + p]


def f1_conf(c):
    """Macro F1 (x100) from confusion counts (..., 9); same definition as train.macro_f1."""
    c = c.reshape(*c.shape[:-1], 3, 3)
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    den = c.sum(-1) + c.sum(-2)
    return 100 * np.divide(2 * tp, den, out=np.zeros_like(tp), where=tp > 0).mean(-1)


def draws(art, n_seeds, rng):
    """art: article index per claim. -> article weights (DRAWS, K) and seed weights (DRAWS, n_seeds)."""
    k = art.max() + 1
    return (rng.multinomial(k, np.full(k, 1 / k), size=DRAWS).astype(float),
            rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=DRAWS).astype(float))


def by_article(art, x):
    """Sum the per-claim rows of x within each article."""
    out = np.zeros((art.max() + 1,) + x.shape[1:])
    np.add.at(out, art, x)
    return out


def compare_f1(y, art, pa, pb, rng):
    """pa, pb: (seeds, claims) predictions of the two arms, paired by seed index."""
    d = np.array([f1_conf(cells(y, a).sum(0)) - f1_conf(cells(y, b).sum(0)) for a, b in zip(pa, pb)])
    w, c = draws(art, len(d), rng)
    boot = np.stack([f1_conf(w @ by_article(art, cells(y, a))) - f1_conf(w @ by_article(art, cells(y, b)))
                     for a, b in zip(pa, pb)], 1)
    stat = (boot * c).sum(1) / len(d)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return d, lo, hi


def verdict(d, lo):
    if d.mean() >= 1.5 and (d > 0).sum() >= 4 and lo > 0:
        return "HOLDS"
    if d.mean() <= 0.5 or (d > 0).sum() <= 2:
        return "FAILS"
    return "INCONCLUSIVE"


def subgroup(claim, article, evidence, vis_a, vis_b):
    """S: evidence starts after B's last visible article char and ends inside A's window. R: ends inside B's window."""
    ev = (evidence or "").strip()
    j = (article or "").find(ev[:60]) if ev else -1
    if j < 0:
        return "not found"
    end = j + len(ev)
    if end <= vis_b:
        return "R"
    return "S" if j >= vis_b and end <= vis_a else "neither"


def visible_end(tok, claim, article, max_len):
    """Last article character the model sees, with train.py's truncation."""
    enc = tok(claim or "", article or "", truncation="only_second", max_length=max_len, return_offsets_mapping=True)
    return max((e for (_, e), s in zip(enc["offset_mapping"], enc.sequence_ids()) if s == 1), default=0)


def compare_acc(art, y, pa, pb, masks, rng):
    """Per group: mean over seeds of acc(A) - acc(B); bootstrap draws of it (same draws for every group)."""
    w, c = draws(art, len(pa), rng)
    point, boot = {}, {}
    for g, m in masks.items():
        diff = ((pa == y).astype(float) - (pb == y)) * m          # (seeds, claims)
        point[g] = 100 * diff.sum(1) / m.sum()
        den = w @ by_article(art, m.astype(float))
        per_seed = (w @ by_article(art, diff.T)) / den[:, None]   # (draws, seeds)
        boot[g] = 100 * (per_seed * c).sum(1) / len(pa)
    return point, boot


def load_run(D, arm, seed, L, y_true):
    name = f"{arm}_s{seed}"
    m = json.load(open(f"{D}/{name}/metrics.json"))
    want = {**BASE, **ARMS[arm][1], "seed": seed}
    bad = {k: (m["args"][k], v) for k, v in want.items() if v is not None and m["args"][k] != v}
    assert not bad, (name, "args differ from the registered arm", bad)
    rows = [json.loads(line) for line in open(f"{D}/{name}/test_L{L}.jsonl", encoding="utf-8")]
    assert [r["i"] for r in rows] == list(range(len(y_true))) and [r["y"] for r in rows] == list(y_true), name
    p = np.array([r["p"] for r in rows]).argmax(1)
    logged = m["test"][str(L)]["f1"]
    assert abs(f1_conf(cells(y_true, p).sum(0)) - logged) < 0.2, (name, "preds disagree with metrics.json", logged)
    return p, m


def mean_sd(v):
    v = np.asarray(v)
    return f"{v.mean():.2f}" + (f" +- {v.std(ddof=1):.2f}" if len(v) > 1 else "")


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else "results/main"
    te = load_split("test", "article")
    y = np.array(te["labels"])
    ids = {u: i for i, u in enumerate(sorted(set(te["Url"])))}
    art = np.array([ids[u] for u in te["Url"]])
    print(f"article-disjoint test: {len(y)} claims, {len(ids)} articles")
    preds, metrics = {}, {}
    for arm in ("A", "B", "E"):
        runs = [load_run(D, arm, s, 2048 if arm == "A" else 512, y) for s in range(5)]
        preds[arm] = np.stack([p for p, _ in runs])
        metrics[arm] = [m for _, m in runs]
    for arm, ms in metrics.items():
        f = [f1_conf(cells(y, p).sum(0)) for p in preds[arm]]
        print(f"{arm}: macro F1 {mean_sd(f)}  acc {mean_sd([100 * (p == y).mean() for p in preds[arm]])}"
              f"  per seed {[round(float(x), 2) for x in f]}  best epochs {[m['test'][str(m['args']['max_len'])]['epoch'] for m in ms]}")
    for L in (512, 1024):
        f = [f1_conf(cells(y, load_run(D, "A", s, L, y)[0]).sum(0)) for s in range(5)]
        print(f"A's adapters tested at L={L}: macro F1 {mean_sd(f)}")

    res = {}
    for h, other in (("H1", "B"), ("H3", "E")):
        d, lo, hi = compare_f1(y, art, preds["A"], preds[other], np.random.default_rng(0))
        res[h] = verdict(d, lo)
        print(f"\n{h} A - {other}: dF1 per seed {[round(float(x), 2) for x in d]}  mean {d.mean():+.2f}  "
              f"seeds > 0: {(d > 0).sum()}/5  95% CI [{lo:+.2f}, {hi:+.2f}]  -> {res[h]}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    group = np.array([subgroup(c, a, e, visible_end(tok, c, a, 2048), visible_end(tok, c, a, 512))
                      for c, a, e in zip(te["Statement"], te["Context"], te["Evidence"])])
    print("\nsubgroups:", dict(Counter(group.tolist())))
    masks = {g: group == g for g in ("S", "R")}
    point, boot = compare_acc(art, y, preds["A"], preds["B"], masks, np.random.default_rng(0))
    low = np.percentile(boot["S"] - boot["R"], 5)
    for g, m in masks.items():
        lo, hi = np.percentile(boot[g], [2.5, 97.5])
        print(f"dAcc({g}) {point[g].mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  n={m.sum()}  labels {np.bincount(y[m], minlength=3).tolist()}")
        for arm in ("A", "B"):
            per = []
            for c in range(3):
                per.append(round(float(np.mean([200 * ((p[m] == c) & (y[m] == c)).sum() / max(1, (p[m] == c).sum() + (y[m] == c).sum())
                                                for p in preds[arm]])), 2))
            print(f"   {arm} per-class F1 in {g} (Supported, Refuted, NEI as labelled 0,1,2): {per}")
    h2 = "HOLDS" if point["S"].mean() >= 5.0 and low > 0 else "DOES NOT HOLD"
    print(f"H2: dAcc(S) - dAcc(R) = {point['S'].mean() - point['R'].mean():+.2f}, one-sided 95% lower bound {low:+.2f}  -> "
          + (h2 if res["H1"] == "HOLDS" else f"not read (H1 {res['H1']}); numbers are descriptive"))

    print("\nofficial split (references, not decided on)")
    off = load_split("test", "official")
    yo = np.array(off["labels"])
    for arm in ("offA", "offB", "offC", "offQ", "offD"):
        runs = [load_run(D, arm, s, ARMS[arm][1]["max_len"] if "max_len" in ARMS[arm][1] else 2048, yo) for s in range(ARMS[arm][0])]
        print(f"{arm}: macro F1 {mean_sd([f1_conf(cells(yo, p).sum(0)) for p, _ in runs])}  "
              f"acc {mean_sd([100 * (p == yo).mean() for p, _ in runs])}  lr {sorted({m['args']['lr'] for _, m in runs})}")
        metrics[arm] = [m for _, m in runs]
    for split, yy in (("article", y), ("official", yo)):
        c = Counter(load_split("train", split)["labels"]).most_common(1)[0][0]
        print(f"majority class ({split}): macro F1 {macro_f1(yy.tolist(), [c] * len(yy)):.2f}  acc {100 * (yy == c).mean():.2f}")

    print("\nfp16 counters after warmup (non-finite losses, skipped steps / steps)")
    for arm, ms in metrics.items():
        print(f"{arm}: " + ", ".join(f"{m['nan_losses_after_warmup']} nan {m['skipped_after_warmup']}/{m['steps'] - m['warmup_steps']}"
                                     + (" fp32" if m["args"]["fp32"] else "") for m in ms))


if __name__ == "__main__":
    main()
