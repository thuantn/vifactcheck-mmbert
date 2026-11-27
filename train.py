"""Fine-tune an encoder on ViFactCheck (claim + article / evidence / nothing -> Supported, Refuted, NEI). See docs/PREREG.md.
python train.py --model jhu-clsp/mmBERT-base --max_len 2048 --lr 5e-5 --seed 0 --out runs/A_2048_lr5e-5_s0 [--eval_lens 512,1024]
Writes <out>/metrics.json (dev F1 per epoch, test F1 / acc per eval length) and <out>/test_L<len>.jsonl (per-claim preds)."""
import argparse, json, math, os, random, re, time
from collections import Counter

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

DATA = "data/vifactcheck/data"


def official(name):
    if not os.path.isdir(DATA):
        from huggingface_hub import snapshot_download
        snapshot_download("tranthaihoa/vifactcheck", repo_type="dataset", local_dir="data/vifactcheck")
    f = next(x for x in os.listdir(DATA) if x.startswith(name + "-"))
    return pq.read_table(os.path.join(DATA, f)).to_pydict()


def load_split(name, split):
    """official: the dataset's own split; article: data/split_article.json (rows of the official files, article-disjoint)."""
    if split == "official":
        return official(name)
    rows = json.load(open("data/split_article.json"))[name]
    src = {s: official(s) for s in {r[0] for r in rows}}
    return {k: [src[s][k][i] for s, i in rows] for k in src[rows[0][0]]}


SENT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"\w+")


def bm25_select(tok, claim, article, budget, k1=1.5, b=0.75):
    """The article's sentences most relevant to the claim (BM25 within the article), up to `budget` tokens, in article order."""
    sents = [s for s in SENT.split(article) if s.strip()]
    docs = [WORD.findall(s.lower()) for s in sents]
    if not docs:
        return article
    avg = sum(map(len, docs)) / len(docs)
    df = Counter(w for d in docs for w in set(d))
    idf = {w: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    q = WORD.findall(claim.lower())
    def bm25(d):
        tf = Counter(d)
        return sum(idf.get(w, 0) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in q if w in tf)
    keep, used = [], 0
    for j in sorted(range(len(sents)), key=lambda j: -bm25(docs[j])):
        n = len(tok(sents[j], add_special_tokens=False)["input_ids"]) + 1
        if used + n <= budget:
            keep.append(j); used += n
    if not keep:  # no sentence fits: the best one, cut by the tokenizer's truncation
        keep = [max(range(len(sents)), key=lambda j: bm25(docs[j]))]
    return " ".join(sents[j] for j in sorted(keep))


def encode(tok, t, inp, max_len, select=""):
    second = {"article": t["Context"], "evidence": t["Evidence"], "claim": None}[inp]
    claims = [c or "" for c in t["Statement"]]
    if second is None:
        enc = tok(claims, truncation=True, max_length=max_len)
    else:
        second = [s or "" for s in second]
        if select == "bm25":  # budget = max_len minus the claim and the 3 special tokens
            second = [bm25_select(tok, c, s, max_len - 3 - len(tok(c, add_special_tokens=False)["input_ids"])) for c, s in zip(claims, second)]
        enc = tok(claims, second, truncation="only_second", max_length=max_len)
    return [(ids, y, i) for i, (ids, y) in enumerate(zip(enc["input_ids"], t["labels"]))]


def batches(items, bs, shuffle, rng):
    """Length-grouped batches (sort within chunks of 50 batches): less padding, still random order when shuffling."""
    items = list(items)
    if shuffle:
        rng.shuffle(items)
    chunk = bs * 50
    out = []
    for a in range(0, len(items), chunk):
        part = sorted(items[a:a + chunk], key=lambda x: len(x[0]))
        out += [part[b:b + bs] for b in range(0, len(part), bs)]
    if shuffle:
        rng.shuffle(out)
    return out


def collate(batch, pad_id, device):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for r, (x, _, _) in enumerate(batch):
        ids[r, : len(x)], mask[r, : len(x)] = torch.tensor(x), 1
    y = torch.tensor([b[1] for b in batch])
    return ids.to(device), mask.to(device), y.to(device)


def macro_f1(y, p, k=3):
    f = []
    for c in range(k):
        tp = sum(a == c and b == c for a, b in zip(y, p))
        fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return 100 * sum(f) / k


@torch.no_grad()
def predict(model, items, bs, pad_id, device, amp):
    model.eval()
    res = {}
    for b in batches(items, bs, False, None):
        ids, mask, _ = collate(b, pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        if not torch.isfinite(logits).all():  # fp16 overflow at eval would silently predict class 0
            raise FloatingPointError("non-finite eval logits: rerun with --fp32")
        for (_, y, i), pr in zip(b, torch.softmax(logits, -1).tolist()):
            res[i] = (y, pr)
    model.train()
    return [res[i] for i in sorted(res)]


def score(preds):
    y = [a for a, _ in preds]
    p = [max(range(3), key=lambda c: pr[c]) for _, pr in preds]
    return {"f1": round(macro_f1(y, p), 2), "acc": round(100 * sum(a == b for a, b in zip(y, p)) / len(y), 2), "n": len(y)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--input", default="article", choices=["article", "evidence", "claim"])
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--split", default="article", choices=["article", "official"])
    ap.add_argument("--select", default="", choices=["", "bm25"], help="bm25: BM25-selected article sentences up to max_len")
    ap.add_argument("--pooling", default="", choices=["", "mean", "cls"], help="ModernBERT classifier pooling (default: the checkpoint's)")
    ap.add_argument("--eval_lens", default="", help="extra test-time max_len values, comma-separated (eval only)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--pilot_steps", type=int, default=0, help="stop after N optimizer steps (stability / speed check)")
    ap.add_argument("--no_test", action="store_true", help="dev only (LR choice)")
    ap.add_argument("--limit", type=int, default=0, help="claims per split (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda" and not a.fp32
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"classifier_pooling": a.pooling} if a.pooling else {}
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3, **kw).to(device)
    if a.max_len > 512:
        model.gradient_checkpointing_enable()
    tr, dv, te = (load_split(s, a.split) for s in ("train", "dev", "test"))
    if a.limit:
        tr, dv, te = ({k: v[: a.limit] for k, v in t.items()} for t in (tr, dv, te))
    train = encode(tok, tr, a.input, a.max_len, a.select)
    dev = encode(tok, dv, a.input, a.max_len, a.select)
    tests = {L: encode(tok, te, a.input, L, a.select) for L in [a.max_len] + [int(x) for x in a.eval_lens.split(",") if x]}

    rng = random.Random(a.seed)
    steps_per_epoch = math.ceil(len(train) / a.bs) // a.accum  # the trailing partial accumulation is dropped
    total = a.pilot_steps or steps_per_epoch * a.epochs
    warm = int(a.warmup * total)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, warm, total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # fp16 rule (docs/PREREG.md): counted after warmup separately; skipped = GradScaler skipped the step (overflow)
    log = {"args": vars(a), "dev": [], "test": {}, "steps": total, "warmup_steps": warm,
           "nan_losses": 0, "nan_losses_after_warmup": 0, "skipped_steps": 0, "skipped_after_warmup": 0}
    best, step, t0 = -1.0, 0, time.time()
    model.train()
    for ep in range(a.epochs):
        opt.zero_grad(set_to_none=True)  # a partial accumulation from the last epoch is dropped
        for bi, b in enumerate(batches(train, a.bs, True, rng)):
            ids, mask, y = collate(b, tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y) / a.accum
            if not torch.isfinite(loss):
                log["nan_losses"] += 1
                log["nan_losses_after_warmup"] += int(step >= warm)
            scaler.scale(loss).backward()
            if (bi + 1) % a.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = scaler.get_scale()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            skipped = int(scaler.get_scale() < scale)
            log["skipped_steps"] += skipped
            log["skipped_after_warmup"] += skipped * int(step >= warm)
            step += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item() * a.accum:.4f} {time.time() - t0:.0f}s", flush=True)
            if a.pilot_steps and step >= a.pilot_steps:
                break
        if a.pilot_steps and step >= a.pilot_steps:
            log["pilot"] = {"steps": step, "secs": round(time.time() - t0), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device == "cuda" else None}
            break
        opt.zero_grad(set_to_none=True)  # free the trailing partial accumulation before eval (T4 memory)
        print("fp16", {k: v for k, v in log.items() if k.startswith(("nan_", "skipped_"))}, flush=True)
        d = score(predict(model, dev, a.bs * 2, tok.pad_token_id, device, amp))
        d["epoch"] = ep
        log["dev"].append(d)
        print("dev", d, flush=True)
        if d["f1"] > best and not a.no_test:
            best = d["f1"]
            for L, items in tests.items():
                preds = predict(model, items, a.bs * 2, tok.pad_token_id, device, amp)
                log["test"][str(L)] = {**score(preds), "epoch": ep}
                with open(os.path.join(a.out, f"test_L{L}.jsonl"), "w", encoding="utf-8") as f:
                    for i, (yy, pr) in enumerate(preds):
                        f.write(json.dumps({"i": i, "y": yy, "p": [round(x, 5) for x in pr]}) + "\n")
            print("test", log["test"], flush=True)
    log["secs"] = round(time.time() - t0)
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
