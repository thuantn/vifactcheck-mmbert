"""Recipe choice (docs/PREREG.md) from the 8 dev-only grid runs. python recipe.py [results/grid]
fp16 rule on every run; score of a config = mean of its best-epoch dev macro F1 at L=2048 and L=512;
tie -> lower LR, then mean pooling."""
import json, sys

D = sys.argv[1] if len(sys.argv) > 1 else "results/grid"
scores, precisions, failed = {}, set(), False
for lr in ("2e-5", "5e-5"):
    for pool in ("mean", "cls"):
        f1 = []
        for L in (2048, 512):
            m = json.load(open(f"{D}/grid_L{L}_lr{lr}_{pool}.json"))
            a = m["args"]
            assert (a["model"], a["input"], a["select"]) == ("jhu-clsp/mmBERT-base", "article", "")
            assert (a["bs"], a["accum"], a["warmup"], a["pilot_steps"]) == (4, 4, 0.06, 0)
            assert (a["max_len"], a["lr"], a["pooling"]) == (L, float(lr), pool) and a["split"] == "article"
            assert a["epochs"] == len(m["dev"]) == 2 and a["seed"] == 0 and a["no_test"] and not m["test"] and not a["limit"]
            precisions.add(a["fp32"])
            if not a["fp32"]:
                post = m["steps"] - m["warmup_steps"]
                bad = m["nan_losses_after_warmup"] > 0 or m["skipped_after_warmup"] > 0.01 * post
                failed |= bad
                print(f"L={L} lr={lr} {pool}: nan {m['nan_losses_after_warmup']}, skipped {m['skipped_after_warmup']}/{post}"
                      + ("  <-- fails the fp16 rule" if bad else ""))
            f1.append(max(d["f1"] for d in m["dev"]))
        scores[lr, pool] = (sum(f1) / 2, f1)
assert len(precisions) == 1, "grid runs mix fp16 and fp32"
# a run with non-finite eval logits writes no metrics.json: its file is missing here, which is the same failure
if failed:
    sys.exit("fp16 rule failed: rerun all 8 grid runs with --fp32 (no recipe from these numbers)")
for (lr, pool), (s, f1) in scores.items():
    print(f"lr={lr} pooling={pool}: dev F1 2048 {f1[0]:.2f}  512 {f1[1]:.2f}  mean {s:.3f}")
lr, pool = min(scores, key=lambda k: (-round(scores[k][0], 6), float(k[0]), k[1] != "mean"))
print(f"recipe: --lr {lr} --pooling {pool}")
