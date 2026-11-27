# Pre-registration: does reading the whole article help a Vietnamese fact-checker? (27/11/2026, before any training)

Revised the same day after two review rounds (each by two models given the measurements and the plan): the decision
moved to an article-disjoint split, the arms share one recipe, the subgroup is computed from the real truncation, and a
retrieve-then-read arm was added. Nothing has been trained yet.

## Question
ViFactCheck (Hoa et al., AAAI 2025, [arXiv:2412.15308](https://arxiv.org/abs/2412.15308); HF `tranthaihoa/vifactcheck`,
MIT) asks whether a Vietnamese news claim is Supported, Refuted or NEI given the article. The paper's encoders read at
most 512 tokens (XLM-R) or 256 (PhoBERT): full-article macro F1 XLM-R-large 75.42, XLM-R-base 65.40; a fine-tuned Gemma
reaches 85.94. mmBERT-base (307M, ModernBERT architecture, pre-trained up to 8,192 tokens) can read whole articles.
- **H1:** mmBERT-base reading up to 2,048 tokens beats the same model reading the first 512.
- **H2 (mechanism):** the gain comes from claims whose gold evidence lies beyond what 512 tokens show.
- **H3 (is length needed?):** reading 2,048 tokens beats reading 512 tokens of BM25-selected sentences.

## Facts measured before writing this (`data/fetch_stats.py`, `data/evidence_pos.py`, `data/sanity.py`)
- Official splits train 5,062 / dev 723 / test 1,447 claims; labels balanced (test 508 / 468 / 471).
- Articles (mmBERT tokens): median ~795, p90 ~1,470; 80% > 512, ~2% > 2,048. Claims: median 45, max 249.
- The gold evidence is found verbatim (its first 60 characters) in the article for 88% of claims, for all three labels
  (NEI rows also carry an evidence sentence). Of those, 22.8% of test claims have it start after article token 512.
- **The official split is by claim, not article:** 753 of 758 test articles also appear in train. A model can learn an
  article's content in training and reuse it on another claim about the same article, and a 2,048-token model sees
  more of each article in training. Labels are not skewed by outlet.
- Pair encoding is `<bos> claim <eos> article <eos>` (3 special tokens).

## Splits
- **Article-disjoint split (decides H1-H3):** all 7,232 claims grouped by article URL; articles shuffled with seed 0
  and cut 70 / 10 / 20 into train / dev / test (`data/make_split.py`, ids committed in `data/split_article.json`).
- **Official split (comparison with the paper only).**

## Runs (Colab T4, fp16 autocast, one code path; `train.py`)
Input `tokenizer(claim, second, truncation="only_second", max_length=L)`: the claim is kept whole, the second text's
tail is cut, the same way in training and test. AdamW (wd 0.01), linear decay, warmup 6%, batch 4 x 4 accumulation,
grad-norm clip 1.0, 3 epochs, the epoch with the best dev macro F1 is kept. **Recipe choice, shared by A, B and E:** on the
article-disjoint dev set, seed 0, 2 epochs, LR {2e-5, 5e-5} x pooling {mean, cls}, each config run at L = 2,048 and
L = 512; the config with the best mean of the two dev F1s is used for every mmBERT arm. No test set is looked at.
Article-disjoint split, seeds 0-4:
- **A**: mmBERT-base, article, L = 2,048.  **B**: same, L = 512 (first 512 tokens).
- **E**: same, L = 512, the article reduced to its BM25 top sentences against the claim (sentences split on . ! ? and
  newlines; filled in score order up to the token budget, then put back in article order).
- Eval only: A's adapters tested at L = 512 and 1,024 (dose-response), reported.
Official split, references, reported and not decided on: A and B seeds 0-2; **D** XLM-R-large L = 512 seeds 0-2 (LR from
{1e-5, 2e-5} on the official dev set); **C** mmBERT-base on claim + gold evidence, L = 1,024, seeds 0-2 (paper XLM-R-large
88.02); **Q** claim only, seed 0; majority class.
fp16 check, on the first grid run at L = 2,048 (before any other run): metrics.json counts non-finite losses and steps
GradScaler skipped, separately after warmup; eval stops on non-finite logits. If any loss is non-finite after warmup,
or more than 1% of the steps after warmup are skipped, every arm runs in fp32 and this is noted.

Clarifications (27/11/2026, after a pre-launch code review, before any run): the first grid run is its own jobs file
(`jobs/01a_fp16.txt`); the 1% is of `steps - warmup_steps`; non-finite eval logits in any grid run also count as an
fp16 failure, and all 8 grid runs must finish at the same precision. A run's dev F1 is that of its best epoch; a tie
between configs goes to the lower LR, then to mean pooling.

Recipe grid result (27/11/2026, `results/grid/`, `python recipe.py`; no test set read). fp16 rule passed in all 8 runs
(no non-finite loss; at most 2 of 604 steps after warmup skipped). Dev macro F1 (best epoch, always the second) at
L = 2,048 / 512 / mean: 2e-5 mean 73.22 / 75.80 / 74.51; 2e-5 cls 81.91 / 76.96 / 79.44; 5e-5 mean 81.15 / 75.81 / 78.48;
5e-5 cls 83.12 / 75.52 / 79.32. **Chosen: LR 2e-5, cls pooling**, ahead of 5e-5 cls by 0.12 on one seed: the rule
decides, the two are not distinguishable. Every run was still improving at epoch 2.

Clarifications before the main runs (27/11/2026, no test number exists; `analysis.py` is written and reviewed now):
runs are `jobs/02_main.txt` (A, B, E, ordered by seed), `jobs/03_refs.txt` (mmBERT references), `jobs/04_dgrid.txt`
(D's LR) and then D's three seeds. D's LR: seed 0,
3 epochs, official dev, best-epoch macro F1, tie to 1e-5. C and Q use the mmBERT recipe; Q reads up to 512; majority =
the commonest training label. CIs are bootstrap percentiles (H1/H3 two-sided 95%: 2.5th; H2 one-sided 95%: 5th), with
one article resample shared by all seeds in a draw. H2's evidence length is that of the stripped string, and "after B's
last visible character" is start >= that character's end offset. If any main run hits a non-finite value, every arm of
its jobs file is rerun in fp32 and this is noted.

D's LR grid result (28/11/2026, `results/dgrid/`, official dev, no test read for D): 1e-5 gave dev macro F1
39.36 / 60.35 / 69.52 per epoch (still rising); 2e-5 stayed at 17.43 in all three epochs (the level of predicting one
class), with no non-finite loss. Chosen: 1e-5. D is likely under-trained at 3 epochs.

## Decision rules (article-disjoint test split)
Per seed i: dF1_i = macro F1(A_i) - macro F1(B_i). CI = 95% two-level bootstrap, 10,000 draws: resample the 5 seed
indices with replacement and the test articles with replacement (keeping their claims), then average dF1 over the drawn
seeds. Pairing A_i with B_i by seed index is arbitrary but fixed before running.
- **H1 holds** if mean dF1 >= +1.5, dF1_i > 0 for at least 4 of 5 seeds, and the CI lower bound > 0.
  **H1 fails** if mean dF1 <= +0.5 or dF1_i > 0 for at most 2 seeds. Otherwise **inconclusive**.
- **H2** (read only if H1 holds). With B's real truncation (tokenizer offsets of claim + article at L = 512): S = claims
  whose evidence starts after B's last visible article character and ends inside A's window; R = evidence entirely
  inside B's window; claims straddling the cut, past A's window, or with evidence not found verbatim are in neither.
  Evidence location: start = first occurrence in the article of the evidence's first 60 characters (stripped); end =
  start + the evidence's length. Not found = that 60-character prefix does not occur (12% of claims).
  Accuracy (not F1: subgroups can lack a class) per group. H2 holds if dAcc(S) >= +5.0 and the one-sided CI lower bound
  of dAcc(S) - dAcc(R) > 0. Per-class F1 inside S and R reported.
- **H3:** same rules as H1 with E in place of B. H3 holding means length helps beyond picking the right sentences;
  H1 holding and H3 failing means retrieval to 512 is enough.
Expected size (stated now, for scale only): S is ~20-27% of claims; if A gains +15-20 points on S and nothing on R,
overall accuracy rises ~3-5 points; macro F1 moves similarly when classes stay balanced.

## Caveats stated now
The article-disjoint split differs from the paper's, so its numbers are not comparable with Table 3; the official-split
references are. Comparisons with the paper also mix tokenizers and unknown recipes, hence D. 3 epochs may under-train
the 2,048 arm (dev F1 per epoch is logged). One dataset, one model family.
