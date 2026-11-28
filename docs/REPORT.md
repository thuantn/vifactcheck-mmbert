# Report: does reading the whole article help a Vietnamese fact-checker? (28/11/2026)

Short write-up of two pre-registered experiments on ViFactCheck with mmBERT-base. Details and numbers: README.md,
docs/PREREG.md (experiment 1), docs/PREREG_cv.md (follow-up), results/analysis*.txt.

## Question
80% of ViFactCheck articles exceed 512 tokens, and for about a quarter of the claims the gold evidence starts after
token 512. mmBERT reads up to 8,192 tokens. Does reading 2,048 tokens beat the usual 512?

## Findings
1. **Yes, against the first 512 tokens.** Article-disjoint test, 5 seeds: 82.54 vs 75.41 macro F1 (+7.13, 95% CI
   [+5.10, +9.24]); the gain sits on claims whose evidence lies beyond token 512 (H1, H2 hold).
2. **No, against 512 tokens of BM25-selected sentences.** 82.54 vs 83.32 (-0.78, CI [-2.74, +1.19]; H3 fails).
   BM25-512 trains in about a third of the time (15 vs 43 min per run on a T4).
3. **Not an under-training artefact.** With 6 epochs and 5-fold article-disjoint cross-validation (2 seeds, 20 runs),
   A - E = -0.56 (CI [-1.45, +0.34]), 0 of 5 folds favour A: by the registered rule, retrieval is enough.
4. Reference points (official split, leaky across articles): claim + gold evidence 87.2, claim only 52.0, XLM-R-large 74.3.

## What this does and does not show
- The long context itself is useful: a model that cannot see the evidence loses 7 points. But a cheap retriever
  recovers all of it, so a long-context encoder is not needed for this task.
- One dataset, one model family. The follow-up was chosen after seeing experiment 1, so it is a replication, not an
  independent confirmation. Inference time and memory were not measured.
- The upper bound on any A advantage is about +0.3 to +1.2 F1, below the +1.5 margin registered as "worth it".

## Cost (Colab T4, account 1)
About 93 CU bought, 55.31 left: roughly 38 CU in total. Follow-up (20 runs at 6 epochs) about 19 CU; the rest went to
the smoke test, recipe grid, main runs, references and the XLM-R learning-rate check. About 2 CU were lost to two runner
bugs (a poll that never ended, a lost launch), fixed and recorded in the commit history.

## Process notes
Decision rules were committed before each run, all seeds are counted, and the fp16 gate (no non-finite loss, under 1%
skipped steps after warmup) held in every run. Two slips were disclosed: a commit made before a stale test was fixed (the
code was right, the test assumed t(9)), and an edit to the runner while it ran.
