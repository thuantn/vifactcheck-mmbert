# Pre-registration of a follow-up: full article vs BM25 retrieval after longer training (28/11/2026, before any run)

## Why, and what it is not
The first experiment (docs/PREREG.md) found H3 failing: reading 2,048 tokens did not beat 512 tokens of BM25-selected
sentences (test dF1 -0.78, 95% CI [-2.74, +1.19]). On its dev set the 2,048 arm was still improving at its last epoch
(mean macro F1 by epoch 79.48 / 83.37 / 84.57) while BM25 had levelled off (79.88 / 83.74 / 83.88). This follow-up asks
the question again with 6 epochs. **It is not an independent confirmation:** same dataset, and the question was chosen
after seeing those results. A reranker was considered and dropped: on dev, BM25-512 already keeps the start of the gold
evidence for 97.0% of the claims that have it verbatim.

## H4
Trained for 6 epochs, mmBERT-base reading the article up to 2,048 tokens (**A**) beats the same model reading 512 tokens
of BM25-selected sentences (**E**).

## Data (`data/make_cv.py`, ids in `data/split_cv.json`)
5-fold article-disjoint cross-validation over the articles of the old article-disjoint **train** split (5,148 claims),
which were never used for scoring. Fold k: test = fold k; of the other four folds' articles 1/8 are the validation set
(best epoch); the rest, plus the old dev rows (used for the recipe choice, so they only ever train), are training data.
The old test split is not used at all.

## Runs
A and E, each fold, seeds 0 and 1 (20 runs; `jobs/06_cv_s0.txt` then `jobs/07_cv_s1.txt`). 6 epochs; everything else as
in the first experiment (LR 2e-5, cls pooling, batch 4 x 4, warmup 6%, linear decay over the 6 epochs, fp16, test
predictions at the best validation epoch). No result is analysed before all 20 runs are in. If any run has a non-finite
loss after warmup, skips more than 1% of its steps after warmup, or stops on non-finite logits, all 20 runs are redone
in fp32 and this is noted. The fp16 counters (only those) are checked after the first jobs file, before the second.

## Decision rule (`analysis_cv.py`)
Units: dF1 = macro F1(A) - macro F1(E) on a fold's test for one seed (10 units). Point estimate: their mean. Interval:
the **envelope** (lowest lower and highest upper bound) of (a) the 95% percentile bootstrap, 10,000 draws, resampling
test articles within each fold, of the pooled out-of-fold dF1 averaged over the 2 seeds, and (b) the 95% t interval
(4 degrees of freedom) of the 5 fold means (the two seeds of a fold are averaged: they share its data, so they are not
independent; fixed after the code review, before any run).
- **Full article wins** if mean >= +1.5, at least 4 of 5 folds (seed-averaged) are positive and the lower bound > 0.
- **Retrieval is enough** if the upper bound < +1.5.
- Otherwise **inconclusive**.

## Stated now
The expected half-width is about 2 F1 points (a rough estimate), so "full article wins" needs a difference of about +2,
"retrieval is enough" one of about -0.5 or lower, and anything between will most likely come out inconclusive. The t
interval treats the 5 folds as independent although they share training data. The recipe was chosen without the BM25 arm, and for 2 epochs; both arms get the same one. Training sets differ
slightly in size from the first experiment, so F1 levels are not comparable with it.
Expected cost: about 16 T4-hours.
