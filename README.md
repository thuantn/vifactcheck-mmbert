# Does reading the whole article help a Vietnamese fact-checker?

**Status (28/11/2026): both experiments finished.** The follow-up ([docs/PREREG_cv.md](docs/PREREG_cv.md)) also found no gain from reading the full article.
**Reading the whole article beats reading its first 512 tokens (+7.1 macro F1), but not reading 512 tokens of BM25-selected
sentences (-0.8).** Design and decision rules, written before the runs: [docs/PREREG.md](docs/PREREG.md).

[ViFactCheck](https://arxiv.org/abs/2412.15308) (AAAI 2025) asks whether a Vietnamese news claim is Supported, Refuted or
NEI given the article. Its encoder baselines read at most 512 tokens (XLM-R) or 256 (PhoBERT), but 80% of the articles
are longer than 512 tokens, and for about a quarter of the claims the gold evidence starts after token 512.
[mmBERT](https://arxiv.org/abs/2509.06888) (ModernBERT architecture, multilingual) reads up to 8,192 tokens. This repo
tests, with rules written before training:
- **H1:** mmBERT-base reading up to 2,048 tokens beats the same model reading the first 512;
- **H2:** the gain comes from claims whose evidence lies beyond token 512;
- **H3:** reading 2,048 tokens beats reading 512 tokens of BM25-selected sentences.

The decision uses an article-disjoint split built here (`data/split_article.json`): in the official split 753 of 758
test articles also appear in train, which would let a model learn articles instead of reading them. The official split
is used only to compare with the paper.

## Results
Article-disjoint test (1,420 claims, 209 articles), mmBERT-base, 5 seeds, macro F1. Full output of `analysis.py`:
[results/analysis.txt](results/analysis.txt).

| Arm | What it reads | Macro F1 | Train time per run (T4) |
|---|---|---:|---:|
| A | article, up to 2,048 tokens | 82.54 +- 1.80 | 43 min |
| B | article, first 512 tokens | 75.41 +- 0.35 | 15 min |
| E | BM25-selected sentences, 512 tokens | 83.32 +- 0.58 | 15 min |

| Question (registered rule) | Measured | Verdict |
|---|---|---|
| H1: A beats B | +7.13, 95% CI [+5.10, +9.24], 5/5 seeds | **holds** |
| H2: the gain is where B cannot see the evidence | accuracy +17.25 on those claims (n=327) vs +3.11 where B sees it (n=837); difference +14.14, one-sided lower bound +10.24 | **holds** |
| H3: A beats E | -0.78, 95% CI [-2.74, +1.19], 1/5 seeds | **fails** |

So on this data, picking the sentences relevant to the claim and reading 512 tokens is enough: it matches full-article
reading at about a third of the training time. The interval does not show that E is better than A. Inference time and
memory were not measured.

Official split, references only (753 of 758 test articles are also in train):

| Run | Macro F1 |
|---|---:|
| mmBERT-base, article 2,048 (3 seeds) | 83.58 +- 1.46 |
| mmBERT-base, first 512 (3 seeds) | 76.15 +- 1.08 |
| mmBERT-base, claim + gold evidence (3 seeds) | 87.24 +- 0.22 |
| mmBERT-base, claim only (1 seed) | 52.02 |
| XLM-R-large, 512, LR 1e-5 (3 seeds) | 74.31 +- 0.35 |
| majority class | 17.32 |

The paper reports 75.42 for XLM-R-large on the full-article task; recipes differ, and our XLM-R-large was still improving
at its third and last epoch.

**Limits.** One dataset, one model family. The article-disjoint split is ours, so its numbers are not comparable with the
paper's tables. All five A runs peaked at the last epoch, so A may be under-trained (the question of the follow-up). The
recipe was chosen on one seed, and the two best recipes were 0.12 apart. The claim-only run is far above the majority
class: claims alone carry some label signal.

## Follow-up: 6 epochs, 5-fold cross-validation (registered after the first result)
Question: is H3 failing only because the 2,048-token arm was under-trained? Both arms trained 6 epochs; 5 article-disjoint
folds over the old train articles (5,148 claims, 728 articles), 2 seeds each. Same dataset, and the question was chosen after
seeing the first result, so it is a replication, not an independent confirmation. Output: [results/analysis_cv.txt](results/analysis_cv.txt).

| Arm | Pooled out-of-fold macro F1 (seed 0 / seed 1) |
|---|---|
| A: article, 2,048 tokens | 82.67 / 81.74 |
| E: BM25 sentences, 512 tokens | 83.25 / 82.27 |

A - E = **-0.56** (article bootstrap [-1.37, +0.23], with the t interval over the 5 folds [-1.45, +0.34]); 0 of 5 folds favour A.
By the registered rule the upper bound is below +1.5: **retrieval is enough**. Longer training did not help A: mean
validation F1 at epochs 3 and 6 was 79.87 and 81.05 for A, 79.80 and 81.35 for E.

## Files
```text
train.py                 fine-tune + eval (article / evidence / claim input, 512-8,192 tokens, BM25 sentence selection)
data/fetch_stats.py      token lengths (mmBERT tokenizer)       data/evidence_pos.py  where the gold evidence sits
data/sanity.py           article overlap, label skew, tokenizer  data/make_split.py    the article-disjoint split
jobs/*.txt               the runs, one line of train.py arguments each
colab/run_jobs.sh        runs a jobs file on a Colab T4 (Colab CLI, WSL), fetching results after each job
analysis.py              decision rules over results/main/
analysis_cv.py           follow-up (docs/PREREG_cv.md): 5-fold cross-validation over results/cv/; data/make_cv.py builds it
recipe.py                recipe choice + fp16 rule over results/grid/
docs/PREREG.md           questions, runs, decision rules (registered before training)
```

## Running
```bash
pip install -r requirements.txt                 # torch (preinstalled on Colab), transformers 5.24.0, pyarrow
python train.py --split article --max_len 2048 --lr 5e-5 --seed 0 --out runs/A_s0   # downloads the data on first use
bash colab/run_jobs.sh jobs/01_grid.txt         # on a Colab T4
```

## Data and licence
ViFactCheck: Hoa et al., *ViFactCheck: A New Benchmark Dataset and Methods for Multi-domain News Fact-Checking in
Vietnamese*, AAAI 2025 ([arXiv:2412.15308](https://arxiv.org/abs/2412.15308), [data](https://huggingface.co/datasets/tranthaihoa/vifactcheck),
MIT). The data is downloaded at run time, not stored here. Code: MIT.
