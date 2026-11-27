# Does reading the whole article help a Vietnamese fact-checker?

**Status (27/11/2026): registered; recipe chosen on dev (LR 2e-5, cls pooling), no test set read yet.** Design and decision rules: [docs/PREREG.md](docs/PREREG.md).

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
