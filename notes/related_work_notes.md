# Related Work Notes

## Closest prior work — Lee & Anderl (2025)

**Citation:** Lee & Anderl (2025). *Business news sentiment in energy stock prediction.*
Frontiers in Artificial Intelligence. doi:10.3389/frai.2025.1559900

**One-paragraph summary (Phase 1, Step 1 deliverable):**

> Lee & Anderl (2025) investigate whether FinBERT-based sentiment analysis of
> business news improves short-term stock prediction for the energy sector, and
> whether headlines or article content carry more predictive signal. The authors
> build a ~10-year dataset (Jan 2013–Nov 2022) of seven top-capitalization NYSE
> energy stocks (XOM, CVX, COP, EOG, SLB, OXY, PXD), collecting 18,254 headlines
> and 17,862 full articles from ten major financial publishers, reduced to 2,497
> daily training samples. FinBERT scores headlines and content *separately*,
> producing positive/negative/neutral probability triples fed as distinct feature
> subsets into a five-layer LSTM (50 hidden units, dropout 0.2) trained on 60-day
> windows to predict the next day's normalized closing price — a regression task,
> with XGBoost as a robustness baseline. Their central finding is that content
> sentiment significantly outperforms headline sentiment: the stock-price-plus-
> content subset cuts MAE from a 0.184 baseline to 0.168 (p = 0.001), while the
> headline subset's 0.176 MAE is not statistically significant (p = 0.219). They
> conclude headline sentiment is too shallow to help on its own, and list as
> limitations their narrow seven-stock energy scope, single LSTM/FinBERT
> specification, closing-price-only target, and that sentiment features alone are
> insufficient.

**Key facts for citation:**
- Dataset: 7 NYSE energy stocks, 2013–2022, 10 publishers.
- Method: FinBERT (pos/neg/neutral softmax), scored on headline AND content separately.
- Model: 5-layer LSTM, 50 hidden units, dropout 0.2, 60-day windows.
- Task: **regression** — next-day normalized closing price.
- Headline subset: MAE 0.176, **p = 0.219 (not significant)**.
- Content subset: MAE 0.168, **p = 0.001 (significant)**, vs baseline MAE 0.184.

## How this work differs (Phase 1, Step 2 — see differentiation.md)

Lee & Anderl *select between* headline and content sentiment as alternative
feature subsets. This work instead models the **discrepancy** between headline
and article-summary sentiment as an explicit *narrative-consistency* signal, and
adds **coverage-density shrinkage** to down-weight sparsely-covered ticker-days.
The question shifts from "which text segment predicts better?" to "how reliable
and internally consistent is the sentiment signal?"

This work also diverges on the prediction target: **binary direction** (1-day
and 5-day) rather than price regression. Direction is the decision-relevant
target and enables rigorous paired significance testing via McNemar's test,
which MAE-based regression comparisons cannot provide as cleanly.

## Supporting citations to collect (Phase 1, Step 3)
- Entman (1993) — framing theory (headline framing vs article content).
- DellaVigna & Pollet (2009) — limited attention to news.
- A MANA-Net-style sentiment-weighting paper — confidence/reliability weighting.
