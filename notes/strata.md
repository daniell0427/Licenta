# Pre-registered evaluation strata

These subgroup definitions are fixed **before** any test-set numbers are
inspected, to prevent post-hoc cherry-picking. They are transcribed verbatim
from the implementation plan (Phase 6.1) — the definitions pre-date all
modelling.

The strata test *where* the sentiment-quality method helps, not just *whether*
it helps on average. Each is evaluated as E3 vs E7 (standard sentiment vs the
full method) on the volatility target.

## Narrative-consistency strata — by headline/summary divergence

A test item `(ticker, date)` is assigned by that day's absolute divergence
`abs_div` (mean per-article `|s - h|`):

| Stratum          | Definition                                  |
|------------------|---------------------------------------------|
| High-divergence  | `abs_div` in the **top 25%** of the test set |
| Rest             | the lower 75%                               |

**Hypothesis:** the narrative-consistency features should help *most* on
high-divergence days — that is exactly where headline and body disagree.

## Coverage-density strata — by news volume

A test item is assigned by that day's `news_count`:

| Stratum        | Definition           |
|----------------|----------------------|
| Low-coverage   | `news_count <= 2`    |
| Medium-coverage| `news_count` in 3–5  |
| High-coverage  | `news_count > 5`     |

**Hypothesis:** coverage-density shrinkage should help *most* in the
low-coverage stratum — that is where a daily sentiment estimate is least
reliable and most needs shrinking toward the prior.

## Reporting rule

If the method helps within a stratum but not overall, that is still a valid,
publishable, *scoped* finding — it is reported honestly, not discarded.
Per-stratum comparisons are McNemar-tested and Bonferroni-corrected alongside
the main significance matrix.
