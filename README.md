# Avito services search — candidate generation (Recall@50)

> Русская версия: [README.ru.md](README.ru.md).

Candidate-generation stage for the two-stage search cascade described in
the task: given a short service-search query, return up to 50 `item_id`
candidates from `benchmark_items.parquet` for the ranking stage to
re-rank. Optimized for **Recall@50**.

## TL;DR

- **Method**: BM25 (classic sparse lexical search, implemented from
  scratch with scipy/numpy — no `rank_bm25` or other search library) over
  a field-weighted bag-of-words built from title / description /
  structured params, **plus a location-match boost that turned out to be
  the single most important signal in this task** (Avito services are a
  hyperlocal market — see below), plus a small microcategory prior.
- **No external APIs, no downloaded models.** Everything runs locally
  with pandas/numpy/scipy/scikit-learn.
- **Offline validation** (held out from `train.parquet`, see below):
  **Recall@50 ≈ 0.751** (up from 0.185 for text-only BM25 — the location
  signal alone is responsible for almost all of that gap).
- Reproduce with:
  ```bash
  pip install -r requirements.txt
  # place train.parquet, benchmark_queries.parquet, benchmark_items.parquet in ./data/
  python3 scripts/generate_answer.py
  ```
  This writes `answer.csv` in the project root, byte-for-byte the same
  file that was submitted (the pipeline is fully deterministic: no random
  sampling, no GPU non-determinism).

## Repository layout

```
data/                     train.parquet, benchmark_queries.parquet, benchmark_items.parquet (not committed, see .gitignore)
src/
  text_utils.py           tokenization, Russian stopwords, field-weighting helper
  data_prep.py            builds item/query bag-of-words text (field weights live here)
  bm25.py                 from-scratch vectorized BM25 index (fit / score / chunked scoring)
  ranking.py              turns BM25 scores + location/category priors into top-50 candidate lists
  eval_utils.py           Recall@K metric matching the competition's definition
scripts/
  run_validation.py       offline validation harness + all parameter sweeps, run against train.parquet only
  generate_answer.py      final pipeline: reads benchmark_*.parquet, writes answer.csv
answer.csv                the submitted output
README.md                 this file
```

## Data and features used

| Used for retrieval | Field(s) | Why |
|---|---|---|
| Item text | `item_title_raw`, `item_infm_params_text`, `item_description_raw` | The only content that can lexically match a free-text query. |
| Query text | `search_query`, `search_infm_params_text` | Same reasoning on the query side. |
| **Location** | `search_location_id` vs `item_location_id` | **The dominant signal** — see below. |
| Item category | `item_microcat_id` | A secondary re-ranking prior, useful once location is already accounted for (see "Method"). |
| `search_category` | — | Not used: 91% of both train and benchmark queries share the same value (`114`), so it carries almost no information in this dataset. |
| Price, rating, phone/message flags | — | Not used in this submission (see "Future work"). |

Everything else (`item_price`, `item_rating`, `item_is_phone_hidden`, etc.)
is available for the downstream *ranking* stage but is not obviously
useful for *candidate generation*, whose only job is not to lose the
relevant item — it doesn't need to know if it's a 5-star or a 3-star
provider yet.

## Method

### 0. Location match — the main finding of this solution

Before tuning any text-scoring detail, I checked a simple hypothesis
directly on `train.parquet`: for the object a user actually picked, how
often is `item_location_id` exactly equal to the `search_location_id` of
the query that found it?

**83.1%** of the time.

This makes complete sense in hindsight: Avito services (a manicure
artist, an electrician, a private tutor) are an inherently **local**
market. The corpus has thousands of near-identically-worded listings for
common services ("маникюр", "ремонт стиральных машин") spread across the
whole country, and plain lexical BM25 has no way to prefer the listing
that's actually in the user's own city — it ranks a perfect-text-match
listing in a different city exactly the same as one next door. On a
189k-item national corpus, that means the true (local) answer is
routinely pushed out of the top 50 by textually-similar-but-geographically-
irrelevant listings.

Adding a location-match boost to the ranking score
(`src/ranking.py::boosted_top_k`, `alpha_location` parameter) raised
offline Recall@50 from **0.184 → 0.735** — by far the largest effect of
any change in this solution, an order of magnitude bigger than every
text-weighting or prior-tuning experiment combined. I did not use this
signal in my first submission and only found it once I went back to
question why text-only BM25 was underperforming so much relative to how
"easy" some of the example queries looked — a reminder to sanity-check
even features that look like they "belong" to the ranking stage.

Implementation: a soft additive boost, not a hard filter — items outside
the searcher's location are never excluded, only ranked lower, so a
query with no local listings at all still falls back gracefully to the
best texts match anywhere. The boost is scaled by that query's own
maximum BM25 score (`alpha_location * max_score`), so a single
`alpha_location` value works consistently across queries with very
different score magnitudes. Recall@50 plateaus already at
`alpha_location=1.0` (tested up to 50 with no further change — see
`scripts/run_validation.py`): at that point the boost already
guarantees every same-location candidate outranks every other-location
candidate for that query, so there's nothing left to gain by increasing
it further. I use the smallest value that reaches this plateau.

### 1. Text preprocessing (`src/text_utils.py`)

Lowercasing, `ё`→`е` normalization, a hand-written ~130-word Russian
stopword list (no downloads — kept fully offline/reproducible; nltk's
`stopwords` corpus would need `nltk.download()` at run time, which the
task's "no external calls" requirement rules out). No stemming/lemmatization:
Avito service ads are full of brand names, professional jargon and rare
compound words, and a quick manual check showed surface tokens already
give strong lexical overlap for this domain — a generic stemmer risked
merging unrelated words more often than it helped.

### 2. Field-weighted BM25 (`src/bm25.py`, `src/data_prep.py`)

We don't use the `rank_bm25` PyPI package — it scores query-vs-corpus with
a pure-Python loop over every document, far too slow for ~190k items ×
~2.5k queries. Instead, BM25 (Robertson & Sparck Jones) is implemented as
sparse-matrix algebra on top of scipy/scikit-learn's `CountVectorizer`,
so a full run finishes in well under a minute once fitted.

Title, structured params and description are folded into a single
bag-of-words per item (and per query, from `search_query` +
`search_infm_params_text`) by **repeating a field's tokens N times**
before counting — a cheap stand-in for per-field BM25 weighting. Grid
search on the offline validation set (`scripts/run_validation.py`) over
title/params/description weight triples found **5 / 3 / 1** to be the
smallest weighting that reaches the recall plateau (numbers below are
text-only BM25, *before* the location boost):

| weights (title/params/description) | Recall@50 |
|---|---|
| 1 / 1 / 1 (unweighted) | 0.174 |
| 3 / 2 / 0 (no description) | 0.154 |
| 3 / 2 / 1 | 0.182 |
| **5 / 3 / 1 (chosen)** | **0.185** |
| 7 / 4 / 1 | 0.184 (no further gain) |
| 5 / 3 / 2 | 0.185 (no further gain) |

Dropping description entirely cost ~0.03 Recall@50 — it still carries
real signal despite being the noisiest field.

**A vocabulary pitfall worth documenting** (see "Errors found" below):
`item_infm_params_text` is a fixed-template field ("Вид услуги ...",
"Место оказания услуг ...", "Тип стоимости за услугу ...", weekday
abbreviations, etc.). Before filtering, template/label words like
*место*, *оказания*, *услуг*, *вид*, *тип*, *стоимости* appeared in
95–100% of all 344,825 items. Because query↔item scoring is a sparse dot
product, common shared terms make the (query × item) score matrix
**structurally dense** — on a 30k-item sample, the raw item–item
similarity matrix was **100% dense**, and the full-corpus run's RSS hit
**>10 GB**, thrashing the machine. Fix: `CountVectorizer(max_df=0.4)` —
drop any term appearing in over 40% of items — which removes only the
near-universal boilerplate (real content words like "ремонт" at df≈0.32
survive) and cuts memory by an order of magnitude with no recall loss.
Query scoring is additionally done in chunks (`BM25Index.score_chunked`,
200 queries at a time) so peak memory never scales with the full
query × item product.

### 3. Historical priors from `train.parquet` (`src/ranking.py`)

Two more signals were tried, learned from query→item pairs in
`train.parquet` and layered on top of BM25 + the location boost, without
ever touching benchmark labels (there are none to touch):

- **Microcategory prior**: boost items whose `item_microcat_id` matches
  the microcategory historically chosen for the same query text.
  **Before the location boost existed, this signal consistently *hurt***
  offline recall (0.182 → 0.176 as its weight increased). **After adding
  the location boost, the same signal *helps*** (+0.016 Recall@50: 0.735
  → 0.751 at the best weight). See "Errors found" #3 for why the same
  prior flips from harmful to helpful depending on what else is already
  in the ranking — this was the most interesting methodological lesson
  in this project.
- **Historical item memorization**: if the exact same (normalized) query
  text previously led to a specific `item_id` still in the corpus, force
  it into the candidates. Useful on top of text-only BM25 (+0.001–0.0013
  Recall@50), but became redundant once location was added (0.7347
  without it vs 0.7344 with it — a difference within noise, consistently
  slightly negative). Likely reason: the same query text leads to
  *different* items in different cities, so a single "nationally most
  common" historical answer is no longer useful once location is modeled
  directly. **Not used in the final pipeline** — the function
  (`build_memo_prior`) is kept working and is still exercised in
  `scripts/run_validation.py` so this comparison stays reproducible.

### 4. Final ranking

For each query: BM25 score over the full item corpus → add the
location-match boost → add the microcategory-match boost → take the top
50 by score.

## How I validated before submitting

`train.parquet` has no query IDs — each row is a `(query, chosen item)`
pair. I reconstructed **354,463 distinct query instances** by grouping
rows on the full query feature set (`search_query`,
`search_location_id`, `search_is_delivery_search`,
`search_infm_params_text`, `search_category`); a group's item_ids are its
relevant set (mean 1.32 relevant items/query, matching the task's "usually
one or two" description).

I split query instances **90/10 into FIT/EVAL** (2,000 EVAL instances,
seeded). FIT plays the role of the historical query log (fits BM25 +
builds the priors); EVAL plays the role of held-out benchmark queries.
The item corpus for scoring is every unique item in the *whole* of
train.parquet (344,825 items) — knowing the corpus isn't a label leak,
only knowing an EVAL query's answer would be, and that's never used.

This split is deliberately realistic, not artificially easy: I confirmed
directly on `benchmark_queries.parquet` that **37.0%** of real benchmark
query texts also appear verbatim somewhere in `train.parquet` — almost
identical to what a random split of train.parquet itself reproduces — so
the offline number should transfer reasonably well to the real
benchmark score.

Full progression on the offline EVAL set (all numbers from
`scripts/run_validation.py`):

| Configuration | Recall@50 |
|---|---|
| Text-only BM25 (5/3/1 weights) | 0.1842 |
| + location boost (`alpha_location=1.0`) | 0.7347 |
| + microcategory prior (`alpha_microcat=0.2`) | **0.7510** (final) |
| + historical memorization on top of the above | 0.7500 (no gain, dropped) |

## Errors found during analysis, and what I did about them

1. **Text-only BM25 was quietly leaving the single biggest signal on the
   table.** Recall@50 of 0.18 looked low for how specific most queries
   are ("баня на дровах", "монтаж видеодомофонов"), which prompted a
   direct check of whether location explains the gap. It did: 83.1% of
   historically-chosen items share their `item_location_id` with the
   query's `search_location_id`. **Fix**: added a location-match boost
   (see "Method" §0) — this single change is responsible for ~85% of the
   total improvement in this solution (0.184 → 0.735 out of a final
   0.751).

2. **Memory blow-up from template boilerplate** (`item_infm_params_text`
   is a fixed form with labels like "Вид услуги", "Место оказания услуг",
   weekday names — present in 95-100% of items). Left unfiltered, this
   makes the query×item score matrix structurally dense (confirmed
   100% density on a 30k-item sample) and pushed RSS above 10 GB on the
   full corpus. **Fix**: `max_df=0.4` on the vectorizer, plus chunked
   scoring (`BM25Index.score_chunked`) so memory is bounded by chunk size
   regardless of density.

3. **The memorization prior made recall *worse*, not better**, when first
   added unconditionally (0.182 → 0.150). Root cause: exact query text is
   a much coarser key than a real query instance — generic one-word
   queries like *"маникюр"* (5,474 distinct historical items),
   *"массаж"* (3,830), *"электрик"* (2,451) map to a different provider
   in every city. Forcing all of a generic query's historical items into
   the top 50 drowns out the location/text-specific BM25 ranking and
   evicts genuinely relevant candidates. **Fix**: only trust this prior
   for query texts with ≤5 distinct historical items (81% of train query
   texts qualify) and cap injected items to the top 3 by frequency.
   (This was later dropped entirely once the location boost was added —
   see §3 of "Method" — but the fix itself is still the right lesson
   about this kind of prior.)

4. **The microcategory prior flipped from harmful to helpful once
   location was added — the most important methodological lesson here.**
   Without the location boost, this prior consistently hurt recall
   (0.182 → 0.176 as its weight increased); with the location boost
   already in place, the *same* prior *helps* (0.735 → 0.751). My
   reading: without location, the candidate pool for a typical query is
   dominated by textually-similar listings from all over the country, and
   a category prior just adds more noise to an already-noisy pool. With
   location narrowing the pool down to one city first, the remaining
   ambiguity (e.g. two different microcategories using similar wording)
   is exactly what the category prior is good at resolving. **Takeaway
   I'm keeping in mind**: a signal that looks harmful in isolation isn't
   necessarily a bad signal — it can be evaluated only relative to what
   else is already in the ranking, so I re-ran the full sweep after each
   change rather than trusting an earlier isolated verdict.

5. **`search_category`** is 114 for 91% of both train and benchmark
   queries — checked and explicitly not used as a signal, to avoid the
   false impression that category filtering was doing useful work.

## What I would try next with more time

- **Graceful location fallback**: right now, a query with zero same-city
  listings falls back to nationwide BM25 ranking. A softer version — e.g.
  a distance-based boost using `item_latitude`/`item_longitude` instead
  of exact `location_id` match, or falling back to a wider region before
  the whole country — could recover some of the ~17% of cases where the
  true answer isn't in the exact same `location_id`.
- **Field-specific BM25** (separate IDF per field) instead of the
  token-repetition weighting trick — the trick is a reasonable
  approximation but a true multi-field BM25F would let title/description
  have their own document-frequency statistics.
- **Fuzzy/typo-tolerant matching** (e.g. character n-gram TF-IDF as a
  second retriever, unioned with BM25 candidates) for query texts with
  no exact vocabulary overlap.

## Reproducibility notes

- Deterministic: no randomness in the final `generate_answer.py` path
  (the FIT/EVAL split used only in `run_validation.py` is seeded).
- All computation is local CPU (pandas/numpy/scipy/scikit-learn); no
  network calls, no pretrained embeddings/models.
- `answer.csv` is validated at generation time to satisfy every
  requirement in the task spec: one row per `query_id`, ≤50 unique
  `item_id`s per row, all ids present in `benchmark_items.parquet`.
