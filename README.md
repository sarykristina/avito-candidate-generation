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
  structured params, plus a small, carefully-capped "historical
  memorization" prior learned from `train.parquet`'s query→item log.
- **No external APIs, no downloaded models.** Everything runs locally
  with pandas/numpy/scipy/scikit-learn.
- **Offline validation** (held out from `train.parquet`, see below):
  **Recall@50 ≈ 0.185**.
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
  ranking.py              turns BM25 scores + historical priors into top-50 candidate lists
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
| Item category | `item_microcat_id` | Explored as a re-ranking prior (see "What didn't work"). |
| `search_category` | — | Not used: 91% of both train and benchmark queries share the same value (`114`), so it carries almost no information in this dataset. |
| Location, price, rating, phone/message flags | — | Not used in this submission (see "Future work"). |

Everything else (`item_price`, `item_rating`, coordinates, `item_is_phone_hidden`,
etc.) is available for the downstream *ranking* stage but is not obviously
useful for *candidate generation*, whose only job is not to lose the
relevant item — it doesn't need to know if it's a 5-star or a 3-star
provider yet.

## Method

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
smallest weighting that reaches the recall plateau:

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

Two priors are learned from query→item pairs in `train.parquet` and
layered on top of the BM25 score, without ever touching benchmark labels
(there are none to touch):

- **Microcategory prior**: boost items whose `item_microcat_id` matches
  the microcategory historically chosen for the same query text.
  **Result: this consistently *hurt* offline recall** (0.182 → 0.176 as
  the boost weight increased) and was **dropped** — see "Errors found".
- **Historical item memorization**: if the exact same (normalized) query
  text previously led to a specific `item_id` that still exists in the
  current corpus, force that item into the candidate list. This
  directly captures "this exact query has a known-good answer" —
  legitimate use of the provided query log, not label leakage (the prior
  is built once from train.parquet and reused verbatim against the
  benchmark corpus; the benchmark's own labels don't exist). **Capped**
  to query texts with ≤5 distinct historical items, keeping only their
  top-3 most frequent — see "Errors found" for why the cap is essential.
  Net effect: **+0.001–0.0013 Recall@50** in offline validation. Small
  but free, and see below for why it's likely to matter more on the real
  benchmark than in this internal test.

### 4. Final ranking

For each query: BM25 score over the full item corpus → apply the (capped)
memorization bonus → take the top 50 by score.

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

Final offline result: **Recall@50 = 0.1855** (BM25 5/3/1 + capped memo
prior, no microcategory boost), vs **0.1842** for BM25 alone.

I also directly checked (still using only `train.parquet` +
`benchmark_items.parquet`, no benchmark labels): of the 907 benchmark
queries (37.0%) whose text matches a train query exactly, **360 (14.7%
of all benchmark queries)** have at least one of their historical
train-side `item_id`s still present in `benchmark_items.parquet`. This
is exactly the case the memorization prior targets, and — unlike the
random 90/10 split above, where BM25 already tends to find these items on
its own because the query text closely echoes the item's own title —
these are real, specific benchmark queries, so I expect this prior to
contribute more visibly on the real benchmark than the +0.001 it showed
offline.

## Errors found during analysis, and what I did about them

1. **Memory blow-up from template boilerplate** (`item_infm_params_text`
   is a fixed form with labels like "Вид услуги", "Место оказания услуг",
   weekday names — present in 95-100% of items). Left unfiltered, this
   makes the query×item score matrix structurally dense (confirmed
   100% density on a 30k-item sample) and pushed RSS above 10 GB on the
   full corpus. **Fix**: `max_df=0.4` on the vectorizer, plus chunked
   scoring (`BM25Index.score_chunked`) so memory is bounded by chunk size
   regardless of density.

2. **The memorization prior made recall *worse*, not better**, when first
   added unconditionally (0.182 → 0.150). Root cause: exact query text is
   a much coarser key than a real query instance — generic one-word
   queries like *"маникюр"* (5,474 distinct historical items),
   *"массаж"* (3,830), *"электрик"* (2,451) map to a different provider
   in every city. Forcing all of a generic query's historical items into
   the top 50 drowns out the location/text-specific BM25 ranking and
   evicts genuinely relevant candidates. **Fix**: only trust this prior
   for query texts with ≤5 distinct historical items (81% of train query
   texts qualify) and cap injected items to the top 3 by frequency —
   turning a harmful signal (−0.032 Recall@50) into a small positive one
   (+0.0013).

3. **The microcategory prior looked promising on paper but hurt in
   practice** — recall dropped monotonically as its weight increased
   (0.182 → 0.176). My read: the historical microcategory distribution
   for a query text, aggregated across the whole country, isn't precise
   enough for this specific held-out instance to be worth re-ranking
   away from plain text relevance, and it can only ever reorder the
   *existing* BM25 shortlist. **Fix**: disabled (`alpha_microcat=0.0`) in
   the final pipeline; kept in the code (and the validation sweep) so the
   negative result is documented and reproducible rather than silently
   dropped.

4. **`search_category`** is 114 for 91% of both train and benchmark
   queries — checked and explicitly not used as a signal, to avoid the
   false impression that category filtering was doing useful work.

## What I would try next with more time

- **Location** (`search_location_id` vs `item_location_id` /
  lat-long): not used in this submission. A soft distance-based boost is
  a natural next experiment, but needs the same discipline as the
  microcategory prior (validate before trusting) since local-service
  marketplaces sometimes have city-spanning providers.
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
