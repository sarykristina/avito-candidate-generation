"""
Офлайн-валидация пайплайна кандидатогенерации ТОЛЬКО по train.parquet
(разметка бенчмарка нигде не используется -- да её и не существует до
отправки решения).

Методология
-----------
train.parquet -- это лог пар (запрос, выбранное объявление): у него нет
явного query_id, каждая строка -- одна пара. Мы восстанавливаем
отдельные *экземпляры запроса*, группируя строки по полному набору
признаков запроса (search_query, search_location_id,
search_is_delivery_search, search_infm_params_text, search_category);
все строки внутри одной группы имеют одинаковые эти признаки, а
множество item_id внутри группы -- это "релевантное множество" для
данного экземпляра запроса.

Экземпляры запроса делятся 90/10 на FIT / EVAL. FIT играет роль
"исторического лога запросов" (на нём строится словарь/веса BM25 и
исторические priors: запрос->объявление, запрос->микрокатегория). EVAL
играет роль отложенных запросов бенчмарка.

Корпус объявлений для расчёта скора на этапе валидации -- это ВСЕ
уникальные item_id из *целого* train.parquet (FIT+EVAL вместе): он
играет роль benchmark_items.parquet.
"""

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_prep import build_item_corpus_text, build_query_text, normalize_query_text
from src.bm25 import BM25Index
from src.eval_utils import recall_at_k
from src.ranking import rank_all

RNG_SEED = 42
K = 50

GROUP_KEYS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]


def log(t0, msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    log(t0, "Загружаю train.parquet ...")
    train = pd.read_parquet("data/train.parquet")
    log(t0, f"строк={len(train)}  уникальных объявлений={train['item_id'].nunique()}")

    train["_qtext_norm"] = normalize_query_text(train["search_query"])
    group_id = train.groupby(GROUP_KEYS, sort=False).ngroup()
    train["_group_id"] = group_id
    n_groups = group_id.nunique()
    log(t0, f"уникальных экземпляров запроса={n_groups}")

    rng = np.random.default_rng(RNG_SEED)
    unique_groups = train["_group_id"].unique()
    rng.shuffle(unique_groups)
    n_eval = min(2000, int(0.1 * len(unique_groups)))
    eval_groups = set(unique_groups[:n_eval])
    fit_mask = ~train["_group_id"].isin(eval_groups)

    fit_rows = train[fit_mask]
    eval_rows = train[~fit_mask]
    log(t0, f"строк FIT={len(fit_rows)}  строк EVAL={len(eval_rows)}  экземпляров EVAL={n_eval}")

    eval_query_df = eval_rows.drop_duplicates("_group_id").set_index("_group_id")
    relevant_sets = eval_rows.groupby("_group_id")["item_id"].apply(set)
    eval_query_df = eval_query_df.loc[relevant_sets.index]

    items = train.drop_duplicates("item_id").set_index("item_id").sort_index()
    item_ids = items.index.to_numpy()
    item_microcat = items["item_microcat_id"].to_numpy()

    log(t0, "Строю текст корпуса объявлений ...")
    item_texts = build_item_corpus_text(items)
    log(t0, "готово")

    log(t0, "Обучаю BM25-индекс ...")
    bm25 = BM25Index(k1=1.5, b=0.75, min_df=2).fit(item_texts)
    log(t0, f"размер словаря={len(bm25.vectorizer.vocabulary_)}")

    log(t0, "Строю текст EVAL-запросов ...")
    eval_query_texts = build_query_text(eval_query_df).tolist()
    eval_qtext_list = eval_query_df["_qtext_norm"].tolist()
    true_relevant = list(relevant_sets.values)

    empty_series = pd.Series(dtype=object)
    baseline_ranked = rank_all(
        eval_query_texts, eval_qtext_list, bm25, item_ids, item_microcat,
        empty_series, empty_series, k=K, alpha_microcat=0.0,
    )
    recall, _ = recall_at_k(true_relevant, baseline_ranked, k=K)
    log(t0, f"[BASELINE только текстовый BM25]  Recall@{K} = {recall:.4f}")

    # Исторические priors, построенные по FIT-строкам.
    fit_by_qtext = fit_rows.groupby("_qtext_norm")
    qtext_to_items = fit_by_qtext["item_id"].apply(lambda s: set(s.tolist()))
    qtext_to_microcat = fit_by_qtext["item_microcat_id"].agg(
        lambda s: s.value_counts(normalize=True).to_dict()
    )

    for alpha in [0.0, 0.3, 0.5, 1.0]:
        ranked = rank_all(
            eval_query_texts, eval_qtext_list, bm25, item_ids, item_microcat,
            qtext_to_items, qtext_to_microcat, k=K, alpha_microcat=alpha,
        )
        r, _ = recall_at_k(true_relevant, ranked, k=K)
        log(t0, f"[BM25 + микрокатегория(alpha={alpha}) + memo]  Recall@{K} = {r:.4f}")

    log(t0, "готово")


if __name__ == "__main__":
    main()
