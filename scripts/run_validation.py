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
данного экземпляра запроса (у запроса может быть больше одного
релевантного объявления, как и написано в условии задания).

Экземпляры запроса делятся 90/10 на FIT / EVAL. FIT играет роль
"исторического лога запросов" (на нём строится словарь/веса BM25 и
исторические priors: запрос->объявление, запрос->микрокатегория). EVAL
играет роль отложенных запросов бенчмарка. Это разбиение специально
сделано реалистичным: доля запросов бенчмарка, чей текст дословно
встречается в train.parquet, была напрямую проверена (~37%, см.
README.md) -- то есть офлайн-валидация не является искусственно более
лёгкой, чем реальная задача.

Корпус объявлений для расчёта скора на этапе валидации -- это ВСЕ
уникальные item_id из *целого* train.parquet (FIT+EVAL вместе): он
играет роль benchmark_items.parquet (фиксированный, полностью известный
корпус, по которому ведётся поиск). Знание состава корпуса -- это не
утечка разметки; утечкой было бы только знание, какой конкретно item_id
является ответом на конкретный EVAL-запрос, а это нигде не используется
при построении FIT-прайоров или BM25-индекса.

Дорогая часть подготовки (сборка текста объявлений + обучение BM25)
кэшируется на диск при первом запуске, чтобы последующие эксперименты
с параметрами (бусты, веса) выполнялись быстро и не пересчитывали BM25
заново.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_prep import build_item_corpus_text, build_query_text, normalize_query_text
from src.bm25 import BM25Index
from src.eval_utils import recall_at_k
from src.ranking import rank_all, build_memo_prior

RNG_SEED = 42          # фиксированный seed -> разбиение FIT/EVAL воспроизводимо
K = 50                 # то же K, что и в задании (Recall@50)
CHUNK = 200            # размер чанка запросов для BM25Index.score_chunked (см. bm25.py)
CACHE_PATH = Path("data/cache/validation_setup.pkl")

# Признаки, по которым восстанавливаются "экземпляры запроса" из плоского
# train.parquet (см. докстринг модуля выше).
GROUP_KEYS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]


def log(t0, msg):
    """Простой логгер с меткой времени от старта скрипта -- удобно видеть,
    какой шаг сколько занимает на полном train.parquet (там это минуты,
    а не секунды)."""
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def build_setup(t0):
    """Один раз выполнить всю дорогую подготовку: загрузить train.parquet,
    восстановить экземпляры запроса, разбить их на FIT/EVAL, собрать текст
    корпуса объявлений и обучить на нём BM25-индекс, построить
    FIT-прайоры. Результат кэшируется в get_setup(), чтобы не повторять
    эти шаги при каждом новом эксперименте с параметрами бустинга."""
    log(t0, "Загружаю train.parquet ...")
    train = pd.read_parquet("data/train.parquet")
    log(t0, f"строк={len(train)}  уникальных объявлений={train['item_id'].nunique()}")

    train["_qtext_norm"] = normalize_query_text(train["search_query"])
    # ngroup() присваивает каждой уникальной комбинации GROUP_KEYS свой
    # числовой id -- это и есть "экземпляр запроса".
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

    # Для EVAL нужен один представительный ряд признаков запроса на
    # группу (они одинаковы внутри группы по построению) и множество
    # реально релевантных item_id этой группы.
    eval_query_df = eval_rows.drop_duplicates("_group_id").set_index("_group_id")
    relevant_sets = eval_rows.groupby("_group_id")["item_id"].apply(set)
    eval_query_df = eval_query_df.loc[relevant_sets.index]

    # Корпус объявлений для поиска = все уникальные объявления train.parquet
    # целиком (FIT+EVAL) -- аналог benchmark_items.parquet в реальной
    # задаче. Сортируем по item_id, чтобы потом можно было делать
    # np.searchsorted при добавлении меморизационных объявлений (см.
    # src/ranking.py).
    items = train.drop_duplicates("item_id").set_index("item_id").sort_index()
    item_ids = items.index.to_numpy()
    item_microcat = items["item_microcat_id"].to_numpy()

    log(t0, "Строю текст корпуса объявлений ...")
    item_texts = build_item_corpus_text(items)
    log(t0, "готово")

    log(t0, "Обучаю BM25-индекс (max_df=0.4, чтобы выбросить шаблонные слова-метки) ...")
    bm25 = BM25Index(k1=1.5, b=0.75, min_df=2, max_df=0.4).fit(item_texts)
    log(t0, f"размер словаря={len(bm25.vectorizer.vocabulary_)}")

    log(t0, "Строю текст EVAL-запросов ...")
    eval_query_texts = build_query_text(eval_query_df).tolist()
    eval_qtext_list = eval_query_df["_qtext_norm"].tolist()
    true_relevant = list(relevant_sets.values)

    # Prior "текст запроса -> распределение микрокатегорий" строится
    # ТОЛЬКО по FIT-строкам -- иначе мы бы подсматривали в саму EVAL-
    # разметку при оценке качества, что сделало бы валидацию нечестной.
    fit_by_qtext = fit_rows.groupby("_qtext_norm")
    qtext_to_microcat = fit_by_qtext["item_microcat_id"].agg(
        lambda s: s.value_counts(normalize=True).to_dict()
    )

    setup = dict(
        bm25=bm25, item_ids=item_ids, item_microcat=item_microcat,
        eval_query_texts=eval_query_texts, eval_qtext_list=eval_qtext_list,
        true_relevant=true_relevant, qtext_to_microcat=qtext_to_microcat,
        fit_qtext_series=fit_rows["_qtext_norm"], fit_item_series=fit_rows["item_id"],
    )
    return setup


def get_setup(t0):
    """Загрузить закэшированную подготовку с диска, если она уже
    посчитана, иначе построить её заново и сохранить в кэш. Это чисто
    техническая оптимизация под итеративный подбор гиперпараметров бустов
    (build_setup -- самая долгая часть, ~2-7 минут на полном
    train.parquet, тогда как сам перебор вариантов бустинга -- секунды)."""
    if CACHE_PATH.exists():
        log(t0, f"Загружаю закэшированную подготовку из {CACHE_PATH} ...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    setup = build_setup(t0)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(setup, f)
    log(t0, f"Подготовка закэширована в {CACHE_PATH}")
    return setup


def main():
    t0 = time.time()
    s = get_setup(t0)
    log(t0, "Подготовка готова.")

    # Инвариант, на котором строится np.searchsorted внутри
    # boosted_top_k (см. src/ranking.py) -- item_ids должен быть
    # отсортирован по возрастанию.
    assert (np.sort(s["item_ids"]) == s["item_ids"]).all()

    def evaluate(alpha_microcat, memo_series, label):
        ranked = rank_all(
            s["eval_query_texts"], s["eval_qtext_list"], s["bm25"],
            s["item_ids"], s["item_microcat"], memo_series, s["qtext_to_microcat"],
            k=K, alpha_microcat=alpha_microcat, chunk_size=CHUNK,
        )
        r, _ = recall_at_k(s["true_relevant"], ranked, k=K)
        log(t0, f"[{label}]  Recall@{K} = {r:.4f}")
        return r

    empty_series = pd.Series(dtype=object)

    # 1) Чистый BM25 без каких-либо исторических бустов -- отправная точка
    #    для сравнения всех последующих экспериментов.
    evaluate(0.0, empty_series, "BASELINE только текстовый BM25")

    # 2) Подбор ограничения меморизационного prior'а (max_distinct, top_n).
    #    Без такого ограничения (см. src/ranking.py) recall катастрофически
    #    падает из-за общих запросов вроде "маникюр" -- здесь мы наглядно
    #    воспроизводим, как разные пороги влияют на итоговую метрику.
    for max_distinct, top_n in [(1, 1), (3, 3), (5, 3), (10, 3)]:
        memo = build_memo_prior(
            s["fit_qtext_series"], s["fit_item_series"],
            max_distinct=max_distinct, top_n=top_n,
        )
        evaluate(0.0, memo,
                 f"BM25 + memo(max_distinct={max_distinct}, top_n={top_n}), без микрокатегории")

    # 3) Взяв разумную конфигурацию меморизации (5, 3), перебираем вес
    #    буста по микрокатегории -- чтобы честно проверить гипотезу, а
    #    не просто выбросить её без замеров. Как показывает результат
    #    (см. README.md), этот буст стабильно ВРЕДИТ recall, поэтому в
    #    финальном пайплайне (scripts/generate_answer.py) он отключён.
    best_memo = build_memo_prior(s["fit_qtext_series"], s["fit_item_series"],
                                  max_distinct=5, top_n=3)
    for alpha in [0.0, 0.3, 0.5, 1.0, 2.0]:
        evaluate(alpha, best_memo, f"BM25 + memo(5,3) + микрокатегория(alpha={alpha})")

    log(t0, "готово")


if __name__ == "__main__":
    main()
