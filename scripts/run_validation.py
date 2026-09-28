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

Экземпляры запроса делятся на FIT / EVAL (по умолчанию 5000 экземпляров
в EVAL -- больше, чем в первых версиях этой валидации (2000), потому что
на 2000 запросах разница между соседними значениями alpha_microcat была
уже сопоставима со статистическим шумом; на 5000 оценки стабильнее).
FIT играет роль "исторического лога запросов" (на нём строится
словарь/веса BM25 и исторические priors: запрос->объявление,
запрос->микрокатегория, запрос->редирект локации). EVAL играет роль
отложенных запросов бенчмарка. Это разбиение специально сделано
реалистичным: доля запросов бенчмарка, чей текст дословно встречается в
train.parquet, была напрямую проверена (~37%, см. README.md) -- то есть
офлайн-валидация не является искусственно более лёгкой, чем реальная
задача.

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
from src.ranking import rank_all, build_memo_prior, build_location_redirect, build_location_index

RNG_SEED = 42          # фиксированный seed -> разбиение FIT/EVAL воспроизводимо
K = 50                 # то же K, что и в задании (Recall@50)
N_EVAL = 5000          # число экземпляров запроса в EVAL (см. докстринг выше)
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
    FIT-прайоры (микрокатегория, редирект локации). Результат кэшируется
    в get_setup(), чтобы не повторять эти шаги при каждом новом
    эксперименте с параметрами бустинга."""
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
    n_eval = min(N_EVAL, int(0.1 * len(unique_groups)))
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
    item_location = items["item_location_id"].to_numpy()

    log(t0, "Строю текст корпуса объявлений ...")
    item_texts = build_item_corpus_text(items)
    log(t0, "готово")

    log(t0, "Обучаю BM25-индекс (max_df=0.4, чтобы выбросить шаблонные слова-метки) ...")
    bm25 = BM25Index(k1=1.5, b=0.75, min_df=2, max_df=0.4).fit(item_texts)
    log(t0, f"размер словаря={len(bm25.vectorizer.vocabulary_)}")

    log(t0, "Строю текст EVAL-запросов ...")
    eval_query_texts = build_query_text(eval_query_df).tolist()
    eval_qtext_list = eval_query_df["_qtext_norm"].tolist()
    eval_search_location = eval_query_df["search_location_id"].to_numpy()
    true_relevant = list(relevant_sets.values)

    # Priors строятся ТОЛЬКО по FIT-строкам -- иначе мы бы подсматривали в
    # саму EVAL-разметку при оценке качества, что сделало бы валидацию
    # нечестной.
    fit_by_qtext = fit_rows.groupby("_qtext_norm")
    qtext_to_microcat = fit_by_qtext["item_microcat_id"].agg(
        lambda s: s.value_counts(normalize=True).to_dict()
    )
    # top_n=6 (а не 5, финальный выбор) -- чтобы в main() можно было
    # честно перебрать top_n от 1 до 6 включительно, просто обрезая этот
    # список, а не перестраивая redirect заново на каждое значение.
    location_redirect = build_location_redirect(
        fit_rows["search_location_id"], fit_rows["item_location_id"], top_n=6
    )
    eval_search_location_redirect_targets = (
        eval_query_df["search_location_id"].map(location_redirect)
        .apply(lambda v: v if isinstance(v, list) else [])
        .tolist()
    )

    # Индекс "локация -> позиции объявлений" -- нужен для гарантированного
    # расширения кандидатов по локации (см. src/ranking.py, "ПЯТЫЙ СЛОЙ").
    location_index = build_location_index(item_location)

    setup = dict(
        bm25=bm25, item_ids=item_ids, item_microcat=item_microcat,
        item_location=item_location, location_index=location_index,
        eval_query_texts=eval_query_texts, eval_qtext_list=eval_qtext_list,
        eval_search_location=eval_search_location,
        eval_search_location_redirect_targets=eval_search_location_redirect_targets,
        true_relevant=true_relevant, qtext_to_microcat=qtext_to_microcat,
        fit_qtext_series=fit_rows["_qtext_norm"], fit_item_series=fit_rows["item_id"],
    )
    return setup


def get_setup(t0):
    """Загрузить закэшированную подготовку с диска, если она уже
    посчитана, иначе построить её заново и сохранить в кэш. Это чисто
    техническая оптимизация под итеративный подбор гиперпараметров бустов
    (build_setup -- самая долгая часть, от ~1 до нескольких минут на
    полном train.parquet в зависимости от загрузки системы, тогда как сам
    перебор вариантов бустинга -- секунды)."""
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

    def make_targets(redirect_top_n, weight_redirect):
        """Обрезать закэшированные top-6 цели редиректа до redirect_top_n
        штук; если weight_redirect=False, заменить их confidence на 1.0
        (полносильное применение, для сравнения)."""
        targets = s["eval_search_location_redirect_targets"]
        if redirect_top_n == 0:
            return [[] for _ in targets]
        out = []
        for row in targets:
            row = row[:redirect_top_n]
            if not weight_redirect:
                row = [(loc, 1.0) for loc, _ in row]
            out.append(row)
        return out

    def evaluate(label, redirect_top_n=5, weight_redirect=True, use_location_index=False, **kwargs):
        ranked = rank_all(
            s["eval_query_texts"], s["eval_qtext_list"], s["bm25"],
            s["item_ids"], s["item_microcat"],
            item_location=s["item_location"],
            search_location_list=s["eval_search_location"],
            search_location_redirect_targets=make_targets(redirect_top_n, weight_redirect),
            location_index=s["location_index"] if use_location_index else None,
            k=K, chunk_size=CHUNK, **kwargs,
        )
        r, _ = recall_at_k(s["true_relevant"], ranked, k=K)
        log(t0, f"[{label}]  Recall@{K} = {r:.4f}")
        return r

    empty_series = pd.Series(dtype=object)

    # 1) Прямая проверка гипотезы "услуги Avito - локальный рынок": какая
    #    доля выбранных в train.parquet объявлений находится ровно в той
    #    же локации, что и сам поиск.
    loc_cols = pd.read_parquet("data/train.parquet", columns=["item_location_id", "search_location_id"])
    match_rate = (loc_cols["item_location_id"] == loc_cols["search_location_id"]).mean()
    log(t0, f"Доля train-строк с item_location_id == search_location_id: {match_rate:.4f}")

    # 2) Чистый BM25 без каких-либо бустов -- отправная точка.
    evaluate("BASELINE только текстовый BM25", redirect_top_n=0,
             qtext_to_items=empty_series, qtext_to_microcat=empty_series,
             alpha_microcat=0.0, alpha_location=0.0)

    # 3) Буст по локации (прямое совпадение) -- главный найденный сигнал.
    #    Проверяем, что результат выходит на плато уже при alpha=1.0.
    #    redirect_top_n=0 здесь -- буст только по прямому совпадению.
    for alpha_loc in [0.1, 0.3, 0.5, 1.0, 2.0]:
        evaluate(f"BM25 + локация(alpha={alpha_loc}), без редиректа", redirect_top_n=0,
                 qtext_to_items=empty_series, qtext_to_microcat=empty_series,
                 alpha_microcat=0.0, alpha_location=alpha_loc)

    # 4) "Редирект" локации: 17.4% запросов бенчмарка не имеют в корпусе
    #    ни одного объявления с тем же search_location_id (маленькие
    #    города/районы без своих исполнителей) - прямой буст для них
    #    бесполезен. Проверяем эффект отдельно и в объединении с прямым
    #    совпадением (см. src/ranking.py, build_location_redirect).
    #    Полносильный редирект (weight_redirect=False) сравнивается со
    #    взвешенным по уверенности (weight_redirect=True, по умолчанию).
    evaluate("BM25 + локация(1.0) С редиректом(top-1), полносильно",
             redirect_top_n=1, weight_redirect=False,
             qtext_to_items=empty_series, qtext_to_microcat=empty_series,
             alpha_microcat=0.0, alpha_location=1.0)
    evaluate("BM25 + локация(1.0) С редиректом(top-1), взвешенным по уверенности",
             redirect_top_n=1, weight_redirect=True,
             qtext_to_items=empty_series, qtext_to_microcat=empty_series,
             alpha_microcat=0.0, alpha_location=1.0)

    # 5) Сколько адресатов редиректа брать (top-N)? У многих локаций
    #    исторические поиски делятся между несколькими соседними хабами
    #    почти поровну -- top-1 в таком случае ловит только часть.
    for n in [1, 2, 3, 4, 5, 6]:
        evaluate(f"BM25 + локация(1.0) + редирект(top-{n}, взвеш.)",
                 redirect_top_n=n, weight_redirect=True,
                 qtext_to_items=empty_series, qtext_to_microcat=empty_series,
                 alpha_microcat=0.0, alpha_location=1.0)

    # 6) На базе локации(1.0)+редирект(top-5, взвешенный) подбираем буст
    #    по микрокатегории. Без буста по локации этот сигнал стабильно
    #    вредил (см. README.md) - проверяем итоговый оптимум веса теперь,
    #    когда кандидат-пул уже сужен географией.
    for alpha_mc in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        evaluate(f"BM25 + локация(1.0)+редирект(top-5) + микрокатегория(alpha={alpha_mc})",
                 redirect_top_n=5, weight_redirect=True,
                 qtext_to_items=empty_series, qtext_to_microcat=s["qtext_to_microcat"],
                 alpha_microcat=alpha_mc, alpha_location=1.0)

    # 7) Меморизация поверх лучшей связки - проверяем, не потеряла ли она
    #    смысл теперь, когда локация (и её редирект) уже учтены явно (см.
    #    докстринг src/ranking.py).
    best_memo = build_memo_prior(s["fit_qtext_series"], s["fit_item_series"],
                                  max_distinct=5, top_n=3)
    evaluate("BM25 + локация(1.0)+редирект(top-5) + микрокатегория(0.2) + memo по тексту(5,3)",
             redirect_top_n=5, weight_redirect=True,
             qtext_to_items=best_memo, qtext_to_microcat=s["qtext_to_microcat"],
             alpha_microcat=0.2, alpha_location=1.0)

    # 8) Диагностика "слепой зоны" текстового поиска: сколько релевантных
    #    объявлений лежат в правильной локации, но не имеют вообще НИ
    #    ОДНОГО общего слова с запросом -- BM25 в принципе не мог их
    #    найти, сколько бы буста по локации ни давали (см. докстринг
    #    src/ranking.py, "ПЯТЫЙ СЛОЙ").
    item_id_to_pos = {iid: pos for pos, iid in enumerate(s["item_ids"])}
    q_all = s["bm25"].transform_query(s["eval_query_texts"])
    scores_all = (q_all @ s["bm25"].bm25_matrix.T).tocsr()
    n_total_rel, n_in_loc, n_in_loc_no_text = 0, 0, 0
    for i, relevant in enumerate(s["true_relevant"]):
        if not relevant:
            continue
        target_locs = {s["eval_search_location"][i]} | {
            loc for loc, _ in s["eval_search_location_redirect_targets"][i][:5]
        }
        row_start, row_end = scores_all.indptr[i], scores_all.indptr[i + 1]
        cols_with_score = set(scores_all.indices[row_start:row_end].tolist())
        for item_id in relevant:
            n_total_rel += 1
            pos = item_id_to_pos.get(item_id)
            if pos is None or s["item_location"][pos] not in target_locs:
                continue
            n_in_loc += 1
            if pos not in cols_with_score:
                n_in_loc_no_text += 1
    log(t0, f"Релевантных объявлений в правильной локации, но с нулевым "
             f"пересечением слов с запросом: {n_in_loc_no_text}/{n_in_loc} "
             f"({n_in_loc_no_text/n_in_loc:.1%} среди 'локальных', "
             f"{n_in_loc_no_text/n_total_rel:.1%} от всех релевантных)")

    # 9) Расширение кандидатов по локации (build_location_index): проверяем
    #    ограничение сверху на размер локации (чтобы не тащить тысячи
    #    объявлений мегагородов в каждый запрос) и без ограничения вовсе.
    #    Без ограничения оказалось лучше всего -- добавление кандидатов
    #    может только помочь (топ-50 всё равно выбирается по итоговому
    #    скору) и никогда не вредит, только чуть замедляет расчёт.
    evaluate("BM25 + локация(1.0)+редирект(top-5) + микрокатегория(0.2) + расширение по локации",
             redirect_top_n=5, weight_redirect=True, use_location_index=True,
             qtext_to_items=empty_series, qtext_to_microcat=s["qtext_to_microcat"],
             alpha_microcat=0.2, alpha_location=1.0)

    log(t0, "готово")


if __name__ == "__main__":
    main()
