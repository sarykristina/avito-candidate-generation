"""
Финальный прогон кандидатогенерации: строит файл ответа `answer.csv` для
benchmark_queries.parquet по корпусу benchmark_items.parquet.

Пайплайн (идентичен тому, что проверялся в scripts/run_validation.py;
полное описание подхода, все цифры и разбор ошибок -- в README.md /
README.ru.md):

  1. BM25 по взвешенному bag-of-words тексту (заголовок x5 + структуриро-
     ванные параметры x3 + описание x1), построенному для каждого
     объявления корпуса (веса заданы в src/data_prep.py и подобраны
     офлайн-валидацией). Термины, встречающиеся более чем в 40%
     объявлений (max_df=0.4), выбрасываются из словаря: как показал
     анализ, это шаблонные слова-метки из item_infm_params_text ("вид
     услуги", "место оказания услуг", названия дней недели и т.п.),
     которые не несут ранжирующей ценности, зато делают матрицу скоров
     запрос x объявление слишком плотной, чтобы уместиться в памяти.
  2. Текст запроса строится аналогично (search_query x5 + фильтры x3) и
     сравнивается с BM25-индексом порциями, ограниченными по памяти.
  3. Поверх BM25-скора накладываются прайоры, выученные по ВСЕМУ
     train.parquet (разметки бенчмарка не существует, и она нигде не
     используется):
       - буст по локации -- ГЛАВНЫЙ сигнал в этом решении: у 83.1%
         объявлений, реально выбранных пользователями в train.parquet,
         локация объявления совпадает с локацией поиска. Услуги на
         Avito -- это локальный рынок, и один текстовый BM25
         систематически предпочитает текстово похожие, но географически
         нерелевантные объявления из других городов.
       - "редирект" локации поверх этого же буста: у 17.4% запросов
         бенчмарка нет вообще ни одного объявления с тем же
         search_location_id в корпусе (маленький город/район без своих
         исполнителей) -- для них прямое совпадение не срабатывает.
         `build_location_redirect` учит по train.parquet, в какие до
         5 "хабов" (обычно соседние крупные города) реально ведут такие
         поиски, и буст проверяет совпадение с любым из них -- не только
         с самым частым (top-1), т.к. у многих локаций поиски делятся
         между несколькими соседними хабами почти поровну. Каждый
         адресат взвешен своей уверенностью (какая доля поисков из этой
         локации реально к нему привела) -- буст по редиректу
         масштабируется этой уверенностью, а не применяется всегда в
         полную силу, как прямое совпадение (см. src/ranking.py: и
         наивное полносильное применение, и top-1-вместо-top-5 оказались
         хуже на офлайн-валидации).
       - буст по микрокатегории, которую пользователи исторически
         выбирали для этого текста запроса -- полезен именно вместе с
         бустом по локации (см. src/ranking.py).
     Меморизация точных исторических объявлений (была в более ранней
     версии решения) не используется -- перестала давать прирост после
     добавления буста по локации (см. README.md).
  3б. Расширение кандидатов по локации (`build_location_index`) --
     САМОЕ крупное по эффекту улучшение из всех перечисленных. Все бусты
     выше применялись только к объявлениям, которые BM25 и так уже нашёл
     по тексту; если у релевантного объявления НЕТ общих слов с запросом
     вообще, оно физически не могло попасть в кандидаты. Проверка по
     train.parquet показала: у 3.1% релевантных объявлений в правильной
     локации нет ни единого общего слова с запросом. Теперь ВСЕ
     объявления из локации поиска и её редирект-адресатов гарантированно
     становятся кандидатами (с нулевым текстовым скором, если BM25 их не
     нашёл), а не только те, что случайно разделили хоть одно слово с
     запросом (см. src/ranking.py, "ПЯТЫЙ СЛОЙ").
  3в. Нечёткое сопоставление по символьным n-граммам (`src/fuzzy.py`) --
     небольшая добавка к скору по косинусной близости TF-IDF на
     3-5-граммах символов, поверх точного по словам BM25. Ловит
     словоформы/опечатки/разное разбиение составных слов, которые точная
     токенизация по словам пропускает, и особенно полезна для кандидатов
     из п.3б с нулевым BM25-скором. Офлайн-валидация показала узкий пик
     на alpha_fuzzy=0.1 (+0.0034 Recall@50 сверх пайплайна без неё), при
     заметно большем весе сигнал вредит (см. src/ranking.py, src/fuzzy.py).
  4. Топ-50 объявлений на запрос (по итоговому скору) записываются в
     answer.csv, дополнительно проходя проверку на соответствие всем
     требованиям формата из задания.

Запуск:
    python3 scripts/generate_answer.py
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_prep import (
    build_item_corpus_text, build_query_text, normalize_query_text,
    build_fuzzy_item_text, build_fuzzy_query_text,
)
from src.bm25 import BM25Index
from src.fuzzy import FuzzyIndex
from src.ranking import rank_all, build_location_redirect, build_location_index

K = 50

# Значения ниже выбраны по итогам офлайн-сравнения в
# scripts/run_validation.py (полные цифры и объяснение -- в README.md):
#   - ALPHA_LOCATION=1.0 -- буст по локации выходит на плато уже при
#     alpha=1.0 (дальнейшее увеличение вплоть до 50 не меняет Recall@50),
#     потому что при alpha=1.0 добавка уже гарантированно перевешивает
#     любую разницу чистых BM25-скоров внутри одного запроса. Берём
#     наименьшее значение, при котором достигается этот эффект -- по той
#     же логике, что и при подборе весов полей в src/data_prep.py.
#   - ALPHA_MICROCAT=0.2 -- подобран отдельным перебором ПОСЛЕ включения
#     буста по локации (без локации этот буст вредил и был отключён);
#     на выборке в 8000 offline-запросов пик находится в районе 0.2-0.3
#     (различия там уже в пределах шума), берём середину этого плато.
#   - ALPHA_FUZZY=0.1 -- пик узкий (см. src/fuzzy.py): полный перебор по
#     сетке на 5000 offline-запросах (scripts/run_validation.py) дал
#     0.05->0.8261, 0.08->0.8261, 0.1->0.8268 (максимум), 0.12->0.8253,
#     0.15->0.8262, 0.2->0.8256, дальше монотонно хуже вплоть до
#     2.0->0.7432 (базовый пайплайн без fuzzy: 0.8234).
ALPHA_LOCATION = 1.0
ALPHA_MICROCAT = 0.2
ALPHA_FUZZY = 0.1
MAX_DF = 0.4
CHUNK = 200


def log(t0, msg):
    """Логирование с меткой времени от старта -- на полном
    benchmark_items.parquet (189 212 объявлений) шаги занимают минуты,
    удобно видеть прогресс и на каком шаге сколько времени уходит."""
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    log(t0, "Загружаю данные ...")
    # Из train.parquet нужны только колонки, реально участвующие в
    # построении исторических prior'ов -- явное перечисление колонок
    # заметно ускоряет чтение parquet и экономит память.
    train = pd.read_parquet(
        "data/train.parquet",
        columns=["search_query", "item_id", "item_microcat_id",
                 "search_location_id", "item_location_id"],
    )
    items = pd.read_parquet("data/benchmark_items.parquet").set_index("item_id").sort_index()
    queries = pd.read_parquet("data/benchmark_queries.parquet")
    log(t0, f"train={len(train)}  объявлений={len(items)}  запросов={len(queries)}")

    # items отсортирован по item_id (см. .sort_index() выше) -- нужно для
    # np.searchsorted внутри boosted_top_k (src/ranking.py).
    item_ids = items.index.to_numpy()
    item_microcat = items["item_microcat_id"].to_numpy()
    item_location = items["item_location_id"].to_numpy()
    # Индекс "локация -> позиции объявлений" -- нужен, чтобы гарантированно
    # добавлять в кандидаты ВСЕ объявления нужной локации, даже с нулевым
    # текстовым скором (см. src/ranking.py, "ПЯТЫЙ СЛОЙ").
    location_index = build_location_index(item_location)

    log(t0, "Строю текст корпуса объявлений и обучаю BM25-индекс ...")
    item_texts = build_item_corpus_text(items)
    bm25 = BM25Index(k1=1.5, b=0.75, min_df=2, max_df=MAX_DF).fit(item_texts)
    log(t0, f"размер словаря={len(bm25.vectorizer.vocabulary_)}")

    log(t0, "Строю индекс нечёткого сопоставления по символьным n-граммам ...")
    fuzzy_item_texts = build_fuzzy_item_text(items)
    fuzzy_index = FuzzyIndex().fit(fuzzy_item_texts)
    log(t0, f"словарь символьных n-грамм={len(fuzzy_index.vectorizer.vocabulary_)}")

    log(t0, "Строю текст запросов ...")
    query_texts = build_query_text(queries).tolist()
    fuzzy_query_texts = build_fuzzy_query_text(queries).tolist()
    # Нормализованный "сырой" текст запроса -- ключ словаря для
    # qtext_to_microcat (микрокатегорийный prior строится по точному
    # совпадению текста запроса, а не по взвешенному BM25-тексту).
    qtext_norm_list = normalize_query_text(queries["search_query"]).tolist()
    search_location_list = queries["search_location_id"].to_numpy()

    log(t0, "Строю исторические priors из train.parquet (микрокатегория, редирект локации) ...")
    train["_qtext_norm"] = normalize_query_text(train["search_query"])
    qtext_to_microcat = train.groupby("_qtext_norm")["item_microcat_id"].agg(
        lambda x: x.value_counts(normalize=True).to_dict()
    )
    location_redirect = build_location_redirect(
        train["search_location_id"], train["item_location_id"], top_n=5
    )
    # Если для какого-то search_location_id из бенчмарка вообще не было
    # строк в train.parquet (не должно случаться -- все 58 "проблемных"
    # локаций бенчмарка встретились в train.parquet, см. README.md), на
    # всякий случай подставляем пустой список целей (тогда буст сработает
    # только по прямому совпадению, если оно есть).
    search_location_redirect_targets = (
        queries["search_location_id"].map(location_redirect)
        .apply(lambda v: v if isinstance(v, list) else [])
        .tolist()
    )
    # Меморизация точных объявлений (qtext_to_items) в этом пайплайне не
    # используется -- см. докстринг модуля и README.md: после добавления
    # буста по локации она перестала давать прирост.
    qtext_to_items = pd.Series(dtype=object)

    log(t0, "Считаю скор и ранжирую (BM25 + priors, порциями) ...")
    ranked = rank_all(
        query_texts, qtext_norm_list, bm25, item_ids, item_microcat,
        qtext_to_items, qtext_to_microcat,
        item_location=item_location, search_location_list=search_location_list,
        search_location_redirect_targets=search_location_redirect_targets,
        location_index=location_index,
        fuzzy_index=fuzzy_index, fuzzy_query_texts=fuzzy_query_texts,
        k=K, alpha_microcat=ALPHA_MICROCAT, alpha_location=ALPHA_LOCATION,
        alpha_fuzzy=ALPHA_FUZZY,
        chunk_size=CHUNK,
    )
    log(t0, "ранжирование завершено")

    # Резервный вариант для редкого случая, когда у запроса вообще нет
    # лексического пересечения с корпусом (например, слово встречается
    # только в этом запросе и ни в одном объявлении -- редкий сленг вне
    # словаря). Пустая строка кандидатов формально допустима форматом
    # answer.csv, но заведомо даёт recall=0 для этого запроса и выглядит
    # как недоработка -- вместо неё подставляем самые "проверенные"
    # (с наибольшим числом отзывов) объявления той же категории и
    # локации (с учётом редиректа), если такие есть, иначе просто той же
    # категории.
    n_empty = sum(1 for r in ranked if len(r) == 0)
    if n_empty:
        log(t0, f"{n_empty} запрос(ов) получили 0 кандидатов -- применяю fallback по популярности")
        popularity_fallback = (
            items.sort_values("item_rating_reviews_count", ascending=False).index.to_numpy()
        )
        cat_to_fallback = {}
        for cat, grp in items.groupby("item_category_id"):
            cat_to_fallback[cat] = grp.sort_values(
                "item_rating_reviews_count", ascending=False
            ).index.to_numpy()[:K]
        for i, r in enumerate(ranked):
            if len(r) == 0:
                cat = queries["search_category"].iloc[i]
                target_locs = {search_location_list[i]} | {
                    loc for loc, _ in search_location_redirect_targets[i]
                }
                loc_mask = np.isin(item_location, list(target_locs))
                same_loc = items[(items["item_category_id"] == cat) & loc_mask]
                if len(same_loc):
                    ranked[i] = same_loc.sort_values(
                        "item_rating_reviews_count", ascending=False
                    ).index.to_numpy()[:K]
                else:
                    ranked[i] = cat_to_fallback.get(cat, popularity_fallback[:K])

    answer = pd.DataFrame({
        "query_id": queries["query_id"].tolist(),
        "answer": [" ".join(map(str, r[:K])) for r in ranked],
    })

    # --- Финальные проверки на соответствие формату из условия задания ---
    # (одна строка на каждый query_id, не более 50 уникальных item_id в
    # строке, все item_id существуют в benchmark_items.parquet).
    assert answer["query_id"].is_unique, "query_id должны быть уникальны"
    assert set(answer["query_id"]) == set(queries["query_id"]), \
        "должна быть ровно одна строка на каждый query_id из benchmark_queries.parquet, без пропусков и лишних строк"
    valid_items = set(items.index)
    for row in answer["answer"]:
        ids = row.split()
        assert len(ids) <= K, "не более 50 item_id в одной строке"
        assert len(ids) == len(set(ids)), "без повторов item_id внутри строки"
        assert all(i in valid_items for i in ids), "все item_id должны существовать в benchmark_items.parquet"

    answer.to_csv("answer.csv", index=False)
    log(t0, f"Записан answer.csv, строк: {len(answer)}.")

    empty = (answer["answer"] == "").sum()
    log(t0, f"запросов с 0 кандидатами (после fallback, ожидается 0): {empty}")


if __name__ == "__main__":
    main()
