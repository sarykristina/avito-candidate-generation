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
  3. Поверх BM25-скора накладывается исторический prior, выученный по
     ВСЕМУ train.parquet (разметки бенчмарка не существует, и она нигде
     не используется):
       - меморизационный бонус, который принудительно добавляет в
         кандидаты любой item_id, исторически выбиравшийся для точно
         такого же текста запроса и всё ещё существующий в
         benchmark_items.parquet -- но ТОЛЬКО если у этого текста запроса
         небольшое (см. MEMO_MAX_DISTINCT) число разных исторических
         объявлений, иначе сигнал зашумляет ранжирование (см. ниже).
       - буст по микрокатегории в этом финальном пайплайне ОТКЛЮЧЁН
         (ALPHA_MICROCAT=0.0), потому что в офлайн-валидации он
         стабильно ухудшал Recall@50.
  4. Топ-50 объявлений на запрос (по итоговому скору) записываются в
     answer.csv, дополнительно проходя проверку на соответствие всем
     требованиям формата из задания.

Запуск:
    python3 scripts/generate_answer.py
"""

import sys
import time
import pandas as pd

sys.path.insert(0, ".")
from src.data_prep import build_item_corpus_text, build_query_text, normalize_query_text
from src.bm25 import BM25Index
from src.ranking import rank_all, build_memo_prior

K = 50

# Значения ниже выбраны по итогам офлайн-сравнения в
# scripts/run_validation.py (полные цифры -- в README.md):
#   - буст по микрокатегории СТАБИЛЬНО УХУДШАЛ Recall@50
#     (0.182 -> 0.176 по мере роста веса буста), поэтому он отключён
#     здесь (ALPHA_MICROCAT=0.0), хотя код буста остаётся рабочим в
#     src/ranking.py -- это осознанный отрицательный результат, а не
#     недоделанная функциональность.
#   - меморизационный prior по историческим объявлениям даёт небольшой,
#     но бесплатный прирост (+0.001..0.0013 Recall@50 офлайн), если
#     ограничить его текстами запроса с небольшим (<=MEMO_MAX_DISTINCT)
#     числом различных исторических объявлений -- без этого ограничения
#     он, наоборот, обрушивает recall (0.182 -> 0.150), потому что общие
#     однословные запросы вроде "маникюр" имеют тысячи разных
#     исторических объявлений по всей стране (по одному на исполнителя
#     в каждом городе), и форсировать их все в топ-50 значит вытеснить
#     оттуда специфичные для этого конкретного запроса кандидаты BM25.
ALPHA_MICROCAT = 0.0
MEMO_MAX_DISTINCT = 5
MEMO_TOP_N = 3
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
    # построении исторического prior'а -- явное перечисление колонок
    # заметно ускоряет чтение parquet (не тянем текстовые поля
    # объявлений/запросов, которые здесь не нужны) и экономит память.
    train = pd.read_parquet("data/train.parquet",
                             columns=["search_query", "item_id", "item_microcat_id"])
    items = pd.read_parquet("data/benchmark_items.parquet").set_index("item_id").sort_index()
    queries = pd.read_parquet("data/benchmark_queries.parquet")
    log(t0, f"train={len(train)}  объявлений={len(items)}  запросов={len(queries)}")

    # items отсортирован по item_id (см. .sort_index() выше) -- это
    # нужно для np.searchsorted внутри boosted_top_k (src/ranking.py),
    # когда меморизационный prior добавляет объявление, которого BM25
    # сам по себе не нашёл.
    item_ids = items.index.to_numpy()
    item_microcat = items["item_microcat_id"].to_numpy()

    log(t0, "Строю текст корпуса объявлений и обучаю BM25-индекс ...")
    item_texts = build_item_corpus_text(items)
    bm25 = BM25Index(k1=1.5, b=0.75, min_df=2, max_df=MAX_DF).fit(item_texts)
    log(t0, f"размер словаря={len(bm25.vectorizer.vocabulary_)}")

    log(t0, "Строю текст запросов ...")
    query_texts = build_query_text(queries).tolist()
    # Отдельно нормализованный "сырой" текст запроса (без взвешивания
    # полей) -- используется как ключ словаря в историческом
    # prior'е qtext_to_items, а не для самого BM25-скора.
    qtext_norm_list = normalize_query_text(queries["search_query"]).tolist()

    log(t0, "Строю исторические priors по train.parquet ...")
    train["_qtext_norm"] = normalize_query_text(train["search_query"])
    qtext_to_items = build_memo_prior(
        train["_qtext_norm"], train["item_id"],
        max_distinct=MEMO_MAX_DISTINCT, top_n=MEMO_TOP_N,
    )
    # Буст по микрокатегории отключён (ALPHA_MICROCAT=0.0, см. выше) --
    # пустой Series здесь просто гарантирует, что соответствующая ветка
    # внутри boosted_top_k всегда будет no-op, без завязки на то, что
    # вызывающий код обязательно передаст alpha_microcat=0.0.
    qtext_to_microcat = pd.Series(dtype=object)

    log(t0, "Считаю скор и ранжирую (BM25 + priors, порциями) ...")
    ranked = rank_all(
        query_texts, qtext_norm_list, bm25, item_ids, item_microcat,
        qtext_to_items, qtext_to_microcat, k=K, alpha_microcat=ALPHA_MICROCAT,
        chunk_size=CHUNK,
    )
    log(t0, "ранжирование завершено")

    # Резервный вариант для редкого случая, когда у запроса вообще нет
    # лексического пересечения с корпусом (например, слово встречается
    # только в этом запросе и ни в одном объявлении -- редкий сленг вне
    # словаря). Пустая строка кандидатов формально допустима форматом
    # answer.csv, но заведомо даёт recall=0 для этого запроса и выглядит
    # как недоработка -- вместо неё подставляем самые "проверенные"
    # (с наибольшим числом отзывов) объявления той же категории.
    # Ожидаемая польза для recall близка к нулю (случай очень редкий и
    # угадать конкретное объявление здесь почти невозможно), но это
    # строго не хуже пустого ответа и не портит остальные запросы.
    popularity_fallback = (
        items.sort_values("item_rating_reviews_count", ascending=False).index.to_numpy()
    )
    n_empty = sum(1 for r in ranked if len(r) == 0)
    if n_empty:
        log(t0, f"{n_empty} запрос(ов) получили 0 кандидатов -- применяю fallback по популярности")
        cat_to_fallback = {}
        for cat, grp in items.groupby("item_category_id"):
            cat_to_fallback[cat] = grp.sort_values(
                "item_rating_reviews_count", ascending=False
            ).index.to_numpy()[:K]
        for i, r in enumerate(ranked):
            if len(r) == 0:
                cat = queries["search_category"].iloc[i]
                # если такой категории вдруг нет среди объявлений
                # (не должно случаться на этом датасете, но на всякий
                # случай) -- откатываемся к глобальному топу по популярности.
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
