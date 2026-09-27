"""
Превращает "сырые" BM25-скоры в финальный список топ-K кандидатов на
запрос, накладывая два дополнительных сигнала, выученных по историческим
парам (запрос -> выбранное объявление) из train.parquet:

1. Prior по микрокатегории: если такой (нормализованный) текст запроса
   уже встречался раньше, смотрим, в каких item_microcat_id пользователи
   в итоге выбирали объявления по нему, и даём мультипликативный буст
   объявлениям той же микрокатегории. Это МЯГКИЙ буст (никогда не жёсткий
   фильтр), поэтому в теории он не должен вредить recall, даже если
   категорийный prior ошибается или текст запроса вообще не встречался
   раньше -- он может только переупорядочить уже отобранный BM25-список.
   НА ПРАКТИКЕ этот буст стабильно ухудшал офлайн-recall и в финальном
   решении отключён (alpha_microcat=0.0) -- см. пункт 3 в разделе
   "Найденные ошибки" README.md. Код и логика буста оставлены в
   реализации намеренно: во-первых, чтобы отрицательный результат был
   воспроизводим и виден в scripts/run_validation.py, а не потерялся;
   во-вторых, чтобы им было легко воспользоваться в будущем, если
   появится более точный источник категорийного prior'а.

2. Историческая "меморизация" точного объявления: если этот же самый
   текст запроса раньше уже приводил к конкретному item_id, который
   всё ещё существует в текущем корпусе, это объявление принудительно
   попадает в список кандидатов (либо получает огромный аддитивный
   бонус, если BM25 и так его нашёл, либо добавляется "с нуля", если
   BM25-проход вообще не выдал по нему ненулевой скор). Это напрямую
   использует сигнал "у этого конкретного запроса уже есть известный
   хороший ответ" -- легитимное использование предоставленного лога
   запросов (train.parquet), а не утечка разметки бенчмарка: разметки
   бенчмарка просто не существует, prior строится один раз по
   train.parquet и затем применяется как есть к корпусу
   benchmark_items.parquet.

   ВАЖНО: этот прием безопасен ТОЛЬКО для текстов запроса, которые
   ведут к небольшому, сконцентрированному множеству исторических
   объявлений. Общий однословный запрос вроде "маникюр" имеет в
   train.parquet больше 5000 различных исторических item_id (по одному
   на каждого исполнителя в каждом городе) -- если форсировать в
   кандидаты их все, это полностью заглушит специфичный для
   локации/текста сигнал BM25 и УМЕНЬШИТ recall (подтверждено
   эмпирически: офлайн Recall@50 упал с 0.182 до 0.151 до того, как
   было добавлено ограничение ниже). Поэтому мы применяем
   меморизационный буст только когда историческое множество объявлений
   для текста запроса не больше `memo_max_distinct` элементов (у 81%
   текстов запросов в train.parquet оно <=3, то есть подавляющее
   большинство "специфичных" запросов, для которых меморизации можно
   доверять, всё ещё покрыто), и дополнительно ограничиваем сверху
   число добавляемых объявлений на запрос параметром `memo_top_n`.

Оба буста накладываются НА УЖЕ ОТОБРАННЫЙ BM25-список кандидатов;
для запросов без (надёжного) исторического совпадения единственным
источником ранжирования всегда остаётся чистая текстовая релевантность.
"""

import numpy as np
import pandas as pd


def build_memo_prior(qtext_series, item_id_series, max_distinct=5, top_n=3):
    """Построить prior "текст запроса -> исторические объявления",
    который использует `boosted_top_k`, по строкам (qtext_series,
    item_id_series) из лога запросов (например, train.parquet).

    Оставляет только те тексты запроса, у которых множество исторических
    объявлений не больше `max_distinct` элементов (см. пояснение в
    докстринге модуля выше, почему это критически важно для recall), и
    среди них берёт только `top_n` самых часто выбираемых объявлений.

    Возвращает pd.Series: нормализованный текст запроса -> list[item_id].
    """
    grouped = item_id_series.groupby(qtext_series)
    result = {}
    for qtext, items in grouped:
        # value_counts() сортирует по убыванию частоты -- это даёт нам
        # "самые частые сначала" бесплатно, без отдельной сортировки.
        counts = items.value_counts()
        if len(counts) <= max_distinct:
            result[qtext] = counts.index[:top_n].tolist()
        # если у текста запроса больше max_distinct различных
        # исторических объявлений (общий запрос вроде "массаж") -- мы
        # НЕ добавляем его в prior вообще: доверять такому "разбросу"
        # исторических ответов небезопасно (см. докстринг выше).
    return pd.Series(result, dtype=object)


def rank_all(
    query_texts,
    qtext_norm_list,
    bm25_index,
    item_ids_sorted,
    item_microcat,
    qtext_to_items,
    qtext_to_microcat,
    k=50,
    alpha_microcat=0.5,
    memo_bonus=1e6,
    chunk_size=200,
):
    """Удобная обёртка: считает скор `query_texts` относительно
    `bm25_index` порциями, безопасными по памяти (см.
    BM25Index.score_chunked), и применяет `boosted_top_k` к каждой
    порции, возвращая объединённые ранжированные списки item_id по
    каждому запросу в исходном порядке.

    Разбиение на чанки здесь -- чисто техническая мера (см. bm25.py):
    сама логика бустинга применяется к каждому чанку независимо и не
    меняется от того, что запросы обрабатываются порциями, а не все
    сразу."""
    results = []
    for start in range(0, len(query_texts), chunk_size):
        chunk_texts = query_texts[start:start + chunk_size]
        chunk_qtext = qtext_norm_list[start:start + chunk_size]
        # score_chunked -- генератор; т.к. мы сами уже нарезали чанк
        # нужного размера, chunk_size=len(chunk_texts) гарантирует, что
        # он отдаст этот чанк целиком за один next() без дополнительного
        # внутреннего разбиения.
        scores_chunk = next(bm25_index.score_chunked(chunk_texts, chunk_size=len(chunk_texts)))
        results.extend(boosted_top_k(
            chunk_qtext, scores_chunk, item_ids_sorted, item_microcat,
            qtext_to_items, qtext_to_microcat, k=k,
            alpha_microcat=alpha_microcat, memo_bonus=memo_bonus,
        ))
    return results


def boosted_top_k(
    qtext_norm_list,
    bm25_scores_csr,
    item_ids_sorted,
    item_microcat,
    qtext_to_items,
    qtext_to_microcat,
    k=50,
    alpha_microcat=0.5,
    memo_bonus=1e6,
):
    """
    Параметры:
      qtext_norm_list    -- list[str], нормализованный текст запроса для
                             каждой строки (см. normalize_query_text)
      bm25_scores_csr    -- разреженная CSR-матрица (n_queries x n_items)
                             BM25-скоров
      item_ids_sorted    -- np.array с item_id, ОТСОРТИРОВАННЫЙ по
                             возрастанию, согласованный по индексам со
                             столбцами bm25_scores_csr (нужен для
                             np.searchsorted при добавлении
                             меморизационных объявлений)
      item_microcat      -- np.array с item_microcat_id, согласованный по
                             индексам с item_ids_sorted
      qtext_to_items     -- pd.Series: текст запроса -> список item_id
                             (результат build_memo_prior)
      qtext_to_microcat  -- pd.Series: текст запроса -> {microcat_id: доля}
    """
    results = []
    n = bm25_scores_csr.shape[0]
    for i in range(n):
        # Достаём ненулевые BM25-скоры этого запроса напрямую из
        # CSR-структуры (indptr/indices/data), без обращения к
        # плотному представлению строки.
        start, end = bm25_scores_csr.indptr[i], bm25_scores_csr.indptr[i + 1]
        cols = bm25_scores_csr.indices[start:end].copy()
        vals = bm25_scores_csr.data[start:end].astype(np.float64).copy()

        qtext = qtext_norm_list[i]

        # --- Буст по микрокатегории (в финальном решении отключён,
        # alpha_microcat=0.0, но логика оставлена для воспроизводимости
        # эксперимента и на будущее -- см. докстринг модуля) ---
        microcat_prior = (
            qtext_to_microcat.get(qtext) if qtext in qtext_to_microcat.index else None
        )
        if microcat_prior:
            max_v = vals.max() if len(vals) else 1.0
            cand_microcats = item_microcat[cols] if len(cols) else np.array([])
            for mc, p in microcat_prior.items():
                boost_mask = cand_microcats == mc
                if boost_mask.any():
                    # Буст пропорционален p (доля исторических выборов
                    # в эту микрокатегорию для данного текста запроса) и
                    # масштабирован относительно максимального BM25-скора
                    # ЭТОГО запроса -- иначе фиксированная константа была
                    # бы либо ничтожной, либо доминирующей в зависимости
                    # от абсолютной величины BM25-скоров конкретного
                    # запроса (она сильно варьируется от запроса к
                    # запросу из-за разной редкости слов).
                    vals[boost_mask] += alpha_microcat * p * max_v

        # --- Меморизация точного исторического объявления ---
        memo_items = qtext_to_items.get(qtext) if qtext in qtext_to_items.index else None
        if memo_items:
            col_pos = {c: j for j, c in enumerate(cols)}
            extra_cols, extra_vals = [], []
            for it in memo_items:
                # item_ids_sorted отсортирован -> бинарный поиск вместо
                # линейного сканирования всего корпуса на каждый запрос.
                pos = np.searchsorted(item_ids_sorted, it)
                if pos < len(item_ids_sorted) and item_ids_sorted[pos] == it:
                    if pos in col_pos:
                        # BM25 и так нашёл это объявление (есть общие
                        # слова с запросом) -- просто добавляем сверху
                        # огромный бонус, чтобы гарантированно попасть
                        # в топ-k.
                        vals[col_pos[pos]] += memo_bonus
                    else:
                        # BM25 вообще не дал по нему ненулевой скор
                        # (текст запроса и текст объявления не пересекаются
                        # лексически) -- добавляем "с нуля" отдельной
                        # записью.
                        extra_cols.append(pos)
                        extra_vals.append(memo_bonus)
                # если item_id из истории отсутствует в текущем корпусе
                # (в реальности -- в benchmark_items.parquet), просто
                # пропускаем: подсунуть несуществующий item_id в ответ
                # нельзя по условию задания.
            if extra_cols:
                cols = np.concatenate([cols, np.array(extra_cols)])
                vals = np.concatenate([vals, np.array(extra_vals)])

        # --- Финальный отбор топ-k по итоговому (BM25 + бусты) скору ---
        if len(vals) > k:
            part = np.argpartition(vals, -k)[-k:]
            cols, vals = cols[part], vals[part]
        order = np.argsort(-vals)
        results.append(item_ids_sorted[cols[order]])
    return results
