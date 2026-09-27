"""
Превращает "сырые" BM25-скоры в финальный список топ-K кандидатов на
запрос, накладывая два дополнительных сигнала, выученных по историческим
парам (запрос -> выбранное объявление) из train.parquet:

1. Prior по микрокатегории: если такой (нормализованный) текст запроса
   уже встречался раньше, смотрим, в каких item_microcat_id пользователи
   в итоге выбирали объявления по нему, и даём мультипликативный буст
   объявлениям той же микрокатегории.

2. Историческая "меморизация" точного объявления: если этот же самый
   текст запроса раньше уже приводил к конкретному item_id, который
   всё ещё существует в текущем корпусе, это объявление принудительно
   попадает в список кандидатов (либо получает огромный аддитивный
   бонус, если BM25 и так его нашёл, либо добавляется "с нуля", если
   BM25-проход вообще не выдал по нему ненулевой скор).

Оба буста накладываются на уже отобранный BM25-список кандидатов; для
запросов без исторического совпадения единственным источником
ранжирования остаётся чистая текстовая релевантность.
"""

import numpy as np


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
):
    """Посчитать BM25-скор всех query_texts разом и применить
    boosted_top_k. Простая обёртка для одного вызова -- без разбиения на
    чанки, пока корпус небольшой."""
    scores = bm25_index.score(query_texts)
    return boosted_top_k(
        qtext_norm_list, scores, item_ids_sorted, item_microcat,
        qtext_to_items, qtext_to_microcat, k=k,
        alpha_microcat=alpha_microcat, memo_bonus=memo_bonus,
    )


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
                             каждой строки
      bm25_scores_csr    -- разреженная CSR-матрица (n_queries x n_items)
                             BM25-скоров
      item_ids_sorted    -- np.array с item_id, ОТСОРТИРОВАННЫЙ по
                             возрастанию, согласованный по индексам со
                             столбцами bm25_scores_csr
      item_microcat      -- np.array с item_microcat_id, согласованный по
                             индексам с item_ids_sorted
      qtext_to_items     -- pd.Series: текст запроса -> множество item_id,
                             исторически выбиравшихся по этому тексту
      qtext_to_microcat  -- pd.Series: текст запроса -> {microcat_id: доля}
    """
    results = []
    n = bm25_scores_csr.shape[0]
    for i in range(n):
        start, end = bm25_scores_csr.indptr[i], bm25_scores_csr.indptr[i + 1]
        cols = bm25_scores_csr.indices[start:end].copy()
        vals = bm25_scores_csr.data[start:end].astype(np.float64).copy()

        qtext = qtext_norm_list[i]

        microcat_prior = (
            qtext_to_microcat.get(qtext) if qtext in qtext_to_microcat.index else None
        )
        if microcat_prior:
            max_v = vals.max() if len(vals) else 1.0
            cand_microcats = item_microcat[cols] if len(cols) else np.array([])
            for mc, p in microcat_prior.items():
                boost_mask = cand_microcats == mc
                if boost_mask.any():
                    vals[boost_mask] += alpha_microcat * p * max_v

        memo_items = qtext_to_items.get(qtext) if qtext in qtext_to_items.index else None
        if memo_items:
            col_pos = {c: j for j, c in enumerate(cols)}
            extra_cols, extra_vals = [], []
            for it in memo_items:
                pos = np.searchsorted(item_ids_sorted, it)
                if pos < len(item_ids_sorted) and item_ids_sorted[pos] == it:
                    if pos in col_pos:
                        vals[col_pos[pos]] += memo_bonus
                    else:
                        extra_cols.append(pos)
                        extra_vals.append(memo_bonus)
            if extra_cols:
                cols = np.concatenate([cols, np.array(extra_cols)])
                vals = np.concatenate([vals, np.array(extra_vals)])

        if len(vals) > k:
            part = np.argpartition(vals, -k)[-k:]
            cols, vals = cols[part], vals[part]
        order = np.argsort(-vals)
        results.append(item_ids_sorted[cols[order]])
    return results
