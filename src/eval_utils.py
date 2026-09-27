"""Метрика Recall@K для офлайн-валидации по train.parquet."""

import numpy as np


def recall_at_k(list_of_relevant_sets, list_of_ranked_item_ids, k=50):
    """
    list_of_relevant_sets[i]   -- множество item_id, действительно
                                   релевантных для запроса i (то, что
                                   пользователь выбрал в train.parquet)
    list_of_ranked_item_ids[i] -- наш ранжированный список
                                   кандидатов-item_id для запроса i
                                   (уже обрезан/упорядочен, длина <= k)

    Возвращает среднюю по запросам полноту (recall), в точности
    соответствующую формуле метрики из задания:
        mean_i( |топ-K_i ∩ релевантные_i| / |релевантные_i| )

    Запросы без релевантных объявлений (relevant пустое множество)
    пропускаются -- по условию задачи такого не бывает (у каждого
    запроса есть хотя бы одно выбранное объявление), но проверка
    оставлена для устойчивости кода при экспериментах с подвыборками.
    """
    scores = []
    for relevant, ranked in zip(list_of_relevant_sets, list_of_ranked_item_ids):
        if not relevant:
            continue
        topk = set(ranked[:k])
        scores.append(len(topk & relevant) / len(relevant))
    return float(np.mean(scores)), scores
