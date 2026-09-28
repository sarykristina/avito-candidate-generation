"""
Превращает "сырые" BM25-скоры в финальный список топ-K кандидатов на
запрос, накладывая дополнительные сигналы поверх чистой текстовой
релевантности.

0. Буст по локации -- ГЛАВНЫЙ сигнал в этом решении, важнее самого
   текста. Прямая проверка по train.parquet показала, что у 83.1%
   выбранных объявлений item_location_id совпадает с search_location_id
   запроса, которым их нашли. Это интуитивно: Avito-услуги -- это
   ЛОКАЛЬНЫЙ рынок (мастер маникюра, электрик, репетитор), и практически
   везде в стране найдутся объявления с очень похожим текстом
   ("маникюр", "ремонт стиральных машин"), но пользователю нужен
   конкретно исполнитель из своего города. Одного текстового BM25 для
   этого категорически недостаточно: он одинаково ранжирует объявления
   из любого города, и на многомиллионном корпусе правильный (локальный)
   исполнитель обычно вытесняется из топ-50 более "текстово удачными", но
   географически нерелевантными объявлениями. Добавление этого буста
   подняло офлайн Recall@50 с 0.184 до 0.735 -- на порядок больше
   эффекта, чем у любого другого отдельного сигнала в этом решении
   (подробности -- в README.md, "Найденные ошибки").

   ВТОРОЙ СЛОЙ ЭТОГО ЖЕ СИГНАЛА -- "редирект" локации. У 17.4% запросов
   бенчмарка (`search_location_id`) в корпусе `benchmark_items.parquet`
   нет вообще ни одного объявления с таким же `item_location_id` -- это
   небольшие города/районы без своих исполнителей, для буста по прямому
   совпадению они просто "не срабатывают". Проверка по train.parquet
   показала, что для 44% всех различных `search_location_id` самый
   частый исторически выбранный `item_location_id` -- это НЕ он сам, а
   один и тот же "соседний хаб" (обычно ближайший крупный город, который
   фактически обслуживает этот маленький населённый пункт).
   `build_location_redirect` учит это отображение `search_location_id ->
   самый частый item_location_id` по всей истории запросов, и буст
   проверяет совпадение с ЛЮБЫМ из двух: и с исходной локацией поиска, и
   с её "редиректом".

   ТРЕТИЙ СЛОЙ -- уверенность редиректа. "Мода" (самый частый адресат)
   объясняет для разных search_location_id от ~5% до 100% исторических
   выборов -- где-то это уверенный вывод, а где-то почти угадывание
   наугад между двумя примерно равновероятными соседними хабами. Слепо
   применять полный буст (как к прямому совпадению) в обоих случаях
   означало бы одинаково доверять надёжному и ненадёжному сигналу.
   Поэтому буст по редиректу (в отличие от буста по прямому совпадению)
   масштабируется на `confidence` -- долю исторических поисков из этой
   локации, которые реально привели к этому адресату. Проверялись также
   вариант с жёстким порогом уверенности (буст либо есть, либо нет) -- он
   оказался ХУЖЕ плавного взвешивания: слишком высокий порог полностью
   исключает буст для многих локаций, для которых даже не самый надёжный
   редирект всё равно лучше, чем никакого гео-сигнала вообще (0.783 при
   пороге 0.8 против 0.804 при плавном взвешивании).

   ЧЕТВЁРТЫЙ СЛОЙ -- top-N редирект вместо top-1. У многих
   search_location_id исторические поиски делятся не между одним
   доминирующим хабом, а между НЕСКОЛЬКИМИ соседними (например 45%/40%
   между двумя близкими городами) -- top-1 редирект в таком случае ловит
   только большую часть, а вторая по популярности цель остаётся
   незамеченной. `build_location_redirect` берёт не одну моду, а до
   `top_n` самых частых адресатов для каждой локации (каждый -- со своей
   confidence), и буст проверяет совпадение с любым из них, каждый раз
   взвешивая своей уверенностью. Recall@50 растёт с top_n=1 (0.804) до
   top_n=5 (0.812) и на этом выходит на плато (top_n=6 уже чуть хуже --
   шестой по частоте адресат для большинства локаций это уже шум) -- см.
   README.md за полную таблицу.

1. Prior по микрокатегории: если такой (нормализованный) текст запроса
   уже встречался раньше, смотрим, в каких item_microcat_id пользователи
   в итоге выбирали объявления по нему, и даём мультипликативный буст
   объявлениям той же микрокатегории. ВАЖНЫЙ НЮАНС: пока в пайплайне не
   было буста по локации, этот буст стабильно УХУДШАЛ recall (см.
   README.md); после добавления локации он, наоборот, ощутимо ПОМОГАЕТ.
   Объяснение: без локации кандидат-пул для типичного запроса засорён
   объявлениями похожей тематики со всей страны, и категорийный prior
   просто путает эту и так шумную картину; с локацией пул уже сужен до
   объявлений одного города (или его хабов), и небольшая добавка
   категорийного prior'а помогает разрешить оставшуюся неоднозначность.

2. Историческая "меморизация" точного объявления: если этот же самый
   текст запроса раньше уже приводил к конкретному item_id, который
   всё ещё существует в текущем корпусе, это объявление принудительно
   попадает в список кандидатов. САМ ПО СЕБЕ (до буста по локации) это
   давало небольшой прирост поверх текстового BM25. После добавления
   буста по локации эффект memo стал нулевым / чуть отрицательным --
   скорее всего потому, что один и тот же текст запроса реально ведёт к
   РАЗНЫМ объявлениям в разных городах, а memo принудительно подставляет
   "усреднённый по стране" исторический ответ вне зависимости от
   локации конкретного запроса, что уже не нужно, когда локация и так
   учтена явно (тем более с редиректом). Функция `build_memo_prior`
   оставлена рабочей (используется в scripts/run_validation.py как
   задокументированный эксперимент), но в финальном пайплайне
   (scripts/generate_answer.py) не используется.

Ключевой методологический вывод: сигналы нельзя оценивать по отдельности
"в вакууме" -- микрокатегорийный prior выглядел вредным, пока не был
добавлен более сильный сигнал (локация), после чего стал полезным. Все
числа перепроверены в scripts/run_validation.py на одном и том же
offline-сплите, чтобы сравнение было честным.
"""

import numpy as np
import pandas as pd


def build_memo_prior(qtext_series, item_id_series, max_distinct=5, top_n=3):
    """Построить prior "текст запроса -> исторические объявления" по
    строкам (qtext_series, item_id_series) из лога запросов (например,
    train.parquet).

    Оставляет только те тексты запроса, у которых множество исторических
    объявлений не больше `max_distinct` элементов (без этого ограничения
    общие запросы вроде "маникюр" с тысячами разных исторических
    объявлений полностью забивают ранжирование шумом), и среди них берёт
    только `top_n` самых часто выбираемых объявлений.

    В финальном пайплайне (scripts/generate_answer.py) этот prior больше
    не используется -- после добавления буста по локации он перестал
    давать прирост (см. докстринг модуля выше), но функция оставлена
    рабочей и используется в scripts/run_validation.py, чтобы это можно
    было перепроверить.

    Возвращает pd.Series: нормализованный текст запроса -> list[item_id].
    """
    grouped = item_id_series.groupby(qtext_series)
    result = {}
    for qtext, items in grouped:
        counts = items.value_counts()
        if len(counts) <= max_distinct:
            result[qtext] = counts.index[:top_n].tolist()
    return pd.Series(result, dtype=object)


def build_location_redirect(search_location_series, item_location_series, top_n=5):
    """Построить отображение "search_location_id -> список из (до
    top_n) самых частых исторически выбранных item_location_id, каждый
    со своей confidence" по строкам лога запросов (например,
    train.parquet).

    Нужно для запросов из небольших городов/районов, у которых нет
    собственных исполнителей: прямое совпадение item_location_id ==
    search_location_id для них никогда не сработает, зато почти всегда
    есть несколько одних и тех же доминирующих "хабов", которые их
    фактически обслуживают (см. докстринг модуля выше -- для 44%
    различных search_location_id самая частая мода НЕ совпадает с самим
    значением). Для "нормальных" локаций, где сами люди обычно находят
    исполнителя в своём же городе, топ-адресат естественным образом
    совпадает с самим search_location_id, так что отдельно выделять эти
    два случая не нужно.

    Берём НЕСКОЛЬКО адресатов (не только самый частый), потому что у
    многих локаций исторические поиски делятся между несколькими
    соседними хабами почти поровну -- top-1 в таком случае ловит только
    часть случаев (см. докстринг модуля выше, "ЧЕТВЁРТЫЙ СЛОЙ").

    `confidence` каждого адресата -- это доля исторических поисков из
    данной локации, которые действительно к нему привели (value_counts,
    делённый на общее число). Используется, чтобы не доверять редким
    "хвостовым" адресатам так же сильно, как доминирующим (см. докстринг
    модуля выше).

    Возвращает pd.Series: search_location_id -> list[(item_location_id,
    confidence)], отсортированный по убыванию confidence.
    """
    grouped = item_location_series.groupby(search_location_series)
    result = {}
    for loc, group in grouped:
        vc = group.value_counts(normalize=True)
        result[loc] = list(zip(vc.index[:top_n].tolist(), vc.iloc[:top_n].tolist()))
    return pd.Series(result, dtype=object)


def rank_all(
    query_texts,
    qtext_norm_list,
    bm25_index,
    item_ids_sorted,
    item_microcat,
    qtext_to_items,
    qtext_to_microcat,
    item_location=None,
    search_location_list=None,
    search_location_redirect_targets=None,
    k=50,
    alpha_microcat=0.5,
    alpha_location=0.0,
    memo_bonus=1e6,
    chunk_size=200,
):
    """Удобная обёртка: считает скор `query_texts` относительно
    `bm25_index` порциями, безопасными по памяти (см.
    BM25Index.score_chunked), и применяет `boosted_top_k` к каждой
    порции, возвращая объединённые ранжированные списки item_id по
    каждому запросу в исходном порядке.

    item_location / search_location_list / search_location_redirect_targets
    -- см. boosted_top_k; можно оставить None (или alpha_location=0.0),
    если буст по локации не нужен."""
    results = []
    for start in range(0, len(query_texts), chunk_size):
        chunk_texts = query_texts[start:start + chunk_size]
        chunk_qtext = qtext_norm_list[start:start + chunk_size]
        chunk_loc = (
            search_location_list[start:start + chunk_size]
            if search_location_list is not None else None
        )
        chunk_loc_targets = (
            search_location_redirect_targets[start:start + chunk_size]
            if search_location_redirect_targets is not None else None
        )
        scores_chunk = next(bm25_index.score_chunked(chunk_texts, chunk_size=len(chunk_texts)))
        results.extend(boosted_top_k(
            chunk_qtext, scores_chunk, item_ids_sorted, item_microcat,
            qtext_to_items, qtext_to_microcat,
            item_location=item_location, search_location_list=chunk_loc,
            search_location_redirect_targets=chunk_loc_targets,
            k=k, alpha_microcat=alpha_microcat, alpha_location=alpha_location,
            memo_bonus=memo_bonus,
        ))
    return results


def boosted_top_k(
    qtext_norm_list,
    bm25_scores_csr,
    item_ids_sorted,
    item_microcat,
    qtext_to_items,
    qtext_to_microcat,
    item_location=None,
    search_location_list=None,
    search_location_redirect_targets=None,
    k=50,
    alpha_microcat=0.5,
    alpha_location=0.0,
    memo_bonus=1e6,
):
    """
    Параметры:
      qtext_norm_list       -- list[str], нормализованный текст запроса
                                для каждой строки (см. normalize_query_text)
      bm25_scores_csr        -- разреженная CSR-матрица (n_queries x n_items)
                                BM25-скоров
      item_ids_sorted        -- np.array с item_id, ОТСОРТИРОВАННЫЙ по
                                возрастанию, согласованный по индексам со
                                столбцами bm25_scores_csr (нужен для
                                np.searchsorted при добавлении
                                меморизационных объявлений)
      item_microcat          -- np.array с item_microcat_id, согласованный
                                по индексам с item_ids_sorted
      qtext_to_items         -- pd.Series: текст запроса -> список item_id
                                (результат build_memo_prior; пустой Series,
                                если меморизация не нужна)
      qtext_to_microcat      -- pd.Series: текст запроса -> {microcat_id: доля}
      item_location          -- np.array с item_location_id, согласованный
                                по индексам с item_ids_sorted (нужен, только
                                если alpha_location > 0)
      search_location_list   -- list, search_location_id для каждой строки
                                запроса (нужен, только если alpha_location > 0)
      search_location_redirect_targets -- list, для каждой строки запроса
                                -- список [(item_location_id, confidence), ...]
                                (результат build_location_redirect; можно
                                не передавать, тогда буст сработает только
                                по прямому совпадению)
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
        max_v = vals.max() if len(vals) else 1.0

        # --- Буст по локации: главный сигнал в этом решении (см.
        # докстринг модуля). Прямое совпадение (search_location_list)
        # получает полный буст -- это достоверный факт из данных
        # объявления, а не оценка. Совпадение с одним из "редирект"-
        # адресатов (search_location_redirect_targets) получает буст,
        # взвешенный его собственной уверенностью -- иначе мы бы
        # одинаково доверяли надёжному и ненадёжному редиректу. Буст
        # затрагивает только объявления, которые BM25 и так уже нашёл по
        # тексту (в cols) -- он переупорядочивает уже отобранный
        # кандидат-пул, а не расширяет его за пределы текстового
        # пересечения. ---
        if alpha_location > 0 and len(cols) and item_location is not None:
            cand_loc = item_location[cols]
            raw_match = cand_loc == search_location_list[i]
            if raw_match.any():
                # При alpha_location=1.0 буст уже гарантированно выводит
                # все объявления из подходящей локации выше любых
                # объявлений из других локаций для этого запроса (т.к.
                # добавка равна максимальному скору запроса, а
                # раздвигаемые значения на неё не превышают исходный
                # максимум). Дальнейшее увеличение alpha_location ничего
                # не меняет -- офлайн-эксперимент подтвердил плато
                # Recall@50 от alpha=1.0 до 50.
                vals[raw_match] += alpha_location * max_v
            if search_location_redirect_targets is not None:
                already = raw_match
                targets = search_location_redirect_targets[i]
                if targets:
                    for target_loc, confidence in targets:
                        m = (cand_loc == target_loc) & ~already
                        if m.any():
                            vals[m] += confidence * alpha_location * max_v
                            already = already | m

        # --- Буст по микрокатегории: см. докстринг модуля -- вреден без
        # буста по локации, полезен вместе с ним. ---
        microcat_prior = (
            qtext_to_microcat.get(qtext) if qtext in qtext_to_microcat.index else None
        )
        if microcat_prior:
            cand_microcats = item_microcat[cols] if len(cols) else np.array([])
            for mc, p in microcat_prior.items():
                boost_mask = cand_microcats == mc
                if boost_mask.any():
                    vals[boost_mask] += alpha_microcat * p * max_v

        # --- Меморизация точного исторического объявления (в финальном
        # пайплайне отключена -- см. докстринг модуля; qtext_to_items
        # обычно пустой Series) ---
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

        # --- Финальный отбор топ-k по итоговому (BM25 + бусты) скору ---
        if len(vals) > k:
            part = np.argpartition(vals, -k)[-k:]
            cols, vals = cols[part], vals[part]
        order = np.argsort(-vals)
        results.append(item_ids_sorted[cols[order]])
    return results
