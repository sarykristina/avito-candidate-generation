"""
Минималистичная, векторизованная реализация BM25 поверх разреженных
(sparse) матриц scipy.

Мы намеренно не используем пакет `rank_bm25` с PyPI: он считает скор
запрос-документ обычным Python-циклом по каждому документу корпуса, что
слишком медленно для ~190 тыс. объявлений x ~2.5 тыс. запросов (реальный
бенчмарк) и тем более для ~345 тыс. объявлений при офлайн-валидации на
train.parquet. Всё, что ниже, -- это классический BM25 (формула
Robertson & Sparck Jones), просто выраженный как матричная алгебра на
разреженных матрицах, поэтому весь расчёт занимает секунды через
scipy/numpy вместо минут/часов на чистом Python.

Формула (Okapi BM25):
    score(q, d) = sum_{t in q} IDF(t) * f(t,d)*(k1+1) /
                  (f(t,d) + k1*(1 - b + b*|d|/avgdl))

    IDF(t) = log(1 + (N - n_t + 0.5) / (n_t + 0.5))

где f(t,d) -- "сырая" частота термина t в документе d, |d| -- длина
документа (число токенов), avgdl -- средняя длина документа по корпусу,
N -- число документов, n_t -- число документов, содержащих термин t.

k1 и b -- стандартные гиперпараметры BM25: k1 отвечает за насыщение
(насколько быстро повторное вхождение термина перестаёт добавлять вес),
b -- за силу нормировки по длине документа. Используются
общепринятые в литературе значения по умолчанию (k1=1.5, b=0.75), без
отдельного тюнинга под этот датасет -- в отличие от весов полей
(ITEM_FIELD_WEIGHTS), которые заметно влияют на recall и поэтому
подбирались отдельно (см. src/data_prep.py).
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, min_df: int = 2):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(min_df=min_df, dtype=np.float32)
        self.bm25_matrix = None   # (n_docs, n_terms) -- BM25-веса документов, разреженная матрица
        self.idf_ = None          # (n_terms,) -- IDF каждого термина словаря

    def fit(self, doc_texts):
        """Обучить словарь и посчитать BM25-веса на корпусе объявлений.

        doc_texts -- pd.Series/список строк, по одной bag-of-words
        строке на объявление (результат build_item_corpus_text)."""
        # Шаг 1: обычный подсчёт "сырых" частот термина в документе (TF)
        # через CountVectorizer -- это даёт разреженную матрицу
        # (n_docs x n_terms), где значение [d, t] -- сколько раз термин t
        # встретился в документе d.
        tf = self.vectorizer.fit_transform(doc_texts).tocsr()
        n_docs, n_terms = tf.shape

        # Длина документа = сумма всех вхождений термов в строке.
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean()

        # Документная частота термина (df) = число документов, где он
        # встретился хотя бы раз = число ненулевых значений в столбце.
        # tf.tocsc().indptr даёт границы столбцов в CSC-представлении,
        # разница соседних границ -- это как раз количество ненулевых
        # элементов в каждом столбце.
        df = np.diff(tf.tocsc().indptr)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.idf_ = idf.astype(np.float32)

        # Множитель нормировки по длине документа (часть знаменателя
        # BM25): длинные документы дают меньший вес каждому отдельному
        # вхождению термина. Считаем его один раз на документ, а затем
        # "размножаем" на каждый ненулевой элемент этой строки через
        # CSR-указатели строк (indptr) -- это позволяет обойтись без
        # Python-цикла по документам.
        len_norm = (1.0 - self.b + self.b * doc_len / avgdl).astype(np.float32)
        row_of_nnz = np.repeat(np.arange(n_docs), np.diff(tf.indptr))

        # Собственно формула BM25 для каждого ненулевого элемента (d, t):
        f = tf.data
        denom = f + self.k1 * len_norm[row_of_nnz]
        tf_weight = f * (self.k1 + 1.0) / denom
        bm25_data = tf_weight * self.idf_[tf.indices]

        # Собираем итоговую разреженную матрицу BM25-весов с той же
        # структурой ненулевых позиций (indices/indptr), что и у tf --
        # меняются только сами значения.
        self.bm25_matrix = sparse.csr_matrix(
            (bm25_data, tf.indices, tf.indptr), shape=tf.shape
        )
        return self

    def transform_query(self, query_texts):
        """Перевести тексты запросов в векторы "сырых" частот термина в
        словаре корпуса объявлений (тем же CountVectorizer, что и для
        документов -- .transform(), а не .fit_transform(), чтобы словарь
        не менялся и оставался согласован со стороной объявлений)."""
        return self.vectorizer.transform(query_texts).tocsr()

    def score(self, query_texts):
        """Вернуть разреженную матрицу BM25-скоров (n_queries x n_docs).

        Скор запроса к документу в BM25 -- это сумма по общим термам
        IDF(term) * BM25_вес_термина_в_документе, что ровно равно
        скалярному произведению вектора "сырых" частот запроса на
        BM25-взвешенную матрицу документов (self.bm25_matrix), потому
        что IDF уже "зашит" в bm25_matrix при fit()."""
        q = self.transform_query(query_texts)
        return q @ self.bm25_matrix.T


def top_k_per_row(score_matrix: sparse.csr_matrix, k: int):
    """Для разреженной матрицы скоров (n_rows x n_cols) вернуть для
    каждой строки индексы столбцов её k наибольших значений (по убыванию),
    ни разу не превращая всю матрицу в плотную (dense)."""
    score_matrix = score_matrix.tocsr()
    n_rows = score_matrix.shape[0]
    results = []
    for i in range(n_rows):
        start, end = score_matrix.indptr[i], score_matrix.indptr[i + 1]
        cols = score_matrix.indices[start:end]
        vals = score_matrix.data[start:end]
        if len(vals) > k:
            # argpartition быстрее полной сортировки: он гарантирует
            # только то, что k наибольших элементов окажутся в конце
            # массива (в произвольном порядке между собой), чего
            # достаточно, чтобы отсечь "хвост".
            part = np.argpartition(vals, -k)[-k:]
            cols, vals = cols[part], vals[part]
        order = np.argsort(-vals)
        results.append(cols[order])
    return results
