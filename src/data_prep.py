"""
Общие функции загрузки данных / построения признаков, которыми
пользуются и офлайн-валидация (scripts/run_validation.py), и финальная
генерация ответа (scripts/generate_answer.py). Вынесены в один модуль,
чтобы эти два пути НЕ могли незаметно разойтись между собой: если бы
текст для BM25 собирался по-разному в валидации и в финальном прогоне,
офлайн-метрика могла бы не отражать реальное качество на бенчмарке.
"""

import pandas as pd
from src.text_utils import build_weighted_text

# Насколько сильно (в виде количества повторов токенов) каждое поле
# участвует при склейке заголовка / описания / структурированных
# параметров в один bag-of-words. Заголовок -- самый сильный сигнал для
# короткого запроса (запрос "баня на дровах" почти всегда дословно
# повторяется в заголовке объявления). Структурированные параметры
# (item_infm_params_text) -- короткое, но очень информативное поле
# (вид услуги / тип услуги и т.п.), поэтому у него тоже повышенный вес.
# Свободное описание всё ещё даёт реальный сигнал -- полное исключение
# описания стоило 0.03 Recall@50 в офлайн-валидации, -- но это самое
# "шумное" поле (длинное, содержит рекламные обороты речи), поэтому у
# него минимальный вес.
#
# Конкретные числа (5 / 3 / 1) подобраны перебором по сетке на
# офлайн-валидации против train.parquet (см. scripts/run_validation.py
# и README.md): Recall@50 вырос с 0.182 при весах 3/2/1 до ~0.185 при
# 5/3/1, а затем вышел на плато (7/4/1 и 10/5/1 прироста уже не дали).
# То есть 5/3/1 -- это минимальные веса, при которых достигается плато.
ITEM_FIELD_WEIGHTS = {
    "item_title_raw": 5,
    "item_infm_params_text": 3,
    "item_description_raw": 1,
}
# Веса на стороне запроса зеркалят веса объявления: search_query --
# аналог заголовка, search_infm_params_text -- аналог структурированных
# параметров (это фильтры, которые пользователь выбрал при поиске).
QUERY_FIELD_WEIGHTS = {
    "search_query": 5,
    "search_infm_params_text": 3,
}


def normalize_query_text(s: pd.Series) -> pd.Series:
    """Нормализация текста запроса для использования как ключа словаря
    (в исторических prior'ах из train.parquet): нижний регистр + обрезка
    пробелов по краям. NaN -> пустая строка, чтобы .str-методы не падали
    и запросы без текста не ломали группировку."""
    return s.fillna("").str.lower().str.strip()


def build_item_corpus_text(items: pd.DataFrame) -> pd.Series:
    """Построить для каждого объявления одну bag-of-words строку из
    заголовка, структурированных параметров и описания (см.
    ITEM_FIELD_WEIGHTS выше). Результат -- вход для CountVectorizer
    внутри BM25Index.fit()."""
    def _row_text(row):
        parts = [(row[f], w) for f, w in ITEM_FIELD_WEIGHTS.items()]
        return build_weighted_text(parts)
    return items.apply(_row_text, axis=1)


def build_query_text(queries: pd.DataFrame) -> pd.Series:
    """То же самое, но для запросов: search_query + search_infm_params_text
    (см. QUERY_FIELD_WEIGHTS выше)."""
    def _row_text(row):
        parts = [(row[f], w) for f, w in QUERY_FIELD_WEIGHTS.items()]
        return build_weighted_text(parts)
    return queries.apply(_row_text, axis=1)
