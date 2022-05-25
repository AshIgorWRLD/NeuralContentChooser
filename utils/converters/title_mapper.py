import pandas as pd

from utils import printer


def map_titles(metadata):
    # Создаем обратную мапу индексов и названий фильмов
    titles_as_indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

    printer.print_data("Пример соответствия названий фильмов индексам замапленного массива", titles_as_indices[:10])
    return titles_as_indices


def get_titles_as_indices(metadata):
    titles_as_indices = map_titles(metadata)
    return titles_as_indices
