import pandas as pd

from utils import printer


def get_metadata():
    metadata = pd.read_csv('input-data/movies_metadata.csv', low_memory=False)
    printer.print_data("Изначальный датасет фильмов", metadata.head(3))

    cast_credits = pd.read_csv('input-data/credits.csv')
    printer.print_data("Датасет информации о касте", cast_credits.head(3))

    keywords = pd.read_csv('input-data/keywords.csv')
    printer.print_data("Датасет ключевых слов", keywords.head(3))

    return metadata, cast_credits, keywords
