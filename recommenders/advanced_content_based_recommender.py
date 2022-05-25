from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommenders import recommend_realiser
from utils import merger, printer
from utils.converters import data_feature_converter, data_type_converter, my_data_format_converter


def join_columns(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def refactor_data(metadata, cast, keywords):
    # удаляем ряды данных с ненужными нам ID
    metadata = metadata.drop([19730, 29503, 35587])

    # приводим ID к типу int
    metadata = data_type_converter.convert(metadata, 'id', 'int')
    cast = data_type_converter.convert(cast, 'id', 'int')
    keywords = data_type_converter.convert(keywords, 'id', 'int')

    # добавляем к нашему датасету каст фильма и ключевые слова
    metadata = merger.merge(metadata, cast, 'id')
    metadata = merger.merge(metadata, keywords, 'id')

    printer.print_data("Пример преобразованного набора данных", metadata.head(5))

    metadata = data_feature_converter.convert_array(metadata, ['cast', 'crew', 'keywords', 'genres'], literal_eval)

    metadata = data_feature_converter.convert_part(metadata, 'crew', 'director', my_data_format_converter.get_director)

    metadata = data_feature_converter.convert_array(metadata, ['cast', 'keywords', 'genres'],
                                                    my_data_format_converter.get_list)

    printer.print_data("Пример новых данных, занесенных в датасет",
                       metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(5))

    metadata = data_feature_converter.convert_array(metadata, ['cast', 'keywords', 'director', 'genres'],
                                                    my_data_format_converter.clean_data)

    metadata = data_feature_converter.convert_array_with_axis(metadata, ['keys'], join_columns, 1)

    printer.print_data("Пример новых данных основанных на ключевых словах, занесенных в датасет",
                       metadata[['keys']].head(5))
    return metadata


def create_cosine_similarity_matrix(metadata):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['keys'])

    cosine_similarity_matrix = cosine_similarity(count_matrix.astype(np.float32), count_matrix.astype(np.float32))
    return cosine_similarity_matrix


def reset_indexes(metadata):
    metadata = metadata.reset_index()
    new_titles_as_indices = pd.Series(metadata.index, index=metadata['title'])
    return metadata, new_titles_as_indices


def run(metadata, cast, keywords, films_to_recommend):
    metadata = refactor_data(metadata, cast, keywords)

    cosine_similarity_matrix = create_cosine_similarity_matrix(metadata)

    # Создание обратного отображения
    metadata, new_titles_as_indices = reset_indexes(metadata)

    recommend_realiser.recommend(metadata, new_titles_as_indices, cosine_similarity_matrix, films_to_recommend)
