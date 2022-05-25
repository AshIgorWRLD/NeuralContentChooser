import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from recommenders import recommend_realiser
from utils import printer


def get_plot_description(metadata):
    printer.print_data("Примеры описания сюжета на основе первых 5 фильмов", metadata['overview'].head(5))


def get_metric_and_remove_unnecessary_words(metadata):
    # Задание TF-IDF метрики для анализа слов в датасете и избавление от вспомогательных слов по типу 'a', 'on', 'the'
    tfidf = TfidfVectorizer(stop_words='english')

    # Замена NaN чисел на пустую строку
    metadata['overview'] = metadata['overview'].fillna('')
    return metadata, tfidf


def prepare_data_to_analise(metadata):
    metadata, tfidf = get_metric_and_remove_unnecessary_words(metadata)

    # Задание матрицы на основе нашей метрики
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])
    return metadata, tfidf, tfidf_matrix


def print_ten_words_example(tfidf, start_idx, message):
    printer.print_data("Примеры " + message, tfidf.get_feature_names()[start_idx:start_idx + 10])


def show_calculated_word_statistic(tfidf):
    print_ten_words_example(tfidf, 200, "цифр")
    print_ten_words_example(tfidf, 4000, "слов на букву 'A'")
    print_ten_words_example(tfidf, 24000, "слов на букву 'F'")


def calculate_cosine_simulation_matrix(tfidf_matrix):
    cosine_similarity_matrix = linear_kernel(tfidf_matrix.astype(np.float32), tfidf_matrix.astype(np.float32))

    printer.print_data("Примеры коэффициентов косинусоидальной схожести", cosine_similarity_matrix[1][:10])
    return cosine_similarity_matrix


def create_cosine_similarity_matrix(metadata):
    get_plot_description(metadata)

    metadata, tfidf, tfidf_matrix = prepare_data_to_analise(metadata)

    show_calculated_word_statistic(tfidf)

    cosine_similarity_matrix = calculate_cosine_simulation_matrix(tfidf_matrix)
    return metadata, cosine_similarity_matrix


def run(metadata, titles_as_indices, films_to_recommend):
    metadata, cosine_similarity_matrix = create_cosine_similarity_matrix(metadata)

    recommend_realiser.recommend(metadata, titles_as_indices, cosine_similarity_matrix, films_to_recommend)
