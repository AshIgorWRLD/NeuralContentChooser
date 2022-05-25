from recommenders import prediction
from utils import printer


def recommend(metadata, titles_as_indices, cosine_similarity_matrix, films):
    for film in films:
        printer.print_data("Сгенерированные предсказания для '" + film + "'",
                           prediction.get_recommendations(metadata, titles_as_indices, cosine_similarity_matrix,
                                                          film))
