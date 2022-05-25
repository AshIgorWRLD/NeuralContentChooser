def get_recommendations(metadata, titles_as_indices, cosine_similarity_matrix, title):
    # Получаем иднекс фильма, соответствующий названию
    idx = titles_as_indices[title]

    # Анализируем схожесть остальных фильмов с заданным
    similarity_scores = list(enumerate(cosine_similarity_matrix[idx]))

    # Сортируем фильмы на основе схожести с заданным
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Оставляем первые 10 максимально похожих фильмов
    similarity_scores = similarity_scores[1:11]

    # Получаем индексы тех 10 отобранных фильмов
    movie_indices = [i[0] for i in similarity_scores]

    # Возвращаем найденную 10-ку похожих фильмов
    return metadata['title'].iloc[movie_indices]
