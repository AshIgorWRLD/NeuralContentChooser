from utils import printer


def imdb_top_form_formula(metadata):
    average_rating = metadata['vote_average'].mean()
    printer.print_data("Cредний рейтинг фильмов в датасете", average_rating)
    min_estimation_number = metadata['vote_count'].quantile(0.90)
    printer.print_data("Минимальное количество оценок пользователей", min_estimation_number)

    # отфильтруем нужные нам фильмы из данного датасета в новый
    filtered_metadata = metadata.copy().loc[metadata['vote_count'] >= min_estimation_number]

    # получение IMDB формулы подсчета рейтинга фильмов
    def weighted_imdb_formula_rating(data, m=min_estimation_number, c=average_rating):
        v = data['vote_count']
        R = data['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * c)

    # добавляем столбец 'score' в который записываем рейтинг, посчитанный нашей формулой imb
    filtered_metadata['score'] = filtered_metadata.apply(weighted_imdb_formula_rating, axis=1)

    printer.print_data("Датасет фильмов с рейтингом и необходимым минимумом оценок",
                       filtered_metadata[['title', 'vote_count', 'vote_average', 'score']])
    return filtered_metadata


def run(metadata):
    metadata_with_rating = imdb_top_form_formula(metadata)

    # сортировка фильмов на основе полученных значений рейтинга
    sorted_metadata_with_rating = metadata_with_rating.sort_values('score', ascending=False)

    # Печатаем 15 фильмов с наивысшим показателем рейтинга
    printer.print_data("Топ 15 фильмов по версии нашей формулы",
                       sorted_metadata_with_rating[['title', 'vote_count', 'vote_average', 'score']].head(15))
