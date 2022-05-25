import dataset_parser
from recommenders import advanced_content_based_recommender, content_based_recommender, simple_recommender
from utils.converters import title_mapper

metadata, cast_credits, keywords = dataset_parser.get_metadata()
titles_as_indices = title_mapper.get_titles_as_indices(metadata)

simple_recommender.run(metadata)

films_to_analise = ['The Lord of the Rings: The Return of the King', 'The Matrix', 'Interstellar']

content_based_recommender.run(metadata, titles_as_indices, films_to_analise)

advanced_content_based_recommender.run(metadata, cast_credits, keywords, films_to_analise)
