from recommenders import advanced_content_based_recommender, content_based_recommender, simple_recommender
import dataset_parser
from utils.converters import title_mapper

metadata, cast_credits, keywords = dataset_parser.get_metadata()
titles_as_indices = title_mapper.get_titles_as_indices(metadata)

simple_recommender.run(metadata)

content_based_recommender.run(metadata, titles_as_indices)

advanced_content_based_recommender.run(metadata, cast_credits, keywords)
