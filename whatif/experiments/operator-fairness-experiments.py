from whatif.example_pipelines.amazon_reviews import execute_review_pipeline
from whatif.example_pipelines.income_classifier import execute_income_pipeline
from whatif.example_pipelines.paper_example import execute_paper_example_pipeline

execute_review_pipeline()
execute_paper_example_pipeline()  # TODO: Which review pipeline should we use?

execute_income_pipeline()
