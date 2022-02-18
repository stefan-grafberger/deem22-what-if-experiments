import pandas as pd
import sys

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from whatif.utils.utils import get_project_root


def execute_review_pipeline():
    target_categories = ['Digital_Video_Games']
    split_date = '2015-07-31'
    start_date = '2015-01-01'

    if len(sys.argv) > 1:
        target_categories = [sys.argv[1]]

    if len(sys.argv) > 2:
        split_date = sys.argv[2]

    if len(sys.argv) > 3:
        start_date = sys.argv[3]

    reviews = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/reviews.csv.gz', compression='gzip', index_col=0)
    products = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/products.csv', index_col=0)
    categories = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/categories.csv', index_col=0)
    ratings = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/ratings.csv', index_col=0)

    reviews = reviews[reviews.verified_purchase == 'Y']
    reviews = reviews[reviews.marketplace == 'US']
    reviews = reviews[reviews.review_date >= start_date]

    reviews_with_ratings = reviews.merge(ratings, on='review_id')

    categories_of_interest = categories[categories.category.isin(target_categories)]
    products_of_interest = products.merge(left_on='category_id', right_on='id', right=categories_of_interest)

    reviews_with_products_and_ratings = reviews_with_ratings.merge(products_of_interest, on='product_id')

    reviews_with_products_and_ratings['product_title'] = \
        reviews_with_products_and_ratings['product_title'].fillna(value='')

    reviews_with_products_and_ratings['review_headline'] = \
        reviews_with_products_and_ratings['review_headline'].fillna(value='')

    reviews_with_products_and_ratings['review_body'] = \
        reviews_with_products_and_ratings['review_body'].fillna(value='')

    reviews_with_products_and_ratings['title_and_review_text'] = \
        reviews_with_products_and_ratings.product_title + ' ' + \
        reviews_with_products_and_ratings.review_headline + ' ' + \
        reviews_with_products_and_ratings.review_body

    train_data = reviews_with_products_and_ratings[reviews_with_products_and_ratings.review_date <= split_date].copy()
    train_data['is_helpful'] = train_data['helpful_votes'] > 0
    train_labels = label_binarize(train_data['is_helpful'], classes=[True, False])

    test_data = reviews_with_products_and_ratings[reviews_with_products_and_ratings.review_date > split_date].copy()
    test_data['is_helpful'] = test_data['helpful_votes'] > 0
    test_labels = label_binarize(test_data['is_helpful'], classes=[True, False])

    numerical_attributes = ['star_rating']
    categorical_attributes = ['vine', 'verified_purchase', 'category_id']

    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_attributes),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_attributes),
        ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100), 'title_and_review_text')
    ])

    pipeline = Pipeline([
        ('features', feature_transformation),
        ('learner', SGDClassifier(loss='log', penalty='l1', max_iter=1000, class_weight="balanced"))])

    model = pipeline.fit(train_data, train_labels.ravel())

    test_predict = model.predict(test_data)

    score = roc_auc_score(test_predict, test_labels, average='macro')
    print(f'AUC Score on the test set: {score}')
    score = f1_score(test_predict, test_labels, average='macro')
    print(f'F1 Score on the test set: {score}')


execute_review_pipeline()
