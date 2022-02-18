import pandas as pd


from jenga.corruptions.generic import CategoricalShift, MissingValues
from jenga.corruptions.numerical import Scaling
from jenga.corruptions.text import BrokenCharacters

from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from whatif.utils.utils import get_project_root


def execute_review_pipeline(corrupt_test_only, corruption_fraction):
    target_categories = ['Digital_Video_Games']
    split_date = '2015-07-31'
    start_date = '2015-01-01'

    reviews = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/reviews.csv.gz', compression='gzip', index_col=0)
    products = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/products.csv', index_col=0)
    categories = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/categories.csv', index_col=0)
    ratings = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/ratings.csv', index_col=0)

    if corrupt_test_only is False:
        products, ratings, reviews = corrupt_train_and_test_data(products, ratings, reviews, corruption_fraction)

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
        ('learner', SGDClassifier(loss='log', penalty='l1', max_iter=1000))])

    model = pipeline.fit(train_data, train_labels)

    if corrupt_test_only is True:
        test_data = corrupt_test_data(test_data, corruption_fraction)
    score = model.score(test_data, test_labels)

    print(f'Accuracy on the test set: {score}')
    return score


def corrupt_train_and_test_data(products, ratings, reviews, corruption_fraction):
    # Data corruptions, later: test them one by one
    print("Corrupting vine train and test...")
    reviews['vine'] = reviews['vine'].astype(str)  # CategoricalShift implemented only for categorical variables
    reviews = CategoricalShift(column='vine', fraction=corruption_fraction).transform(reviews)
    print("Corrupting verified_purchase train and test...")
    reviews = CategoricalShift(column='verified_purchase', fraction=corruption_fraction).transform(reviews)
    print("Corrupting product_title train and test...")
    products = BrokenCharacters(column='product_title', fraction=corruption_fraction).transform(products)
    print("Corrupting review_headline train and test...")
    reviews = MissingValues(column='review_headline', fraction=corruption_fraction).transform(reviews)
    print("Corrupting review_body train and test...")
    reviews = MissingValues(column='review_body', fraction=corruption_fraction).transform(reviews)
    print("Corrupting star_rating train and test...")
    ratings = Scaling(column='star_rating', fraction=corruption_fraction).transform(ratings)
    print("Corrupting category_id train and test...")
    products['category_id'] = products['category_id'].astype(
        str)  # CategoricalShift implemented only for categorical variables
    products = CategoricalShift(column='category_id', fraction=corruption_fraction).transform(products)
    products['category_id'] = products['category_id'].astype(int)
    return products, ratings, reviews


def corrupt_test_data(test_data, corruption_fraction):
    # Data corruptions, later: test them one by one
    print("Corrupting vine train and test...")
    test_data['vine'] = test_data['vine'].astype(str)  # CategoricalShift implemented only for categorical variables
    test_data = CategoricalShift(column='vine', fraction=corruption_fraction).transform(test_data)
    print("Corrupting verified_purchase train and test...")
    test_data = CategoricalShift(column='verified_purchase', fraction=corruption_fraction).transform(test_data)
    print("Corrupting product_title train and test...")
    test_data = BrokenCharacters(column='product_title', fraction=corruption_fraction).transform(test_data)
    print("Corrupting review_headline train and test...")
    test_data = MissingValues(column='review_headline', fraction=corruption_fraction).transform(test_data)
    print("Corrupting review_body train and test...")
    test_data = MissingValues(column='review_body', fraction=corruption_fraction).transform(test_data)
    print("Corrupting star_rating train and test...")
    test_data = Scaling(column='star_rating', fraction=corruption_fraction).transform(test_data)
    print("Corrupting category_id train and test...")
    test_data['category_id'] = test_data['category_id'].astype(
        str)  # CategoricalShift implemented only for categorical variables
    test_data = CategoricalShift(column='category_id', fraction=corruption_fraction).transform(test_data)
    test_data['category_id'] = test_data['category_id'].astype(int)
    return test_data


print("Train and test")
execute_review_pipeline(False, 0.5)
print("Test")
execute_review_pipeline(True, 0.5)
