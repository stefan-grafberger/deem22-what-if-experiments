import pandas as pd

from jenga.corruptions.generic import CategoricalShift, MissingValues
from jenga.corruptions.numerical import Scaling
from jenga.corruptions.text import BrokenCharacters
from sklearn.metrics import f1_score, precision_score, roc_auc_score

from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from whatif.utils.utils import get_project_root


def execute_review_pipeline(corrupt_train, corrupt_test, corruption_fraction):
    target_categories = ['Digital_Video_Games']
    split_date = '2015-07-31'
    start_date = '2015-01-01'

    reviews = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/reviews.csv.gz',
                          compression='gzip', index_col=0)
    products = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/products.csv',
                           index_col=0)
    categories = pd.read_csv(
        f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/categories.csv', index_col=0)
    ratings = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/amazon-reviews/ratings.csv',
                          index_col=0)

    reviews = reviews[reviews.verified_purchase == 'Y']
    reviews = reviews[reviews.marketplace == 'US']
    reviews = reviews[reviews.review_date >= start_date]

    reviews_with_ratings = reviews.merge(ratings, on='review_id')

    categories_of_interest = categories[categories.category.isin(target_categories)]
    products_of_interest = products.merge(left_on='category_id', right_on='id', right=categories_of_interest)

    reviews_with_products_and_ratings = reviews_with_ratings.merge(products_of_interest, on='product_id')

    train_data = reviews_with_products_and_ratings[reviews_with_products_and_ratings.review_date <= split_date].copy()
    test_data = reviews_with_products_and_ratings[reviews_with_products_and_ratings.review_date > split_date].copy()

    if corrupt_train is True:
        train_data = corrupt_data(train_data, corruption_fraction)
    if corrupt_test is True:
        test_data = corrupt_data(test_data, corruption_fraction)

    train_data['product_title'] = train_data['product_title'].fillna(value='')
    test_data['product_title'] = test_data['product_title'].fillna(value='')

    train_data['review_headline'] = train_data['review_headline'].fillna(value='')
    test_data['review_headline'] = test_data['review_headline'].fillna(value='')

    train_data['review_body'] = train_data['review_body'].fillna(value='')
    test_data['review_body'] = test_data['review_body'].fillna(value='')

    train_data['title_and_review_text'] = train_data.product_title + ' ' + train_data.review_headline + ' ' + \
                                            train_data.review_body
    test_data['title_and_review_text'] = test_data.product_title + ' ' + test_data.review_headline + ' ' + \
                                            test_data.review_body

    train_data['is_helpful'] = train_data['helpful_votes'] > 0
    train_labels = label_binarize(train_data['is_helpful'], classes=[True, False])

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
    test_predict = model.predict(test_data)
    # Potential error with corruptions: Only one class present in y_true
    # score = roc_auc_score(test_predict, test_labels)
    # print(f'AUC Score on the test set: {score}')
    score = f1_score(test_predict, test_labels, average='macro')
    print(f'F1 Score on the test set: {score}')

    return score


def corrupt_data(data, corruption_fraction):
    # Data corruptions, later: test them one by one
    print("Corrupting vine...")
    data = CategoricalShift(column='vine', fraction=corruption_fraction).transform(data)

    print("Corrupting verified_purchase...")
    data = CategoricalShift(column='verified_purchase', fraction=corruption_fraction).transform(data)

    print("Corrupting category_id...")
    data['category_id'] = data['category_id'].astype(str)  # CategoricalShift implemented only for categorical variables
    data = CategoricalShift(column='category_id', fraction=corruption_fraction).transform(data)
    data['category_id'] = data['category_id'].astype(int)

    print("Corrupting star_rating...")
    data = Scaling(column='star_rating', fraction=corruption_fraction).transform(data)

    print("Corrupting product_title...")
    data = BrokenCharacters(column='product_title', fraction=corruption_fraction).transform(data)

    print("Corrupting review_headline...")
    data = BrokenCharacters(column='review_headline', fraction=corruption_fraction).transform(data)

    print("Corrupting review_body...")
    data = MissingValues(column='review_body', fraction=corruption_fraction).transform(data)

    return data

print("No corruptions")
execute_review_pipeline(False, False, 0.9)
print("Corruptions in Test")
execute_review_pipeline(False, True, 0.9)
print("Corruptions in Train and test")
execute_review_pipeline(True, True, 0.9)
