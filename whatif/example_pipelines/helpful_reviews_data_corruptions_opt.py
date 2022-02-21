import timeit
from inspect import cleandoc

import numpy
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


def execute_review_pipeline_opt(debug):
    iteration_results = {
        "test_corruption": [],
        "train_corruption": [],
        "corruption_fraction": [],
        "feature": [],
        "roc_auc": [],
        "f1": []
    }

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

    if False is True:  # TODO
        train_data = corrupt_data(train_data, corruption_fraction, corrupt_feature)
    if False is True:  # TODO
        test_data = corrupt_data(test_data, corruption_fraction, corrupt_feature)

    train_data['product_title'] = train_data['product_title'].fillna(value='')
    test_data['product_title'] = test_data['product_title'].fillna(value='')

    train_data['review_headline'] = train_data['review_headline'].fillna(value='')
    test_data['review_headline'] = test_data['review_headline'].fillna(value='')

    train_data['review_body'] = train_data['review_body'].fillna(value='')
    test_data['review_body'] = test_data['review_body'].fillna(value='')

    train_data['is_helpful'] = train_data['helpful_votes'] > 0
    train_labels = label_binarize(train_data['is_helpful'], classes=[True, False])

    test_data['is_helpful'] = test_data['helpful_votes'] > 0
    test_labels = label_binarize(test_data['is_helpful'], classes=[True, False])

    star_rating_train = train_data[['star_rating']]
    star_rating_train_featurizer = StandardScaler()
    star_rating_train_featurized = star_rating_train_featurizer.fit_transform(star_rating_train)
    star_rating_test = test_data[['star_rating']]
    star_rating_test_featurized = star_rating_train_featurizer.transform(star_rating_test)

    vine_train = train_data[['vine']]
    vine_train_featurizer = OneHotEncoder(handle_unknown='ignore')
    vine_train_featurized = vine_train_featurizer.fit_transform(vine_train).toarray()
    vine_test = test_data[['vine']]
    vine_test_featurized = vine_train_featurizer.transform(vine_test).toarray()

    verified_purchase_train = train_data[['verified_purchase']]
    verified_purchase_train_featurizer = OneHotEncoder(handle_unknown='ignore')
    verified_purchase_train_featurized = verified_purchase_train_featurizer.fit_transform(verified_purchase_train).toarray()
    verified_purchase_test = test_data[['verified_purchase']]
    verified_purchase_test_featurized = verified_purchase_train_featurizer.transform(verified_purchase_test).toarray()

    category_id_train = train_data[['category_id']]
    category_id_featurizer = OneHotEncoder(handle_unknown='ignore')
    category_id_train_featurized = category_id_featurizer.fit_transform(category_id_train).toarray()
    category_id_test = test_data[['category_id']]
    category_id_test_featurized = category_id_featurizer.transform(category_id_test).toarray()

    review_headline_train = train_data['review_headline']
    title_and_review_text_train = train_data.product_title + ' ' + review_headline_train + ' ' + \
                                          train_data.review_body
    title_and_review_text_train_featurizer = HashingVectorizer(ngram_range=(1, 3), n_features=100)
    title_and_review_text_train_featurized = title_and_review_text_train_featurizer\
        .fit_transform(title_and_review_text_train).toarray()
    review_headline_test = test_data['review_headline']
    title_and_review_text_test = test_data.product_title + ' ' + review_headline_test + ' ' + \
                                  test_data.review_body
    title_and_review_text_test_featurized = title_and_review_text_train_featurizer \
        .transform(title_and_review_text_test).toarray()

    if debug is True:
        print("No corruptions")

    train_wo_corruptions = numpy.hstack([star_rating_train_featurized, vine_train_featurized,
                                             verified_purchase_train_featurized,
                              category_id_train_featurized, title_and_review_text_train_featurized])
    test_wo_corruptions = numpy.hstack([star_rating_test_featurized, vine_test_featurized,
                                             verified_purchase_test_featurized,
                              category_id_test_featurized, title_and_review_text_test_featurized])

    model_wo_corruptions = SGDClassifier(loss='log', penalty='l1', max_iter=1000, class_weight="balanced")

    model_wo_corruptions = model_wo_corruptions.fit(train_wo_corruptions, train_labels.ravel())
    test_predict = model_wo_corruptions.predict(test_wo_corruptions)
    # Potential error with corruptions: Only one class present in y_true
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict, test_labels)
    if debug is True:
        print(f'AUC Score on the test set: {scores["roc_auc"]}')
    scores['f1'] = f1_score(test_predict, test_labels, average='macro')
    if debug is True:
        print(f'F1 Score on the test set: {scores["f1"]}')

    iteration_results["test_corruption"].append(False)
    iteration_results["train_corruption"].append(False)
    iteration_results["corruption_fraction"].append(None)
    iteration_results["feature"].append(None)
    iteration_results["roc_auc"].append(scores["roc_auc"])
    iteration_results["f1"].append(scores["f1"])

    if debug is True:
        print("____")
        print(f"Now testing corruption of 20% of feature star_rating")
        print("Corruptions in Test")
    corruption_fraction = 0.2
    sampled_star_rating_rows_to_corrupt = numpy.random.permutation(star_rating_test.index)[:int(len(star_rating_test) * corruption_fraction)]
    scale_factor = numpy.random.choice([10, 100, 1000])
    star_rating_test_corrupt02 = star_rating_test.copy()
    star_rating_test_corrupt02.loc[sampled_star_rating_rows_to_corrupt, 'star_rating'] *= scale_factor
    star_rating_test_featurized_corrupt02 = star_rating_train_featurizer.transform(star_rating_test_corrupt02)

    test_star_rating_corrupt02 = numpy.hstack([star_rating_test_featurized_corrupt02, vine_test_featurized,
                                        verified_purchase_test_featurized,
                                        category_id_test_featurized, title_and_review_text_test_featurized])

    # This should fail
    # numpy.testing.assert_allclose(test_wo_corruptions, test_star_rating_corrupt02, rtol=1e-5, atol=0)
    test_predict_star_rating_corrupt02 = model_wo_corruptions.predict(test_star_rating_corrupt02)
    # Potential error with corruptions: Only one class present in y_true
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict_star_rating_corrupt02, test_labels)
    if debug is True:
        print(f'AUC Score on the test set: {scores["roc_auc"]}')
    scores['f1'] = f1_score(test_predict_star_rating_corrupt02, test_labels, average='macro')
    if debug is True:
        print(f'F1 Score on the test set: {scores["f1"]}')

    iteration_results["test_corruption"].append(True)
    iteration_results["train_corruption"].append(False)
    iteration_results["corruption_fraction"].append(0.2)
    iteration_results["feature"].append("star_rating")
    iteration_results["roc_auc"].append(scores["roc_auc"])
    iteration_results["f1"].append(scores["f1"])

    # TODO: Can we only corrupt all data once and then sample from it to further share work across iterations?

    if debug is True:
        print("____")
        print(f"Now testing corruption of {{corruption_fraction * 100}}% of feature {{corrupt_feature}}")
        print("Corruptions in Test")

    return pd.DataFrame(iteration_results)


def corrupt_data(data, corruption_fraction, corrupt_feature):
    # Data corruptions, later: test them one by one
    if corrupt_feature == "vine":
        data = CategoricalShift(column='vine', fraction=corruption_fraction).transform(data)
    elif corrupt_feature == "verified_purchase":
        data = CategoricalShift(column='verified_purchase', fraction=corruption_fraction).transform(data)
    elif corrupt_feature == "category_id":
        data['category_id'] = data['category_id'].astype(
            str)  # CategoricalShift implemented only for categorical variables
        data = CategoricalShift(column='category_id', fraction=corruption_fraction).transform(data)
        data['category_id'] = data['category_id'].astype(int)
    elif corrupt_feature == "star_rating":
        data = Scaling(column='star_rating', fraction=corruption_fraction).transform(data)
    elif corrupt_feature == "product_title":
        data = BrokenCharacters(column='product_title', fraction=corruption_fraction).transform(data)
    elif corrupt_feature == "review_headline":
        data = BrokenCharacters(column='review_headline', fraction=corruption_fraction).transform(data)
    elif corrupt_feature == "review_body":
        data = MissingValues(column='review_body', fraction=corruption_fraction).transform(data)
    else:
        assert False

    return data


def do_review_corruption_opt(debug, corruption_percentages, corrupt_features):
    # TODO: Make this general enough for the last 2 params to work? Prbbly not necessary
    iteration_results = execute_review_pipeline_opt(debug)
    return pd.DataFrame(iteration_results)


def measure_review_corruption_opt_exec_time(debug, corruption_percentages, corrupt_features, repeats=10):
    result = timeit.repeat(stmt=cleandoc(f"""
    if {debug} is True:
        print("No corruptions")
    execute_review_pipeline_opt(False, False, 0.0, 0, {debug})
    for corrupt_feature in {corrupt_features}:
        for corruption_fraction in {corruption_percentages}:
            if {debug} is True:
                print("____")
                print(f"Now testing corruption of {{corruption_fraction*100}}% of feature {{corrupt_feature}}")
                print("Corruptions in Test")
            execute_review_pipeline_opt(False, True, corruption_fraction, corrupt_feature, {debug})
            if {debug} is True:
                print("Corruptions in Train and test")
            execute_review_pipeline_opt(True, True, corruption_fraction, corrupt_feature, {debug})
    print("Done!")
    """),
                           setup=cleandoc(f"""
    from whatif.example_pipelines.helpful_reviews_data_corruptions_opt import execute_review_pipeline_opt
    """),
                           repeat=repeats, number=1)
    return pd.DataFrame({"runtimes": result})


do_review_corruption_opt(debug=True, corruption_percentages=[0.5, 0.9],
                           corrupt_features=["star_rating", "verified_purchase", "review_headline"])

# measure_review_corruption_opt_exec_time(debug=True, corruption_percentages=[0.5, 0.9],
#                                         corrupt_features=["star_rating", "verified_purchase", "review_headline"],
#                                         repeats=2)
