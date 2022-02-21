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
    train_labels = label_binarize(train_data['is_helpful'], classes=[True, False]).ravel()

    test_data['is_helpful'] = test_data['helpful_votes'] > 0
    test_labels = label_binarize(test_data['is_helpful'], classes=[True, False]).ravel()

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
    verified_purchase_train_featurized = verified_purchase_train_featurizer.fit_transform(
        verified_purchase_train).toarray()
    verified_purchase_test = test_data[['verified_purchase']]
    verified_purchase_test_featurized = verified_purchase_train_featurizer.transform(verified_purchase_test).toarray()

    category_id_train = train_data[['category_id']]
    category_id_featurizer = OneHotEncoder(handle_unknown='ignore')
    category_id_train_featurized = category_id_featurizer.fit_transform(category_id_train).toarray()
    category_id_test = test_data[['category_id']]
    category_id_test_featurized = category_id_featurizer.transform(category_id_test).toarray()

    review_headline_train = train_data[['review_headline']]
    title_and_review_text_train = train_data.product_title + ' ' + review_headline_train.review_headline + ' ' + \
                                  train_data.review_body
    title_and_review_text_train_featurizer = HashingVectorizer(ngram_range=(1, 3), n_features=100)
    title_and_review_text_train_featurized = title_and_review_text_train_featurizer \
        .fit_transform(title_and_review_text_train).toarray()
    review_headline_test = test_data[['review_headline']]
    title_and_review_text_test = test_data.product_title + ' ' + review_headline_test.review_headline + ' ' + \
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

    model_wo_corruptions = model_wo_corruptions.fit(train_wo_corruptions, train_labels)
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

    feature = "star_rating"
    star_rating_test_all_corrupt = star_rating_test.copy()
    scale_factor = numpy.random.choice([10, 100, 1000])
    star_rating_test_all_corrupt.loc[:, 'star_rating'] *= scale_factor
    test_star_rating_02c = corrupt_star_rating_test_with_corruption_fraction(category_id_test_featurized, 0.2, debug,
                                                                             feature,
                                                                             iteration_results, model_wo_corruptions,
                                                                             star_rating_test,
                                                                             star_rating_test_all_corrupt,
                                                                             star_rating_train_featurizer,
                                                                             test_labels,
                                                                             title_and_review_text_test_featurized,
                                                                             verified_purchase_test_featurized,
                                                                             vine_test_featurized)
    test_star_rating_05c = corrupt_star_rating_test_with_corruption_fraction(category_id_test_featurized, 0.5, debug,
                                                                             feature,
                                                                             iteration_results, model_wo_corruptions,
                                                                             star_rating_test,
                                                                             star_rating_test_all_corrupt,
                                                                             star_rating_train_featurizer,
                                                                             test_labels,
                                                                             title_and_review_text_test_featurized,
                                                                             verified_purchase_test_featurized,
                                                                             vine_test_featurized)
    test_star_rating_09c = corrupt_star_rating_test_with_corruption_fraction(category_id_test_featurized, 0.9, debug,
                                                                             feature,
                                                                             iteration_results, model_wo_corruptions,
                                                                             star_rating_test,
                                                                             star_rating_test_all_corrupt,
                                                                             star_rating_train_featurizer,
                                                                             test_labels,
                                                                             title_and_review_text_test_featurized,
                                                                             verified_purchase_test_featurized,
                                                                             vine_test_featurized)

    feature = "verified_purchase"
    verified_purchase_test_all_corrupt = verified_purchase_test.copy()
    histogram = verified_purchase_test_all_corrupt["verified_purchase"].value_counts()
    random_other_val = numpy.random.permutation(histogram.index)
    verified_purchase_test_all_corrupt.loc[:, "verified_purchase"] = verified_purchase_test_all_corrupt.loc[:,
                                                                     "verified_purchase"].replace(histogram.index,
                                                                                                  random_other_val)
    test_verified_purchase_02c = corrupt_verified_purchase_test_with_corruption_fraction(category_id_test_featurized,
                                                                                         0.2, debug, feature,
                                                                                         iteration_results,
                                                                                         model_wo_corruptions,
                                                                                         verified_purchase_test,
                                                                                         verified_purchase_test_all_corrupt,
                                                                                         verified_purchase_train_featurizer,
                                                                                         test_labels,
                                                                                         title_and_review_text_test_featurized,
                                                                                         star_rating_test_featurized,
                                                                                         vine_test_featurized)
    test_verified_purchase_05c = corrupt_verified_purchase_test_with_corruption_fraction(category_id_test_featurized,
                                                                                         0.5, debug, feature,
                                                                                         iteration_results,
                                                                                         model_wo_corruptions,
                                                                                         verified_purchase_test,
                                                                                         verified_purchase_test_all_corrupt,
                                                                                         verified_purchase_train_featurizer,
                                                                                         test_labels,
                                                                                         title_and_review_text_test_featurized,
                                                                                         star_rating_test_featurized,
                                                                                         vine_test_featurized)
    test_verified_purchase_09c = corrupt_verified_purchase_test_with_corruption_fraction(category_id_test_featurized,
                                                                                         0.9, debug, feature,
                                                                                         iteration_results,
                                                                                         model_wo_corruptions,
                                                                                         verified_purchase_test,
                                                                                         verified_purchase_test_all_corrupt,
                                                                                         verified_purchase_train_featurizer,
                                                                                         test_labels,
                                                                                         title_and_review_text_test_featurized,
                                                                                         star_rating_test_featurized,
                                                                                         vine_test_featurized)

    feature = "review_headline"
    review_headline_test_all_corrupt = review_headline_test.copy()
    replacements = {
        'a': 'á',
        'A': 'Á',
        'e': 'é',
        'E': 'É',
        'o': 'ớ',
        'O': 'Ớ',
        'u': 'ú',
        'U': 'Ú'
    }
    for index, row in review_headline_test_all_corrupt.iterrows():
        column_value = row["review_headline"]
        for character, replacement in replacements.items():
            column_value = str(column_value).replace(character, replacement)
        review_headline_test_all_corrupt.at[index, "review_headline"] = column_value
    # dont forget additional column creation
    test_review_headline_02c = corrupt_review_headline_test_with_corruption_fraction(category_id_test_featurized, 0.2,
                                                                                     debug,
                                                                                     feature,
                                                                                     iteration_results,
                                                                                     model_wo_corruptions,
                                                                                     verified_purchase_test_featurized,
                                                                                     review_headline_test,
                                                                                     review_headline_test_all_corrupt,
                                                                                     title_and_review_text_train_featurizer,
                                                                                     test_labels,
                                                                                     star_rating_test_featurized,
                                                                                     vine_test_featurized,
                                                                                     test_data.product_title,
                                                                                     test_data.review_body)
    test_review_headline_05c = corrupt_review_headline_test_with_corruption_fraction(category_id_test_featurized, 0.5,
                                                                                     debug,
                                                                                     feature,
                                                                                     iteration_results,
                                                                                     model_wo_corruptions,
                                                                                     verified_purchase_test_featurized,
                                                                                     review_headline_test,
                                                                                     review_headline_test_all_corrupt,
                                                                                     title_and_review_text_train_featurizer,
                                                                                     test_labels,
                                                                                     star_rating_test_featurized,
                                                                                     vine_test_featurized,
                                                                                     test_data.product_title,
                                                                                     test_data.review_body)
    test_review_headline_09c = corrupt_review_headline_test_with_corruption_fraction(category_id_test_featurized, 0.9,
                                                                                     debug,
                                                                                     feature,
                                                                                     iteration_results,
                                                                                     model_wo_corruptions,
                                                                                     verified_purchase_test_featurized,
                                                                                     review_headline_test,
                                                                                     review_headline_test_all_corrupt,
                                                                                     title_and_review_text_train_featurizer,
                                                                                     test_labels,
                                                                                     star_rating_test_featurized,
                                                                                     vine_test_featurized,
                                                                                     test_data.product_title,
                                                                                     test_data.review_body)

    feature = "star_rating"
    star_rating_train_all_corrupt = star_rating_train.copy()
    scale_factor = numpy.random.choice([10, 100, 1000])
    star_rating_train_all_corrupt.loc[:, 'star_rating'] *= scale_factor

    corrupt_star_rating_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                    0.2, debug, feature, iteration_results,
                                                    star_rating_train, star_rating_train_all_corrupt, test_labels,
                                                    test_star_rating_02c, title_and_review_text_test_featurized,
                                                    title_and_review_text_train_featurized, train_labels,
                                                    verified_purchase_test_featurized,
                                                    verified_purchase_train_featurized, vine_test_featurized,
                                                    vine_train_featurized)
    corrupt_star_rating_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                    0.5, debug, feature, iteration_results,
                                                    star_rating_train, star_rating_train_all_corrupt, test_labels,
                                                    test_star_rating_05c, title_and_review_text_test_featurized,
                                                    title_and_review_text_train_featurized, train_labels,
                                                    verified_purchase_test_featurized,
                                                    verified_purchase_train_featurized, vine_test_featurized,
                                                    vine_train_featurized)
    corrupt_star_rating_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                    0.9, debug, feature, iteration_results,
                                                    star_rating_train, star_rating_train_all_corrupt, test_labels,
                                                    test_star_rating_09c, title_and_review_text_test_featurized,
                                                    title_and_review_text_train_featurized, train_labels,
                                                    verified_purchase_test_featurized,
                                                    verified_purchase_train_featurized, vine_test_featurized,
                                                    vine_train_featurized)

    feature = "verified_purchase"
    verified_purchase_train_all_corrupt = verified_purchase_train.copy()
    histogram = verified_purchase_train_all_corrupt["verified_purchase"].value_counts()
    random_other_val = numpy.random.permutation(histogram.index)
    verified_purchase_train_all_corrupt.loc[:, "verified_purchase"] = verified_purchase_train_all_corrupt.loc[:,
                                                                      "verified_purchase"].replace(histogram.index,
                                                                                                   random_other_val)
    corrupt_verified_purchase_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                          0.2, debug, feature, iteration_results,
                                                          star_rating_test_featurized, star_rating_train_featurized,
                                                          test_labels, test_verified_purchase_02c,
                                                          title_and_review_text_test_featurized,
                                                          title_and_review_text_train_featurized, train_labels,
                                                          verified_purchase_train, verified_purchase_train_all_corrupt,
                                                          vine_test_featurized, vine_train_featurized)
    corrupt_verified_purchase_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                          0.5, debug, feature, iteration_results,
                                                          star_rating_test_featurized, star_rating_train_featurized,
                                                          test_labels, test_verified_purchase_05c,
                                                          title_and_review_text_test_featurized,
                                                          title_and_review_text_train_featurized, train_labels,
                                                          verified_purchase_train, verified_purchase_train_all_corrupt,
                                                          vine_test_featurized, vine_train_featurized)
    corrupt_verified_purchase_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                          0.9, debug, feature, iteration_results,
                                                          star_rating_test_featurized, star_rating_train_featurized,
                                                          test_labels, test_verified_purchase_09c,
                                                          title_and_review_text_test_featurized,
                                                          title_and_review_text_train_featurized, train_labels,
                                                          verified_purchase_train, verified_purchase_train_all_corrupt,
                                                          vine_test_featurized, vine_train_featurized)

    feature = "review_headline"
    review_headline_train_all_corrupt = review_headline_train.copy()
    for index, row in review_headline_train_all_corrupt.iterrows():
        column_value = row["review_headline"]
        for character, replacement in replacements.items():
            column_value = str(column_value).replace(character, replacement)
        review_headline_train_all_corrupt.at[index, "review_headline"] = column_value

    corrupt_review_headline_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                        0.2, debug, feature, iteration_results,
                                                        train_data.product_title, train_data.review_body, review_headline_train,
                                                        review_headline_train_all_corrupt, star_rating_test_featurized,
                                                        star_rating_train_featurized, test_labels,
                                                        test_review_headline_02c, train_labels,
                                                        verified_purchase_test_featurized,
                                                        verified_purchase_train_featurized, vine_test_featurized,
                                                        vine_train_featurized)
    corrupt_review_headline_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                        0.5, debug, feature, iteration_results,
                                                        train_data.product_title, train_data.review_body,
                                                        review_headline_train,
                                                        review_headline_train_all_corrupt, star_rating_test_featurized,
                                                        star_rating_train_featurized, test_labels,
                                                        test_review_headline_05c, train_labels,
                                                        verified_purchase_test_featurized,
                                                        verified_purchase_train_featurized, vine_test_featurized,
                                                        vine_train_featurized)
    corrupt_review_headline_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                        0.9, debug, feature, iteration_results,
                                                        train_data.product_title, train_data.review_body,
                                                        review_headline_train,
                                                        review_headline_train_all_corrupt, star_rating_test_featurized,
                                                        star_rating_train_featurized, test_labels,
                                                        test_review_headline_09c, train_labels,
                                                        verified_purchase_test_featurized,
                                                        verified_purchase_train_featurized, vine_test_featurized,
                                                        vine_train_featurized)

    return pd.DataFrame(iteration_results)


def corrupt_review_headline_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                        corruption_fraction, debug, feature, iteration_results,
                                                        product_title_train, review_body_train, review_headline_train,
                                                        review_headline_train_all_corrupt, star_rating_test_featurized,
                                                        star_rating_train_featurized, test_labels,
                                                        test_review_headline_02c, train_labels,
                                                        verified_purchase_test_featurized,
                                                        verified_purchase_train_featurized, vine_test_featurized,
                                                        vine_train_featurized):
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Train and Test")
    review_headline_train_w_corrupt_fraction = review_headline_train.copy()
    indexes_to_corrupt = numpy.random.permutation(review_headline_train.index)[
                         :int(len(review_headline_train) * corruption_fraction)]
    review_headline_train_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = review_headline_train_all_corrupt.loc[
        indexes_to_corrupt, feature]
    title_and_review_text_train_w_corrupt_fraction = product_title_train + ' ' + review_headline_train_w_corrupt_fraction.review_headline + ' ' + \
                                                     review_body_train
    title_and_review_text_train_featurizer_02c = HashingVectorizer(ngram_range=(1, 3), n_features=100)
    title_and_review_text_train_featurized_w_corrupt_fraction = title_and_review_text_train_featurizer_02c \
        .fit_transform(title_and_review_text_train_w_corrupt_fraction).toarray()
    train_review_headline_w_corrupt_fraction = numpy.hstack([star_rating_train_featurized, vine_train_featurized,
                                                             verified_purchase_train_featurized,
                                                             category_id_train_featurized,
                                                             title_and_review_text_train_featurized_w_corrupt_fraction])
    model_review_headline_c02 = SGDClassifier(loss='log', penalty='l1', max_iter=1000, class_weight="balanced")
    model_review_headline_c02 = model_review_headline_c02.fit(train_review_headline_w_corrupt_fraction,
                                                              train_labels)
    title_and_review_text_test_featurized_w_corrupt_fraction = title_and_review_text_train_featurizer_02c.transform(
        test_review_headline_02c).toarray()
    test_predict_review_headline_corrupt_fraction = numpy.hstack(
        [star_rating_test_featurized, vine_test_featurized,
         verified_purchase_test_featurized,
         category_id_test_featurized, title_and_review_text_test_featurized_w_corrupt_fraction])
    # This should fail
    # numpy.testing.assert_allclose(test_wo_corruptions, test_verified_purchase_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict_review_headline_corrupt_fraction = model_review_headline_c02.predict(
        test_predict_review_headline_corrupt_fraction)
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict_review_headline_corrupt_fraction, test_labels)
    scores['f1'] = f1_score(test_predict_review_headline_corrupt_fraction, test_labels, average='macro')
    print_if_debug_and_append_iteration_results(debug, iteration_results, scores, True, True,
                                                corruption_fraction, feature)


def corrupt_verified_purchase_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                          corruption_fraction, debug, feature, iteration_results,
                                                          star_rating_test_featurized, star_rating_train_featurized,
                                                          test_labels, test_verified_purchase_02c,
                                                          title_and_review_text_test_featurized,
                                                          title_and_review_text_train_featurized, train_labels,
                                                          verified_purchase_train, verified_purchase_train_all_corrupt,
                                                          vine_test_featurized, vine_train_featurized):
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Train and Test")
    verified_purchase_train_w_corrupt_fraction = verified_purchase_train.copy()
    indexes_to_corrupt = numpy.random.permutation(verified_purchase_train.index)[
                         :int(len(verified_purchase_train) * corruption_fraction)]
    verified_purchase_train_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = \
    verified_purchase_train_all_corrupt.loc[
        indexes_to_corrupt, feature]
    verified_purchase_train_featurizer_c02 = OneHotEncoder(handle_unknown='ignore')
    verified_purchase_train_featurized_w_corrupt_fraction = verified_purchase_train_featurizer_c02 \
        .fit_transform(verified_purchase_train_w_corrupt_fraction).toarray()
    train_verified_purchase_w_corrupt_fraction = numpy.hstack([star_rating_train_featurized, vine_train_featurized,
                                                               verified_purchase_train_featurized_w_corrupt_fraction,
                                                               category_id_train_featurized,
                                                               title_and_review_text_train_featurized])
    model_verified_purchase_c02 = SGDClassifier(loss='log', penalty='l1', max_iter=1000, class_weight="balanced")
    model_verified_purchase_c02 = model_verified_purchase_c02.fit(train_verified_purchase_w_corrupt_fraction,
                                                                  train_labels)
    verified_purchase_test_featurized_w_corrupt_fraction = verified_purchase_train_featurizer_c02.transform(
        test_verified_purchase_02c).toarray()
    test_predict_verified_purchase_corrupt_fraction = numpy.hstack(
        [star_rating_test_featurized, vine_test_featurized,
         verified_purchase_test_featurized_w_corrupt_fraction,
         category_id_test_featurized, title_and_review_text_test_featurized])
    # This should fail
    # numpy.testing.assert_allclose(test_wo_corruptions, test_verified_purchase_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict_verified_purchase_corrupt_fraction = model_verified_purchase_c02.predict(
        test_predict_verified_purchase_corrupt_fraction)
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict_verified_purchase_corrupt_fraction, test_labels)
    scores['f1'] = f1_score(test_predict_verified_purchase_corrupt_fraction, test_labels, average='macro')
    print_if_debug_and_append_iteration_results(debug, iteration_results, scores, True, True,
                                                corruption_fraction, feature)


def corrupt_star_rating_train_w_corruption_fraction(category_id_test_featurized, category_id_train_featurized,
                                                    corruption_fraction, debug, feature, iteration_results,
                                                    star_rating_train, star_rating_train_all_corrupt, test_labels,
                                                    test_star_rating_02c, title_and_review_text_test_featurized,
                                                    title_and_review_text_train_featurized, train_labels,
                                                    verified_purchase_test_featurized,
                                                    verified_purchase_train_featurized, vine_test_featurized,
                                                    vine_train_featurized):
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Train and Test")
    star_rating_train_w_corrupt_fraction = star_rating_train.copy()
    indexes_to_corrupt = numpy.random.permutation(star_rating_train.index)[
                         :int(len(star_rating_train) * corruption_fraction)]
    star_rating_train_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = star_rating_train_all_corrupt.loc[
        indexes_to_corrupt, feature]
    star_rating_train_featurizer_c02 = StandardScaler()
    star_rating_train_featurized_w_corrupt_fraction = star_rating_train_featurizer_c02.fit_transform(
        star_rating_train_w_corrupt_fraction)
    train_star_rating_w_corrupt_fraction = numpy.hstack(
        [star_rating_train_featurized_w_corrupt_fraction, vine_train_featurized,
         verified_purchase_train_featurized,
         category_id_train_featurized, title_and_review_text_train_featurized])
    # This should fail
    # numpy.testing.assert_allclose(test_wo_corruptions, test_star_rating_w_corrupt_fraction, rtol=1e-5, atol=0)
    model_star_rating_c02 = SGDClassifier(loss='log', penalty='l1', max_iter=1000, class_weight="balanced")
    model_star_rating_c02 = model_star_rating_c02.fit(train_star_rating_w_corrupt_fraction, train_labels)
    star_rating_test_featurized_w_corrupt_fraction = star_rating_train_featurizer_c02.transform(
        test_star_rating_02c)
    test_star_rating_w_corrupt_fraction = numpy.hstack(
        [star_rating_test_featurized_w_corrupt_fraction, vine_test_featurized,
         verified_purchase_test_featurized,
         category_id_test_featurized, title_and_review_text_test_featurized])
    test_predict_star_rating_corrupt_fraction = model_star_rating_c02.predict(
        test_star_rating_w_corrupt_fraction)
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict_star_rating_corrupt_fraction, test_labels)
    scores['f1'] = f1_score(test_predict_star_rating_corrupt_fraction, test_labels, average='macro')
    print_if_debug_and_append_iteration_results(debug, iteration_results, scores, True, True,
                                                corruption_fraction, feature)


def corrupt_star_rating_test_with_corruption_fraction(category_id_test_featurized, corruption_fraction, debug, feature,
                                                      iteration_results, model_wo_corruptions, star_rating_test,
                                                      star_rating_test_all_corrupt, star_rating_train_featurizer,
                                                      test_labels, title_and_review_text_test_featurized,
                                                      verified_purchase_test_featurized, vine_test_featurized):
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Test")
    star_rating_test_w_corrupt_fraction = star_rating_test.copy()
    indexes_to_corrupt = numpy.random.permutation(star_rating_test.index)[
                         :int(len(star_rating_test) * corruption_fraction)]
    star_rating_test_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = star_rating_test_all_corrupt.loc[
        indexes_to_corrupt, feature]
    star_rating_test_featurized_w_corrupt_fraction = star_rating_train_featurizer.transform(
        star_rating_test_w_corrupt_fraction)
    test_star_rating_w_corrupt_fraction = numpy.hstack(
        [star_rating_test_featurized_w_corrupt_fraction, vine_test_featurized,
         verified_purchase_test_featurized,
         category_id_test_featurized, title_and_review_text_test_featurized])
    # This should fail
    # numpy.testing.assert_allclose(test_wo_corruptions, test_star_rating_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict_star_rating_corrupt_fraction = model_wo_corruptions.predict(test_star_rating_w_corrupt_fraction)
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict_star_rating_corrupt_fraction, test_labels)
    scores['f1'] = f1_score(test_predict_star_rating_corrupt_fraction, test_labels, average='macro')
    print_if_debug_and_append_iteration_results(debug, iteration_results, scores, True, False,
                                                corruption_fraction, feature)

    return star_rating_test_w_corrupt_fraction


def corrupt_verified_purchase_test_with_corruption_fraction(category_id_test_featurized, corruption_fraction, debug,
                                                            feature,
                                                            iteration_results, model_wo_corruptions,
                                                            verified_purchase_test,
                                                            verified_purchase_test_all_corrupt,
                                                            verified_purchase_train_featurizer,
                                                            test_labels, title_and_review_text_test_featurized,
                                                            star_rating_test_featurized, vine_test_featurized):
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Test")
    verified_purchase_test_w_corrupt_fraction = verified_purchase_test.copy()
    indexes_to_corrupt = numpy.random.permutation(verified_purchase_test.index)[
                         :int(len(verified_purchase_test) * corruption_fraction)]
    verified_purchase_test_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = verified_purchase_test_all_corrupt.loc[
        indexes_to_corrupt, feature]
    verified_purchase_test_featurized_w_corrupt_fraction = verified_purchase_train_featurizer \
        .transform(verified_purchase_test_w_corrupt_fraction).toarray()
    test_verified_purchase_w_corrupt_fraction = numpy.hstack([star_rating_test_featurized, vine_test_featurized,
                                                              verified_purchase_test_featurized_w_corrupt_fraction,
                                                              category_id_test_featurized,
                                                              title_and_review_text_test_featurized])
    # This should fail
    # numpy.testing.assert_allclose(test_wo_corruptions, test_verified_purchase_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict_verified_purchase_corrupt_fraction = model_wo_corruptions.predict(
        test_verified_purchase_w_corrupt_fraction)
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict_verified_purchase_corrupt_fraction, test_labels)
    scores['f1'] = f1_score(test_predict_verified_purchase_corrupt_fraction, test_labels, average='macro')
    print_if_debug_and_append_iteration_results(debug, iteration_results, scores, True, False,
                                                corruption_fraction, feature)

    return verified_purchase_test_w_corrupt_fraction


def corrupt_review_headline_test_with_corruption_fraction(category_id_test_featurized, corruption_fraction, debug,
                                                          feature,
                                                          iteration_results, model_wo_corruptions,
                                                          verified_purchase_test_featurized,
                                                          review_headline_test, review_headline_test_all_corrupt,
                                                          title_and_review_text_train_featurizer,
                                                          test_labels, star_rating_test_featurized,
                                                          vine_test_featurized,
                                                          product_title_test, review_body_test):
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Test")
    review_headline_test_w_corrupt_fraction = review_headline_test.copy()
    indexes_to_corrupt = numpy.random.permutation(review_headline_test.index)[
                         :int(len(review_headline_test) * corruption_fraction)]
    review_headline_test_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = review_headline_test_all_corrupt.loc[
        indexes_to_corrupt, feature]
    title_and_review_text_test_w_corrupt_fraction = product_title_test + ' ' + review_headline_test_w_corrupt_fraction.review_headline + ' ' + \
                                                    review_body_test

    title_and_review_text_test_featurized_w_corrupt_fraction = title_and_review_text_train_featurizer \
        .transform(title_and_review_text_test_w_corrupt_fraction).toarray()
    test_verified_purchase_w_corrupt_fraction = numpy.hstack([star_rating_test_featurized, vine_test_featurized,
                                                              verified_purchase_test_featurized,
                                                              category_id_test_featurized,
                                                              title_and_review_text_test_featurized_w_corrupt_fraction])
    # This should fail
    # numpy.testing.assert_allclose(test_wo_corruptions, test_verified_purchase_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict_verified_purchase_corrupt_fraction = model_wo_corruptions.predict(
        test_verified_purchase_w_corrupt_fraction)
    scores = {}
    scores['roc_auc'] = roc_auc_score(test_predict_verified_purchase_corrupt_fraction, test_labels)
    scores['f1'] = f1_score(test_predict_verified_purchase_corrupt_fraction, test_labels, average='macro')
    print_if_debug_and_append_iteration_results(debug, iteration_results, scores, True, False,
                                                corruption_fraction, feature)

    return title_and_review_text_test_w_corrupt_fraction


def print_if_debug_and_append_iteration_results(debug, iteration_results, scores, test_corruption,
                                                train_corruption, corruption_fraction, feature):
    if debug is True:
        print(f'AUC Score on the test set: {scores["roc_auc"]}')
    if debug is True:
        print(f'F1 Score on the test set: {scores["f1"]}')
    iteration_results["test_corruption"].append(test_corruption)
    iteration_results["train_corruption"].append(train_corruption)
    iteration_results["corruption_fraction"].append(corruption_fraction)
    iteration_results["feature"].append(feature)
    iteration_results["roc_auc"].append(scores["roc_auc"])
    iteration_results["f1"].append(scores["f1"])


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


def do_review_corruption_opt(debug):
    # TODO: Make this general enough for the last 2 params to work? Prbbly not necessary
    iteration_results = execute_review_pipeline_opt(debug)
    return pd.DataFrame(iteration_results)


def measure_review_corruption_opt_exec_time(debug, repeats=10):
    result = timeit.repeat(stmt=cleandoc(f"""
    execute_review_pipeline_opt({debug})
    print("Done!")
    """),
                           setup=cleandoc(f"""
    from whatif.example_pipelines.helpful_reviews_data_corruptions_opt import execute_review_pipeline_opt
    """),
                           repeat=repeats, number=1)
    return pd.DataFrame({"runtimes": result})


# do_review_corruption_opt(debug=True)

# measure_review_corruption_opt_exec_time(debug=True, repeats=2)
