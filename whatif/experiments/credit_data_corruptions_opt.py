import timeit
from inspect import cleandoc

import numpy
import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, label_binarize

from whatif.experiments.credit_data_corruptions_naive import compute_fairness_metric
from whatif.utils.utils import get_project_root


def execute_credit_pipeline_opt(debug):
    iteration_results = {
        "test_corruption": [],
        "train_corruption": [],
        "corruption_fraction": [],
        "feature": [],
        "accuracy": [],
        "non_protected_fnr": [],
        "protected_fnr": []
    }

    def load_train_and_test_data(adult_train_location, adult_test_location, excluded_employment_types):

        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'native-country', 'income-per-year']

        adult_train = pd.read_csv(adult_train_location, names=columns, sep=', ', engine='python', na_values="?")
        adult_test = pd.read_csv(adult_test_location, names=columns, sep=', ', engine='python', na_values="?",
                                 skiprows=1)

        for employment_type in excluded_employment_types:
            adult_train = adult_train[adult_train['workclass'] != employment_type]
            adult_test = adult_test[adult_test['workclass'] != employment_type]

        return adult_train, adult_test

    def extract_labels(adult_train, adult_test):
        adult_train_labels = label_binarize(adult_train['income-per-year'], classes=['<=50K', '>50K']).ravel()
        # The test data has a dot in the class names for some reason...
        adult_test_labels = label_binarize(adult_test['income-per-year'], classes=['<=50K.', '>50K.'])

        return adult_train_labels, adult_test_labels

    def safe_log(x):
        return np.log(x, out=np.zeros_like(x), where=(x != 0))

    train_location = f'{str(get_project_root())}/whatif/experiments/datasets/income/adult.data'
    test_location = f'{str(get_project_root())}/whatif/experiments/datasets/income/adult.test'

    government_employed = ['Federal-gov', 'State-gov']

    train, test = load_train_and_test_data(train_location, test_location, excluded_employment_types=government_employed)

    train_labels, test_labels = extract_labels(train, test)

    def get_cat_featurizer():
        return Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

    def get_num_featurizer():
        return Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('log_transform', FunctionTransformer(lambda x: safe_log(x))),
            ('scale', StandardScaler())
        ])

    workclass_featurizer = get_cat_featurizer()
    education_featurizer = get_cat_featurizer()
    occupation_featurizer = get_cat_featurizer()
    age_featurizer = get_num_featurizer()
    capital_gain_featurizer = get_num_featurizer()
    capital_loss_featurizer = get_num_featurizer()
    hours_per_week_featurizer = get_num_featurizer()

    workclass_train = workclass_featurizer.fit_transform(train[['workclass']])
    education_train = education_featurizer.fit_transform(train[['education']])
    occupation_train = occupation_featurizer.fit_transform(train[['occupation']])
    age_train = age_featurizer.fit_transform(train[['age']])
    capital_gain_train = capital_gain_featurizer.fit_transform(train[['capital-gain']])
    capital_loss_train = capital_loss_featurizer.fit_transform(train[['capital-loss']])
    hours_per_week_train = hours_per_week_featurizer.fit_transform(train[['hours-per-week']])

    featurized_train = numpy.hstack([workclass_train, education_train, occupation_train, age_train,
                                     capital_gain_train, capital_loss_train, hours_per_week_train])

    model_wo_corruption = SGDClassifier(loss='log')

    model_wo_corruption = model_wo_corruption.fit(featurized_train, train_labels)

    workclass_test = workclass_featurizer.transform(test[['workclass']])
    education_test = education_featurizer.transform(test[['education']])
    occupation_test = occupation_featurizer.transform(test[['occupation']])
    age_test = age_featurizer.transform(test[['age']])
    capital_gain_test = capital_gain_featurizer.transform(test[['capital-gain']])
    capital_loss_test = capital_loss_featurizer.transform(test[['capital-loss']])
    hours_per_week_test = hours_per_week_featurizer.transform(test[['hours-per-week']])

    featurized_test = numpy.hstack([workclass_test, education_test, occupation_test, age_test,
                                    capital_gain_test, capital_loss_test, hours_per_week_test])
    test_predict = model_wo_corruption.predict(featurized_test)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict)
    if debug is True:
        print(f'Accuracy on the test set: {scores["accuracy"]}')
        print(f'non_protected_fnr on the test set: {scores["non_protected_fnr"]}')
        print(f'protected_fnr on the test set: {scores["protected_fnr"]}')

    iteration_results["test_corruption"].append(False)
    iteration_results["train_corruption"].append(False)
    iteration_results["corruption_fraction"].append(None)
    iteration_results["feature"].append(None)
    iteration_results["accuracy"].append(scores["accuracy"])
    iteration_results["non_protected_fnr"].append(scores["non_protected_fnr"])
    iteration_results["protected_fnr"].append(scores["protected_fnr"])

    feature = "age"
    age_test_all_corrupt = test[['age']].copy()
    scale_factor = numpy.random.choice([10, 100, 1000])
    age_test_all_corrupt.loc[:, 'age'] *= scale_factor
    age_test_c02 = corrupt_age_test_set_only(age_featurizer, age_test_all_corrupt, capital_gain_test, capital_loss_test,
                                             0.2, debug, education_test, feature, featurized_test, hours_per_week_test,
                                             iteration_results, model_wo_corruption, occupation_test, test, test_labels,
                                             workclass_test)
    age_test_c05 = corrupt_age_test_set_only(age_featurizer, age_test_all_corrupt, capital_gain_test, capital_loss_test,
                                             0.5, debug, education_test, feature, featurized_test, hours_per_week_test,
                                             iteration_results, model_wo_corruption, occupation_test, test, test_labels,
                                             workclass_test)
    age_test_c09 = corrupt_age_test_set_only(age_featurizer, age_test_all_corrupt, capital_gain_test, capital_loss_test,
                                             0.9, debug, education_test, feature, featurized_test, hours_per_week_test,
                                             iteration_results, model_wo_corruption, occupation_test, test, test_labels,
                                             workclass_test)

    feature = "workclass"
    workclass_test_all_corrupt = test[['workclass']].copy()
    histogram = workclass_test_all_corrupt["workclass"].value_counts()
    random_other_val = numpy.random.permutation(histogram.index)
    workclass_test_all_corrupt.loc[:, "workclass"] = workclass_test_all_corrupt.loc[:, "workclass"] \
        .replace(histogram.index, random_other_val)
    workclass_test_c02 = corrupt_workclass_test_set_only(workclass_featurizer, workclass_test_all_corrupt,
                                                         capital_gain_test, capital_loss_test,
                                                         0.2, debug, education_test, feature, featurized_test,
                                                         hours_per_week_test,
                                                         iteration_results, model_wo_corruption, occupation_test, test,
                                                         test_labels,
                                                         age_test)
    workclass_test_c05 = corrupt_workclass_test_set_only(workclass_featurizer, workclass_test_all_corrupt,
                                                         capital_gain_test, capital_loss_test,
                                                         0.5, debug, education_test, feature, featurized_test,
                                                         hours_per_week_test,
                                                         iteration_results, model_wo_corruption, occupation_test, test,
                                                         test_labels,
                                                         age_test)
    workclass_test_c09 = corrupt_workclass_test_set_only(workclass_featurizer, workclass_test_all_corrupt,
                                                         capital_gain_test, capital_loss_test,
                                                         0.9, debug, education_test, feature, featurized_test,
                                                         hours_per_week_test,
                                                         iteration_results, model_wo_corruption, occupation_test, test,
                                                         test_labels,
                                                         age_test)

    feature = "education"
    education_test_c02 = corrupt_education_test_set_only(education_featurizer, None,
                                                         capital_gain_test, capital_loss_test,
                                                         0.2, debug, workclass_test, feature, featurized_test,
                                                         hours_per_week_test,
                                                         iteration_results, model_wo_corruption, occupation_test, test,
                                                         test_labels,
                                                         age_test)
    education_test_c05 = corrupt_education_test_set_only(education_featurizer, None,
                                                         capital_gain_test, capital_loss_test,
                                                         0.5, debug, workclass_test, feature, featurized_test,
                                                         hours_per_week_test,
                                                         iteration_results, model_wo_corruption, occupation_test, test,
                                                         test_labels,
                                                         age_test)
    education_test_c09 = corrupt_education_test_set_only(education_featurizer, None,
                                                         capital_gain_test, capital_loss_test,
                                                         0.9, debug, workclass_test, feature, featurized_test,
                                                         hours_per_week_test,
                                                         iteration_results, model_wo_corruption, occupation_test, test,
                                                         test_labels,
                                                         age_test)
    feature = "age"
    age_train_all_corrupt = train[['age']].copy()
    scale_factor = numpy.random.choice([10, 100, 1000])
    age_train_all_corrupt.loc[:, 'age'] *= scale_factor

    corrupt_age_train_and_test(age_test_c02, capital_gain_test, capital_gain_train, capital_loss_test,
                               capital_loss_train,  0.2, debug, education_test, education_train, feature,
                               featurized_train, get_num_featurizer, hours_per_week_test, hours_per_week_train,
                               iteration_results, occupation_test, occupation_train, test, test_labels, train,
                               train_labels, workclass_test, workclass_train, age_train_all_corrupt)
    corrupt_age_train_and_test(age_test_c05, capital_gain_test, capital_gain_train, capital_loss_test,
                               capital_loss_train, 0.5, debug, education_test, education_train, feature,
                               featurized_train, get_num_featurizer, hours_per_week_test, hours_per_week_train,
                               iteration_results, occupation_test, occupation_train, test, test_labels, train,
                               train_labels, workclass_test, workclass_train, age_train_all_corrupt)
    corrupt_age_train_and_test(age_test_c09, capital_gain_test, capital_gain_train, capital_loss_test,
                               capital_loss_train, 0.9, debug, education_test, education_train, feature,
                               featurized_train, get_num_featurizer, hours_per_week_test, hours_per_week_train,
                               iteration_results, occupation_test, occupation_train, test, test_labels, train,
                               train_labels, workclass_test, workclass_train, age_train_all_corrupt)

    feature = "workclass"
    workclass_train_all_corrupt = train[['workclass']].copy()
    histogram = workclass_train_all_corrupt["workclass"].value_counts()
    random_other_val = numpy.random.permutation(histogram.index)
    workclass_train_all_corrupt.loc[:, "workclass"] = workclass_train_all_corrupt.loc[:, "workclass"] \
        .replace(histogram.index, random_other_val)

    corrupt_workclass_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, 0.2, debug, education_test, education_train,
                                     feature, get_cat_featurizer, hours_per_week_test, hours_per_week_train,
                                     iteration_results, occupation_test, occupation_train, test, test_labels, train,
                                     train_labels, workclass_test_c02, workclass_train_all_corrupt)
    corrupt_workclass_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, 0.5, debug, education_test, education_train,
                                     feature, get_cat_featurizer, hours_per_week_test, hours_per_week_train,
                                     iteration_results, occupation_test, occupation_train, test, test_labels, train,
                                     train_labels, workclass_test_c05, workclass_train_all_corrupt)
    corrupt_workclass_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, 0.9, debug, education_test, education_train,
                                     feature, get_cat_featurizer, hours_per_week_test, hours_per_week_train,
                                     iteration_results, occupation_test, occupation_train, test, test_labels, train,
                                     train_labels, workclass_test_c09, workclass_train_all_corrupt)

    feature = "education"
    corrupt_education_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, 0.2, debug, education_test_c02, feature,
                                     get_cat_featurizer, hours_per_week_test, hours_per_week_train, iteration_results,
                                     occupation_test, occupation_train, test, test_labels, train, train_labels,
                                     workclass_test, workclass_train)
    corrupt_education_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, 0.5, debug, education_test_c05, feature,
                                     get_cat_featurizer, hours_per_week_test, hours_per_week_train, iteration_results,
                                     occupation_test, occupation_train, test, test_labels, train, train_labels,
                                     workclass_test, workclass_train)
    corrupt_education_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, 0.9, debug, education_test_c09, feature,
                                     get_cat_featurizer, hours_per_week_test, hours_per_week_train, iteration_results,
                                     occupation_test, occupation_train, test, test_labels, train, train_labels,
                                     workclass_test, workclass_train)

    return iteration_results


def corrupt_education_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, corruption_fraction, debug, education_test_c02, feature,
                                     get_cat_featurizer, hours_per_week_test, hours_per_week_train, iteration_results,
                                     occupation_test, occupation_train, test, test_labels, train, train_labels,
                                     workclass_test, workclass_train):
    education_train_unfeaturized = train[[feature]]
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Train and Test")
    education_train_w_corrupt_fraction = education_train_unfeaturized.copy()
    indexes_to_corrupt = numpy.random.permutation(education_train_unfeaturized.index)[
                         :int(len(education_train_unfeaturized) * corruption_fraction)]
    education_train_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = numpy.nan
    education_c02_featurizer = get_cat_featurizer()
    education_train_featurized_w_corrupt_fraction = education_c02_featurizer.fit_transform(
        education_train_w_corrupt_fraction)
    train_education_w_corrupt_fraction = numpy.hstack([workclass_train, education_train_featurized_w_corrupt_fraction,
                                                       occupation_train, age_train,
                                                       capital_gain_train, capital_loss_train, hours_per_week_train])
    # numpy.testing.assert_allclose(featurized_train, train_education_w_corrupt_fraction, rtol=1e-5, atol=0)
    model_education_c02 = SGDClassifier(loss='log')
    model_education_c02.fit(train_education_w_corrupt_fraction, train_labels)
    featurized_education_c02_test = education_c02_featurizer.transform(education_test_c02)
    featurized_test_w_education_c02 = numpy.hstack(
        [workclass_test, featurized_education_c02_test, occupation_test, age_test,
         capital_gain_test, capital_loss_test, hours_per_week_test])
    test_predict_w_workclass_c02 = model_education_c02.predict(featurized_test_w_education_c02)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict_w_workclass_c02)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict_w_workclass_c02)
    print_if_debug_and_store_iteration_results(True, True, corruption_fraction, debug, feature, iteration_results,
                                               scores)


def corrupt_workclass_train_and_test(age_test, age_train, capital_gain_test, capital_gain_train, capital_loss_test,
                                     capital_loss_train, corruption_fraction, debug, education_test, education_train,
                                     feature, get_cat_featurizer, hours_per_week_test, hours_per_week_train,
                                     iteration_results, occupation_test, occupation_train, test, test_labels, train,
                                     train_labels, workclass_test_c02, workclass_train_all_corrupt):
    workclass_train_unfeaturized = train[[feature]]
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Train and Test")
    workclass_train_w_corrupt_fraction = workclass_train_unfeaturized.copy()
    indexes_to_corrupt = numpy.random.permutation(workclass_train_unfeaturized.index)[
                         :int(len(workclass_train_unfeaturized) * corruption_fraction)]
    workclass_train_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = workclass_train_all_corrupt.loc[
        indexes_to_corrupt, feature]
    workclass_c02_featurizer = get_cat_featurizer()
    workclass_train_featurized_w_corrupt_fraction = workclass_c02_featurizer.fit_transform(
        workclass_train_w_corrupt_fraction)
    train_workclass_w_corrupt_fraction = numpy.hstack([workclass_train_featurized_w_corrupt_fraction, education_train,
                                                       occupation_train, age_train,
                                                       capital_gain_train, capital_loss_train, hours_per_week_train])
    # numpy.testing.assert_allclose(featurized_train, train_workclass_w_corrupt_fraction, rtol=1e-5, atol=0)
    model_workclass_c02 = SGDClassifier(loss='log')
    model_workclass_c02.fit(train_workclass_w_corrupt_fraction, train_labels)
    featurized_workclass_c02_test = workclass_c02_featurizer.transform(workclass_test_c02)
    featurized_test_w_workclass_c02 = numpy.hstack(
        [featurized_workclass_c02_test, education_test, occupation_test, age_test,
         capital_gain_test, capital_loss_test, hours_per_week_test])
    test_predict_w_workclass_c02 = model_workclass_c02.predict(featurized_test_w_workclass_c02)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict_w_workclass_c02)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict_w_workclass_c02)
    print_if_debug_and_store_iteration_results(True, True, corruption_fraction, debug, feature, iteration_results,
                                               scores)


def corrupt_age_train_and_test(age_test_c02, capital_gain_test, capital_gain_train, capital_loss_test,
                               capital_loss_train, corruption_fraction, debug, education_test, education_train, feature,
                               featurized_train, get_num_featurizer, hours_per_week_test, hours_per_week_train,
                               iteration_results, occupation_test, occupation_train, test, test_labels, train,
                               train_labels, workclass_test, workclass_train, age_train_all_corrupt):
    age_train_unfeaturized = train[[feature]]
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Train and Test")
    age_train_w_corrupt_fraction = age_train_unfeaturized.copy()
    indexes_to_corrupt = numpy.random.permutation(age_train_unfeaturized.index)[
                         :int(len(age_train_unfeaturized) * corruption_fraction)]
    age_train_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = age_train_all_corrupt.loc[
        indexes_to_corrupt, feature]
    age_c02_featurizer = get_num_featurizer()
    age_train_featurized_w_corrupt_fraction = age_c02_featurizer.fit_transform(
        age_train_w_corrupt_fraction)
    train_age_w_corrupt_fraction = numpy.hstack([workclass_train, education_train, occupation_train,
                                                 age_train_featurized_w_corrupt_fraction,
                                                 capital_gain_train, capital_loss_train, hours_per_week_train])
    # numpy.testing.assert_allclose(featurized_train, train_age_w_corrupt_fraction, rtol=1e-5, atol=0)
    model_age_c02 = SGDClassifier(loss='log')
    model_age_c02.fit(train_age_w_corrupt_fraction, train_labels)
    featurized_age_c02_test = age_c02_featurizer.transform(age_test_c02)
    featurized_test_w_age_c02 = numpy.hstack([workclass_test, education_test, occupation_test, featurized_age_c02_test,
                                              capital_gain_test, capital_loss_test, hours_per_week_test])
    test_predict_w_age_c02 = model_age_c02.predict(featurized_test_w_age_c02)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict_w_age_c02)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict_w_age_c02)
    print_if_debug_and_store_iteration_results(True, True, corruption_fraction, debug, feature, iteration_results,
                                               scores)


def corrupt_age_test_set_only(age_featurizer, age_test_all_corrupt, capital_gain_test, capital_loss_test,
                              corruption_fraction, debug, education_test, feature, featurized_test, hours_per_week_test,
                              iteration_results, model_wo_corruption, occupation_test, test, test_labels,
                              workclass_test):
    age_test = test[[feature]]
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Test")
    age_test_w_corrupt_fraction = age_test.copy()
    indexes_to_corrupt = numpy.random.permutation(age_test.index)[
                         :int(len(age_test) * corruption_fraction)]
    age_test_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = age_test_all_corrupt.loc[
        indexes_to_corrupt, feature]
    age_test_featurized_w_corrupt_fraction = age_featurizer.transform(
        age_test_w_corrupt_fraction)
    test_age_w_corrupt_fraction = numpy.hstack([workclass_test, education_test, occupation_test,
                                                age_test_featurized_w_corrupt_fraction,
                                                capital_gain_test, capital_loss_test, hours_per_week_test])
    # numpy.testing.assert_allclose(featurized_test, test_age_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict = model_wo_corruption.predict(test_age_w_corrupt_fraction)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict)
    print_if_debug_and_store_iteration_results(True, False, corruption_fraction, debug, feature, iteration_results,
                                               scores)
    return age_test_w_corrupt_fraction


def corrupt_workclass_test_set_only(workclass_featurizer, workclass_test_all_corrupt, capital_gain_test,
                                    capital_loss_test,
                                    corruption_fraction, debug, education_test, feature, featurized_test,
                                    hours_per_week_test,
                                    iteration_results, model_wo_corruption, occupation_test, test, test_labels,
                                    age_test):
    workclass_test = test[[feature]]
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Test")
    workclass_test_w_corrupt_fraction = workclass_test.copy()
    indexes_to_corrupt = numpy.random.permutation(workclass_test.index)[
                         :int(len(workclass_test) * corruption_fraction)]
    workclass_test_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = workclass_test_all_corrupt.loc[
        indexes_to_corrupt, feature]
    workclass_test_featurized_w_corrupt_fraction = workclass_featurizer.transform(
        workclass_test_w_corrupt_fraction)
    test_workclass_w_corrupt_fraction = numpy.hstack([workclass_test_featurized_w_corrupt_fraction, education_test,
                                                      occupation_test, age_test,
                                                      capital_gain_test, capital_loss_test, hours_per_week_test])
    # numpy.testing.assert_allclose(featurized_test, test_workclass_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict = model_wo_corruption.predict(test_workclass_w_corrupt_fraction)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict)
    print_if_debug_and_store_iteration_results(True, False, corruption_fraction, debug, feature, iteration_results,
                                               scores)
    return workclass_test_w_corrupt_fraction


def corrupt_education_test_set_only(education_featurizer, _, capital_gain_test,
                                    capital_loss_test,
                                    corruption_fraction, debug, workclass_test, feature, featurized_test,
                                    hours_per_week_test,
                                    iteration_results, model_wo_corruption, occupation_test, test, test_labels,
                                    age_test):
    education_test = test[[feature]]
    if debug is True:
        print("____")
        print(f"Now testing corruption of {corruption_fraction * 100}% of feature {feature}")
        print("Corruptions in Test")
    education_test_w_corrupt_fraction = education_test.copy()
    indexes_to_corrupt = numpy.random.permutation(education_test.index)[
                         :int(len(education_test) * corruption_fraction)]
    education_test_w_corrupt_fraction.loc[indexes_to_corrupt, feature] = numpy.nan
    education_test_featurized_w_corrupt_fraction = education_featurizer.transform(
        education_test_w_corrupt_fraction)
    test_education_w_corrupt_fraction = numpy.hstack([workclass_test, education_test_featurized_w_corrupt_fraction,
                                                      occupation_test, age_test,
                                                      capital_gain_test, capital_loss_test, hours_per_week_test])
    # numpy.testing.assert_allclose(featurized_test, test_education_w_corrupt_fraction, rtol=1e-5, atol=0)
    test_predict = model_wo_corruption.predict(test_education_w_corrupt_fraction)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict)
    print_if_debug_and_store_iteration_results(True, False, corruption_fraction, debug, feature, iteration_results,
                                               scores)
    return education_test_w_corrupt_fraction


def print_if_debug_and_store_iteration_results(test_corruption, train_corruption, corruption_fraction, debug, feature,
                                               iteration_results, scores):
    if debug is True:
        print(f'Accuracy on the test set: {scores["accuracy"]}')
        print(f'non_protected_fnr on the test set: {scores["non_protected_fnr"]}')
        print(f'protected_fnr on the test set: {scores["protected_fnr"]}')
    iteration_results["test_corruption"].append(test_corruption)
    iteration_results["train_corruption"].append(train_corruption)
    iteration_results["corruption_fraction"].append(corruption_fraction)
    iteration_results["feature"].append(feature)
    iteration_results["accuracy"].append(scores["accuracy"])
    iteration_results["non_protected_fnr"].append(scores["non_protected_fnr"])
    iteration_results["protected_fnr"].append(scores["protected_fnr"])


def do_credit_corruption_opt(debug):
    # TODO: Make this general enough for the last 2 params to work? Prbbly not necessary
    iteration_results = execute_credit_pipeline_opt(debug)
    return pd.DataFrame(iteration_results)


def measure_credit_corruption_opt_exec_time(debug, repeats=10):
    result = timeit.repeat(stmt=cleandoc(f"""
    execute_credit_pipeline_opt({debug})
    print("Done!")
    """),
                           setup=cleandoc(f"""
    from whatif.experiments.credit_data_corruptions_opt import execute_credit_pipeline_opt
    """),
                           repeat=repeats, number=1)
    return pd.DataFrame({"runtimes": result})


# do_credit_corruption_opt(debug=True)

# measure_credit_corruption_opt_exec_time(debug=True, repeats=2)
