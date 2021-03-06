import timeit
from inspect import cleandoc

import pandas as pd
import numpy as np
from jenga.corruptions.generic import CategoricalShift, MissingValues
from jenga.corruptions.numerical import Scaling

from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer

from whatif.utils.utils import get_project_root


def execute_credit_pipeline_naive(corrupt_train, corrupt_test, corruption_fraction, corrupt_feature, debug):
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

        if corrupt_train is True:
            adult_train = corrupt_data(adult_train, corruption_fraction, corrupt_feature)
        if corrupt_test is True:
            adult_test = corrupt_data(adult_test, corruption_fraction, corrupt_feature)

        return adult_train, adult_test

    def create_feature_encoding_pipeline():

        def safe_log(x):
            return np.log(x, out=np.zeros_like(x), where=(x != 0))

        impute_and_one_hot_encode = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        impute_and_scale = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('log_transform', FunctionTransformer(lambda x: safe_log(x))),
            ('scale', StandardScaler())
        ])

        featurisation = ColumnTransformer(transformers=[
            ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['workclass', 'education', 'occupation']),
            ('impute_and_scale', impute_and_scale, ['age', 'capital-gain', 'capital-loss', 'hours-per-week']),
        ], remainder='drop')

        return featurisation

    def extract_labels(adult_train, adult_test):
        adult_train_labels = label_binarize(adult_train['income-per-year'], classes=['<=50K', '>50K'])
        # The test data has a dot in the class names for some reason...
        adult_test_labels = label_binarize(adult_test['income-per-year'], classes=['<=50K.', '>50K.'])

        return adult_train_labels, adult_test_labels

    def create_training_pipeline():
        featurisation = create_feature_encoding_pipeline()
        return Pipeline([
            ('features', featurisation),
            ('learner', SGDClassifier(loss='log'))
        ])

    train_location = f'{str(get_project_root())}/whatif/experiments/datasets/income/adult.data'
    test_location = f'{str(get_project_root())}/whatif/experiments/datasets/income/adult.test'

    government_employed = ['Federal-gov', 'State-gov']

    train, test = load_train_and_test_data(train_location, test_location, excluded_employment_types=government_employed)

    train_labels, test_labels = extract_labels(train, test)
    pipeline = create_training_pipeline()

    model = pipeline.fit(train, train_labels.ravel())

    test_predict = model.predict(test)
    scores = {}
    scores['accuracy'] = accuracy_score(test_labels, test_predict)
    scores['non_protected_fnr'], scores['protected_fnr'] = compute_fairness_metric("race", "White", test, test_labels,
                                                                                   test_predict)
    if debug is True:
        print(f'Accuracy on the test set: {scores["accuracy"]}')
        print(f'non_protected_fnr on the test set: {scores["non_protected_fnr"]}')
        print(f'protected_fnr on the test set: {scores["protected_fnr"]}')
    return scores


def corrupt_data(data, corruption_fraction, corrupt_feature):
    # Data corruptions, later: test them one by one
    if corrupt_feature == "workclass":
        data = CategoricalShift(column='workclass', fraction=corruption_fraction).transform(data)
    elif corrupt_feature == "age":
        data = Scaling(column='age', fraction=corruption_fraction).transform(data)
    elif corrupt_feature == "education":
        data = MissingValues(column='education', fraction=corruption_fraction).transform(data)
    else:
        assert False

    return data


def compute_fairness_metric(sensitive_attribute, non_protected_class, test_x, test_y, y_pred):
    metric_calc_df = pd.DataFrame({'sensitive_attribute': test_x[sensitive_attribute],
                                   'pred': y_pred,
                                   'label': np.squeeze(test_y)})

    non_protected = metric_calc_df[metric_calc_df['sensitive_attribute'] == non_protected_class]
    protected = metric_calc_df[metric_calc_df['sensitive_attribute'] != non_protected_class]

    non_protected_false_negatives = len(non_protected[(non_protected['pred'] == 1.0) & (non_protected['label'] == 0.0)])
    non_protected_true_positives = len(non_protected[(non_protected['pred'] == 1.0) & (non_protected['label'] == 1.0)])
    # non_protected_true_negatives = 0
    # non_protected_false_positives = 0

    protected_false_negatives = len(protected[(protected['pred'] == 1.0) & (protected['label'] == 0.0)])
    protected_true_positives = len(protected[(protected['pred'] == 1.0) & (protected['label'] == 1.0)])
    # protected_true_negatives = 0
    # protected_false_positives = 0

    # False negative rates (as example)
    non_protected_fnr = float(non_protected_false_negatives) / \
                        (float(non_protected_false_negatives) + float(non_protected_true_positives))
    protected_fnr = float(protected_false_negatives) / \
                    (float(protected_false_negatives) + float(protected_true_positives))

    return non_protected_fnr, protected_fnr


def do_credit_corruption_naive(debug, corruption_percentages, corrupt_features):
    iteration_results = {
        "test_corruption": [],
        "train_corruption": [],
        "corruption_fraction": [],
        "feature": [],
        "accuracy": [],
        "non_protected_fnr": [],
        "protected_fnr": []
    }
    if debug is True:
        print("No corruptions")
    scores = execute_credit_pipeline_naive(False, False, 0.0, None, debug)
    iteration_results["test_corruption"].append(False)
    iteration_results["train_corruption"].append(False)
    iteration_results["corruption_fraction"].append(None)
    iteration_results["feature"].append(None)
    iteration_results["accuracy"].append(scores["accuracy"])
    iteration_results["non_protected_fnr"].append(scores["non_protected_fnr"])
    iteration_results["protected_fnr"].append(scores["protected_fnr"])
    for corrupt_feature in corrupt_features:
        for corruption_fraction in corruption_percentages:
            if debug is True:
                print("____")
                print(f"Now testing corruption of {corruption_fraction * 100}% of feature {corrupt_feature}")
                print("Corruptions in Test")
            scores = execute_credit_pipeline_naive(False, True, corruption_fraction, corrupt_feature, debug)
            iteration_results["test_corruption"].append(False)
            iteration_results["train_corruption"].append(True)
            iteration_results["corruption_fraction"].append(corruption_fraction)
            iteration_results["feature"].append(corrupt_feature)
            iteration_results["accuracy"].append(scores["accuracy"])
            iteration_results["non_protected_fnr"].append(scores["non_protected_fnr"])
            iteration_results["protected_fnr"].append(scores["protected_fnr"])
            if debug is True:
                print("Corruptions in Train and test")
            scores = execute_credit_pipeline_naive(True, True, corruption_fraction, corrupt_feature, debug)
            iteration_results["test_corruption"].append(True)
            iteration_results["train_corruption"].append(True)
            iteration_results["corruption_fraction"].append(corruption_fraction)
            iteration_results["feature"].append(corrupt_feature)
            iteration_results["accuracy"].append(scores["accuracy"])
            iteration_results["non_protected_fnr"].append(scores["non_protected_fnr"])
            iteration_results["protected_fnr"].append(scores["protected_fnr"])
    return pd.DataFrame(iteration_results)


def measure_credit_corruption_naive_exec_time(debug, corruption_percentages, corrupt_features, repeats=10):
    result = timeit.repeat(stmt=cleandoc(f"""
    if {debug} is True:
        print("No corruptions")
    execute_credit_pipeline_naive(False, False, 0.0, 0, {debug})
    for corrupt_feature in {corrupt_features}:
        for corruption_fraction in {corruption_percentages}:
            if {debug} is True:
                print("____")
                print(f"Now testing corruption of {{corruption_fraction*100}}% of feature {{corrupt_feature}}")
                print("Corruptions in Test")
            execute_credit_pipeline_naive(False, True, corruption_fraction, corrupt_feature, {debug})
            if {debug} is True:
                print("Corruptions in Train and test")
            execute_credit_pipeline_naive(True, True, corruption_fraction, corrupt_feature, {debug})
    print("Done!")
    """),
                           setup=cleandoc(f"""
    from whatif.experiments.credit_data_corruptions_naive import execute_credit_pipeline_naive
    """),
                           repeat=repeats, number=1)
    return pd.DataFrame({"runtimes": result})


# do_credit_corruption_naive(debug=True, corruption_percentages=[0.2, 0.5, 0.9],
#                            corrupt_features=["age", "workclass", "education"])

# measure_credit_corruption_naive_exec_time(debug=True,
#                                           corruption_percentages=[0.2, 0.5, 0.9],
#                                           corrupt_features=["age", "workclass", "education"],
#                                           repeats=2)
