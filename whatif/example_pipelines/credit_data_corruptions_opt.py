import numpy
import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer

from whatif.example_pipelines.credit_data_corruptions_naive import compute_fairness_metric
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
        adult_test = pd.read_csv(adult_test_location, names=columns, sep=', ', engine='python', na_values="?", skiprows=1)

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

    train_location = f'{str(get_project_root())}/whatif/example_pipelines/datasets/income/adult.data'
    test_location = f'{str(get_project_root())}/whatif/example_pipelines/datasets/income/adult.test'

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
    workclass_featurizater = get_cat_featurizer()
    education_featurizater = get_cat_featurizer()
    occupation_featurizer = get_cat_featurizer()
    age_featurizer = get_num_featurizer()
    capital_gain_featurizer = get_num_featurizer()
    capital_loss_featurizer = get_num_featurizer()
    hours_per_week_featurizer = get_num_featurizer()

    workclass_train = workclass_featurizater.fit_transform(train[['workclass']])
    education_train = education_featurizater.fit_transform(train[['education']])
    occupation_train = occupation_featurizer.fit_transform(train[['occupation']])
    age_featurizer_train = age_featurizer.fit_transform(train[['age']])
    capital_gain_train = capital_gain_featurizer.fit_transform(train[['capital-gain']])
    capital_loss_train = capital_loss_featurizer.fit_transform(train[['capital-loss']])
    hours_per_week_train = hours_per_week_featurizer.fit_transform(train[['hours-per-week']])

    featurized_train = numpy.hstack([workclass_train, education_train, occupation_train, age_featurizer_train,
                                     capital_gain_train, capital_loss_train, hours_per_week_train])

    model = SGDClassifier(loss='log')

    model = model.fit(featurized_train, train_labels)

    workclass_test = workclass_featurizater.transform(test[['workclass']])
    education_test = education_featurizater.transform(test[['education']])
    occupation_test = occupation_featurizer.transform(test[['occupation']])
    age_featurizer_test = age_featurizer.transform(test[['age']])
    capital_gain_test = capital_gain_featurizer.transform(test[['capital-gain']])
    capital_loss_test = capital_loss_featurizer.transform(test[['capital-loss']])
    hours_per_week_test = hours_per_week_featurizer.transform(test[['hours-per-week']])

    featurized_test = numpy.hstack([workclass_test, education_test, occupation_test, age_featurizer_test,
                                     capital_gain_test, capital_loss_test, hours_per_week_test])
    test_predict = model.predict(featurized_test)
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


def do_credit_corruption_opt(debug):
    # TODO: Make this general enough for the last 2 params to work? Prbbly not necessary
    iteration_results = execute_credit_pipeline_opt(debug)
    return pd.DataFrame(iteration_results)


do_credit_corruption_opt(debug=True)
