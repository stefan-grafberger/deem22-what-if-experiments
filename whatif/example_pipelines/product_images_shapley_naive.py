import timeit
from inspect import cleandoc

import numpy as np
import pandas as pd

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, label_binarize
from sklearn.pipeline import Pipeline

from whatif.refinements import _data_valuation
from whatif.utils.utils import get_project_root


def execute_image_pipeline_w_shapley_naive(corrupted_row_ids: pd.DataFrame, label_corrections: pd.DataFrame,
                                           shapley_value_cleaning=True, shapley_value_k=10,
                                           cleaning_batch_size=50, do_model_train_and_score=True):
    def decode_image(img_str):
        return np.array([int(val) for val in img_str.split(':')])

    # fuer qualitaetsvariante 2 csv damit accuracy stimmt in plots

    # TODO change this to pyarrow + parquet, which can handle numpy arrays well
    train_data = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/'
                             f'product_images_corrupted.csv',
                             converters={'image': decode_image})

    enable_fact_table_row_tracking_naive(train_data)

    train_data = apply_label_corrections_to_train_data_naive(label_corrections, train_data)

    product_categories = pd.read_csv(
        f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/product_categories.csv')
    # Binary classification, let us assume we have label mapping always, but think about this again
    # train_data['category_lineage_id'] = range(0, len(train_data))
    with_categories = train_data.merge(product_categories, on='category_id')

    categories_to_distinguish = ['Sneaker', 'Ankle boot']

    images_of_interest = with_categories[with_categories['category_name'].isin(categories_to_distinguish)]

    def normalise_image(images):
        return images / 255.0

    def reshape_images(images):
        return np.concatenate(images['image'].values) \
            .reshape(images.shape[0], 28, 28, 1)

    def create_cnn():
        model = Sequential([
            Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=2),
            Dropout(0.3),
            Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    pipeline_without_model = Pipeline(steps=[
        ('normalisation', FunctionTransformer(normalise_image)),
        ('reshaping', FunctionTransformer(reshape_images))
    ])

    model_without_pipeline = KerasClassifier(create_cnn)

    random_seed_for_splitting = 1337
    # if len(sys.argv) > 1:
    #     random_seed_for_splitting = int(sys.argv[1])

    train, test = train_test_split(images_of_interest, test_size=0.2, random_state=random_seed_for_splitting)

    y_train = label_binarize(train['category_name'], classes=categories_to_distinguish)
    y_test = label_binarize(test['category_name'], classes=categories_to_distinguish)

    image_lineage_ids = save_row_tracking_information_naive(train)

    x_train = pipeline_without_model.fit_transform(train[['image']])  # beide varianten mal ausprobieren, unklar was fair
    x_test = pipeline_without_model.transform(test[['image']])

    if do_model_train_and_score:
        model_without_pipeline.fit(x_train, y_train)
        model_score = model_without_pipeline.score(x_test, y_test)
    else:
        model_score = None
    # Disable printing for the experiments to not spam the console
    # print(model_score)

    cleaning_results = do_shapley_value_cleaning_naive(corrupted_row_ids, image_lineage_ids, label_corrections,
                                                       shapley_value_cleaning, train, x_test, x_train,
                                                       y_test,
                                                       y_train, shapley_value_k, cleaning_batch_size)
    label_corrections, iteration_info = cleaning_results
    iteration_info['model_score'] = model_score
    return cleaning_results


def save_row_tracking_information_naive(train):
    # Let us record the fact table identifiers so we can build the label corrections map
    image_lineage_ids = train["image_lineage_id"]
    return image_lineage_ids


def apply_label_corrections_to_train_data_naive(label_corrections, train_data):
    # Apply label corrections
    train_data = train_data.merge(label_corrections, how='left', on='image_lineage_id', indicator=True)
    train_data.loc[train_data['_merge'] == 'left_only', 'category_id'] = train_data.loc[
        train_data['_merge'] == 'left_only', 'category_id_x']
    train_data.loc[train_data['_merge'] != 'left_only', 'category_id'] = train_data.loc[
        train_data['_merge'] != 'left_only', 'category_id_y']
    train_data.set_index('image_lineage_id')
    train_data = train_data.sort_index()
    train_data['category_id'] = train_data['category_id'].astype(int)
    train_data = train_data.drop(columns=['_merge'])
    return train_data


def enable_fact_table_row_tracking_naive(train_data):
    # We need some way to identify fact table rows throughout the pipeline and across runs
    train_data['image_lineage_id'] = range(0, len(train_data))


def do_shapley_value_cleaning_naive(corrupted_row_ids, image_lineage_ids, label_corrections, shapley_value_cleaning,
                                    train, x_test, x_train, y_test, y_train, shapley_value_k,
                                    cleaning_batch_size):
    iteration_info = {}
    shapley_values = _data_valuation._compute_shapley_values(x_train,
                                                             np.squeeze(y_train),
                                                             x_test,
                                                             np.squeeze(y_test),
                                                             shapley_value_k)
    df_with_id_and_shapley_value = pd.DataFrame(
        {"image_lineage_id": image_lineage_ids, "shapley_value": shapley_values})
    if shapley_value_cleaning is True:
        rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
    else:
        rows_to_fix = df_with_id_and_shapley_value.sample(n=cleaning_batch_size, replace=False)
    joined_rows_to_fix = train.merge(rows_to_fix, on="image_lineage_id")
    # Show problematic imgs with labels
    # for row_index, row in list(joined_rows_to_fix.iterrows())[0:3]:
    #     from matplotlib import pyplot as plt
    #     print(f"""row["category_name"]: {row["category_name"]}""")
    #     plt.imshow(np.reshape(row["image"], (28, 28, 1)), interpolation='nearest')
    #     plt.show()
    # print(rows_to_fix)
    # fix labels:
    already_cleaned_rows = len(label_corrections)
    iteration_info["already_cleaned_rows"] = already_cleaned_rows
    total_corrupted_rows = len(corrupted_row_ids)
    iteration_info["total_corrupted_rows"] = total_corrupted_rows
    # Get rows that are still corrupted
    corrupted_row_ids = corrupted_row_ids.merge(label_corrections, how='outer', on='image_lineage_id', indicator=True)
    corrupted_row_ids = corrupted_row_ids[corrupted_row_ids['_merge'] == 'left_only']
    corrupted_row_ids = corrupted_row_ids.drop(columns=['_merge'])
    assert (total_corrupted_rows - len(corrupted_row_ids)) == len(label_corrections)
    corrupted_row_ids.image_lineage_id = corrupted_row_ids.image_lineage_id.astype(int)
    corrections_and_corrupted = joined_rows_to_fix.merge(corrupted_row_ids, how='outer', on="image_lineage_id",
                                                         indicator=True)
    # Rows to fix that were not corrupted
    false_corruption_alarm = len(corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'left_only'])
    iteration_info["false_corruption_alarm"] = false_corruption_alarm
    correct_corruption_alarm = len(corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'both'])
    iteration_info["correct_corruption_alarm"] = correct_corruption_alarm
    corruption_not_detected_yet = len(corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'right_only'])
    iteration_info["corruption_not_detected_yet"] = corruption_not_detected_yet
    new_label_corrections = corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'both'].copy()
    if len(new_label_corrections) != 0:
        new_label_corrections.loc[new_label_corrections.category_name == 'Sneaker', 'category_id'] = 9
        new_label_corrections.loc[new_label_corrections.category_name == 'Ankle boot', 'category_id'] = 7
    else:
        new_label_corrections['category_id'] = None
    new_label_corrections = new_label_corrections[['image_lineage_id', 'category_id']]
    label_corrections = pd.concat([label_corrections, new_label_corrections])
    label_corrections['image_lineage_id'] = label_corrections['image_lineage_id'].astype(int)
    label_corrections['category_id'] = label_corrections['category_id'].astype(int)
    already_cleaned_rows += correct_corruption_alarm
    fraction_data_cleaned = already_cleaned_rows / total_corrupted_rows
    iteration_info["fraction_data_cleaned"] = fraction_data_cleaned

    return label_corrections, iteration_info


def create_corrupt_data(corruption_fraction=0.5):
    # TODO change this to pyarrow + parquet, which can handle numpy arrays well
    train_data = pd.read_csv(
        f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/product_images.csv')
    # We need some way to identify fact table rows throughout the pipeline and across runs
    train_data = train_data.reset_index(drop=False)
    train_data = train_data.rename(columns={'index': 'image_lineage_id'})

    product_categories = pd.read_csv(
        f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/product_categories.csv')

    # Binary classification, let us assume we have label mapping always, but think about this again
    # train_data['category_lineage_id'] = range(0, len(train_data))
    with_categories = train_data.merge(product_categories, on='category_id')
    categories_to_distinguish = ['Sneaker', 'Ankle boot']
    images_of_interest = with_categories[with_categories['category_name'].isin(categories_to_distinguish)]
    rows_to_corrupt = images_of_interest.sample(frac=corruption_fraction, replace=False)
    rows_to_corrupt.loc[rows_to_corrupt['category_name'] == 'Sneaker', 'category_id'] = 9
    rows_to_corrupt.loc[rows_to_corrupt['category_name'] == 'Ankle boot', 'category_id'] = 7

    # apply corruption
    corrupted_train_data = train_data.merge(rows_to_corrupt, how='left', on='image_lineage_id', indicator=True)
    corrupted_train_data.loc[:, 'image'] = corrupted_train_data.loc[:, 'image_x']
    corrupted_train_data.loc[corrupted_train_data['_merge'] == 'left_only', 'category_id'] = corrupted_train_data.loc[
        corrupted_train_data['_merge'] == 'left_only', 'category_id_x']
    corrupted_train_data.loc[corrupted_train_data['_merge'] != 'left_only', 'category_id'] = corrupted_train_data.loc[
        corrupted_train_data['_merge'] != 'left_only', 'category_id_y']
    corrupted_train_data.set_index('image_lineage_id')
    corrupted_train_data = corrupted_train_data.sort_index()
    corrupted_train_data['category_id'] = corrupted_train_data['category_id'].astype(int)

    corrupted_df_to_save = corrupted_train_data[['image', 'category_id']]
    corrupted_df_to_save.to_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/'
                                f'sneakers/product_images_corrupted.csv', index=False)

    corrupted_row_ids = rows_to_corrupt[['image_lineage_id']]
    corrupted_row_ids = corrupted_row_ids.sort_index()
    corrupted_row_ids.to_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/'
                             f'sneakers/product_image_ids_corrupted.csv', index=False)
    return corrupted_row_ids


def do_shapley_value_naive(corruption_fraction, num_iterations, use_shapley_weighting, shapley_value_k,
                           cleaning_batch_size, do_model_train_and_score):
    corrupted_row_ids = create_corrupt_data(corruption_fraction)
    label_corrections = pd.DataFrame({'image_lineage_id': [], "category_id": []})
    iteration_results = {
        "iteration": [],
        "already_cleaned_rows": [],
        "total_corrupted_rows": [],
        "false_corruption_alarm": [],
        "correct_corruption_alarm": [],
        "corruption_not_detected_yet": [],
        "fraction_data_cleaned": [],
        "model_score": []
    }
    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration} now...")
        label_corrections, total_updates, iteration_info = execute_image_pipeline_w_shapley_naive(corrupted_row_ids,
                                                                                                  label_corrections,
                                                                                                  use_shapley_weighting,
                                                                                                  shapley_value_k,
                                                                                                  cleaning_batch_size)
        iteration_results["iteration"].append(iteration)
        iteration_results["already_cleaned_rows"].append(iteration_info["already_cleaned_rows"])
        iteration_results["total_corrupted_rows"].append(iteration_info["total_corrupted_rows"])
        iteration_results["false_corruption_alarm"].append(iteration_info["false_corruption_alarm"])
        iteration_results["correct_corruption_alarm"].append(iteration_info["correct_corruption_alarm"])
        iteration_results["corruption_not_detected_yet"].append(iteration_info["corruption_not_detected_yet"])
        iteration_results["fraction_data_cleaned"].append(iteration_info["fraction_data_cleaned"])
        iteration_results["model_score"].append(iteration_info["model_score"])

    print("Done!")
    return pd.DataFrame(iteration_results)


def print_legend():
    print("iteration: In each iteration, we execute the pipeline and clean #cleaning_batch_size rows afterwards, "
          "either using shapley values or random sampling. Param for this: $use_shapley_weighting")
    print("already_cleaned_rows: Already cleaned rows")
    print("total_corrupted_rows: Total corrupted rows")
    print("false_corruption_alarm: False alarms in that iteration for #rows that were not corrupted")
    print("correct_corruption_alarm: Correct alarms in that iteration for #rows that were corrupted")
    print("corruption_not_detected_yet: In that and previous iterations, there were corrupted #rows not discover yet")
    print("fraction_data_cleaned: fraction of corrupted data that was cleaned already")
    print("model_score: Model score on current data with potential left-over corruptions")


def measure_shapley_naive_exec_time(corruption_fraction, num_iterations, use_shapley_weighting, shapley_value_k,
                                    cleaning_batch_size, do_model_train_and_score, repeats=10):
    result = timeit.repeat(stmt=cleandoc(f"""
    for iteration in range({num_iterations}):
        print(f"Starting iteration {{iteration}} now...")
        label_corrections, total_updates, iteration_info = execute_image_pipeline_w_shapley_naive(corrupted_row_ids,
                                                                                            label_corrections,
                                                                                            total_updates,
                                                                                            {use_shapley_weighting},
                                                                                            {shapley_value_k},
                                                                                            {cleaning_batch_size},
                                                                                            {do_model_train_and_score})
        iteration_results["iteration"].append(iteration)
        iteration_results["already_cleaned_rows"].append(iteration_info["already_cleaned_rows"])
        iteration_results["total_corrupted_rows"].append(iteration_info["total_corrupted_rows"])
        iteration_results["false_corruption_alarm"].append(iteration_info["false_corruption_alarm"])
        iteration_results["correct_corruption_alarm"].append(iteration_info["correct_corruption_alarm"])
        iteration_results["corruption_not_detected_yet"].append(iteration_info["corruption_not_detected_yet"])
        iteration_results["fraction_data_cleaned"].append(iteration_info["fraction_data_cleaned"])
        iteration_results["model_score"].append(iteration_info["model_score"])

    print("Done!")
    
    """),
                           setup=cleandoc(f"""
    from whatif.example_pipelines.product_images_shapley_naive import create_corrupt_data, execute_image_pipeline_w_shapley_naive
    import pandas as pd
    
    corrupted_row_ids = create_corrupt_data({corruption_fraction})
    label_corrections = pd.DataFrame({{'image_lineage_id': [], "category_id": []}})
    total_updates = 0
    
    iteration_results = {{
        "iteration": [],
        "already_cleaned_rows": [],
        "total_corrupted_rows": [],
        "false_corruption_alarm": [],
        "correct_corruption_alarm": [],
        "corruption_not_detected_yet": [],
        "fraction_data_cleaned": [],
        "model_score": []
    }}
    """),
                           repeat=repeats, number=1)
    return pd.DataFrame({"runtimes": result})
