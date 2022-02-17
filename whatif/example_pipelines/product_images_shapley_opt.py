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

from whatif.example_pipelines.product_images_shapley_naive import create_corrupt_data
from whatif.refinements import _data_valuation
from whatif.utils.utils import get_project_root


def execute_image_pipeline_w_shapley_opt(corrupted_row_ids: pd.DataFrame, shapley_value_cleaning=True,
                                         shapley_value_k=10, cleaning_batch_size=50, do_model_train_and_score=True,
                                         num_iterations=10):
    def decode_image(img_str):
        return np.array([int(val) for val in img_str.split(':')])

    # TODO change this to pyarrow + parquet, which can handle numpy arrays well
    train_data = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/'
                             f'product_images_corrupted.csv',
                             converters={'image': decode_image})

    enable_fact_table_row_tracking_opt(train_data)

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

    train, test = train_test_split(images_of_interest, test_size=0.2) # , random_state=random_seed_for_splitting)

    y_train = label_binarize(train['category_name'], classes=categories_to_distinguish)
    y_test = label_binarize(test['category_name'], classes=categories_to_distinguish)

    train_image_lineage_ids = save_row_tracking_information_opt(train)
    test_image_lineage_ids = save_row_tracking_information_opt(test)

    x_train = pipeline_without_model.fit_transform(train[['image']])
    x_test = pipeline_without_model.transform(test[['image']])

    # Cleaning start
    iteration_results = cleaning_with_maybe_model_training(cleaning_batch_size, corrupted_row_ids,
                                                           do_model_train_and_score,
                                                           train_image_lineage_ids,
                                                           test_image_lineage_ids,
                                                           model_without_pipeline, num_iterations,
                                                           shapley_value_cleaning, shapley_value_k, images_of_interest, x_test,
                                                           x_train, y_test, y_train, random_seed_for_splitting)

    return iteration_results


def cleaning_with_maybe_model_training(cleaning_batch_size, corrupted_row_ids, do_model_train_and_score,
                                       train_image_lineage_ids, test_image_lineage_ids, model_without_pipeline,
                                       num_iterations, shapley_value_cleaning, shapley_value_k, images_of_interest,
                                       x_test, x_train, y_test, y_train, random_seed_for_splitting):
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
    already_cleaned_rows = 0
    total_corrupted_rows = len(corrupted_row_ids)
    y_train_squeezed = np.squeeze(y_train)
    y_test_squeezed = np.squeeze(y_test)
    still_corrupted_rows = corrupted_row_ids
    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration} now...")
        if do_model_train_and_score:
            model_without_pipeline.fit(x_train, np.expand_dims(y_train_squeezed, axis=1))
            model_score = model_without_pipeline.score(x_test, np.expand_dims(y_test_squeezed, axis=1))
        else:
            model_score = None
        # Disable printing for the experiments to not spam the console
        # print(model_score)

        cleaning_results = do_shapley_value_cleaning_opt(still_corrupted_rows, train_image_lineage_ids,
                                                         already_cleaned_rows, total_corrupted_rows,
                                                         shapley_value_cleaning, images_of_interest, x_test, x_train,
                                                         y_test_squeezed,
                                                         y_train_squeezed, shapley_value_k, cleaning_batch_size)
        new_label_corrections, iteration_info, already_cleaned_rows = cleaning_results

        # Get rows that are still corrupted
        still_corrupted_rows = apply_label_corrections_to_corrupted_rows(still_corrupted_rows, new_label_corrections)
        # only count corruptions in train! If we would track statistics here, we would want to make sure
        # only train contains corruptions, like in the qualitative experiment, but thats difficult if we want to
        # still have a train_test_split
        y_train_squeezed = apply_label_corrections_to_train_data_opt(new_label_corrections, y_train_squeezed,
                                                                     train_image_lineage_ids)
        # TODO: Redo train test split, continue and make sure this works
        # error: does not work because x train is not one dimensional
        big_x = np.concatenate([x_train, x_test])
        big_y = np.concatenate([y_train_squeezed, y_test_squeezed])
        big_lineage = np.concatenate([train_image_lineage_ids, test_image_lineage_ids])
        x_train, x_test, y_train_squeezed, y_test_squeezed, train_image_lineage_ids, test_image_lineage_ids = \
            train_test_split(big_x, big_y, big_lineage, test_size=0.2)  #, random_state=random_seed_for_splitting)

        iteration_info['model_score'] = model_score

        iteration_results["iteration"].append(iteration)
        iteration_results["already_cleaned_rows"].append(iteration_info["already_cleaned_rows"])
        iteration_results["total_corrupted_rows"].append(iteration_info["total_corrupted_rows"])
        iteration_results["false_corruption_alarm"].append(iteration_info["false_corruption_alarm"])
        iteration_results["correct_corruption_alarm"].append(iteration_info["correct_corruption_alarm"])
        iteration_results["corruption_not_detected_yet"].append(iteration_info["corruption_not_detected_yet"])
        iteration_results["fraction_data_cleaned"].append(iteration_info["fraction_data_cleaned"])
        iteration_results["model_score"].append(iteration_info["model_score"])
    return iteration_results


def apply_label_corrections_to_corrupted_rows(corrupted_row_ids, label_corrections):
    corrupted_row_ids = corrupted_row_ids.merge(label_corrections, how='outer', on='image_lineage_id',
                                                indicator=True)
    corrupted_row_ids = corrupted_row_ids[corrupted_row_ids['_merge'] == 'left_only']
    corrupted_row_ids = corrupted_row_ids.drop(columns=['_merge'])
    return corrupted_row_ids


def save_row_tracking_information_opt(train):
    # Let us record the fact table identifiers so we can build the label corrections map
    image_lineage_ids = train["image_lineage_id"]
    return image_lineage_ids


def apply_label_corrections_to_train_data_opt(label_corrections, y_train, image_lineage_ids):
    # Apply label corrections
    y_train_df = pd.DataFrame({"label": y_train, "image_lineage_id": image_lineage_ids})
    y_train_df = y_train_df.merge(label_corrections, how='left', on='image_lineage_id', indicator=True)
    y_train_df.loc[y_train_df['_merge'] != 'left_only', 'label'] = \
        (y_train_df.loc[y_train_df['_merge'] != 'left_only', 'label'] + 1) % 2
    y_train_df.set_index('image_lineage_id')
    y_train_df = y_train_df.sort_index()
    y_train_df['label'] = y_train_df['label'].astype(int)

    # Not category name is used
    return y_train_df['label'].to_numpy()


def enable_fact_table_row_tracking_opt(train_data):
    # We need some way to identify fact table rows throughout the pipeline and across runs
    train_data['image_lineage_id'] = range(0, len(train_data))


def do_shapley_value_cleaning_opt(corrupted_row_ids, image_lineage_ids,
                                  already_cleaned_rows, total_corrupted_rows,
                                  shapley_value_cleaning,
                                  images_of_interest, x_test, x_train, y_test, y_train, shapley_value_k,
                                  cleaning_batch_size):
    iteration_info = {}
    shapley_values = _data_valuation._compute_shapley_values(x_train,
                                                             y_train,
                                                             x_test,
                                                             y_test,
                                                             shapley_value_k)
    df_with_id_and_shapley_value = pd.DataFrame(
        {"image_lineage_id": image_lineage_ids, "shapley_value": shapley_values})
    if shapley_value_cleaning is True:
        rows_to_fix = df_with_id_and_shapley_value.nsmallest(cleaning_batch_size, "shapley_value")
    else:
        rows_to_fix = df_with_id_and_shapley_value.sample(n=cleaning_batch_size, replace=False)
    joined_rows_to_fix = images_of_interest.merge(rows_to_fix, on="image_lineage_id")  # Issue is here, not all corruptions are in train
    # Show problematic imgs with labels
    # for row_index, row in list(joined_rows_to_fix.iterrows())[0:3]:
    #     from matplotlib import pyplot as plt
    #     print(f"""row["category_name"]: {row["category_name"]}""")
    #     plt.imshow(np.reshape(row["image"], (28, 28, 1)), interpolation='nearest')
    #     plt.show()
    # print(rows_to_fix)
    # fix labels:
    iteration_info["already_cleaned_rows"] = already_cleaned_rows
    iteration_info["total_corrupted_rows"] = total_corrupted_rows
    assert (total_corrupted_rows - len(corrupted_row_ids)) == already_cleaned_rows
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

    assert false_corruption_alarm + correct_corruption_alarm == cleaning_batch_size

    already_cleaned_rows += correct_corruption_alarm
    fraction_data_cleaned = already_cleaned_rows / total_corrupted_rows
    iteration_info["fraction_data_cleaned"] = fraction_data_cleaned


    new_label_corrections = corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'both'].copy()
    new_label_corrections = new_label_corrections[['image_lineage_id']]
    new_label_corrections['image_lineage_id'] = new_label_corrections['image_lineage_id'].astype(int)

    return new_label_corrections, iteration_info, already_cleaned_rows


def do_shapley_value_opt(corruption_fraction, num_iterations, use_shapley_weighting, shapley_value_k,
                         cleaning_batch_size, do_model_train_and_score):
    corrupted_row_ids = create_corrupt_data(corruption_fraction)

    iteration_results = execute_image_pipeline_w_shapley_opt(corrupted_row_ids,
                                                             use_shapley_weighting,
                                                             shapley_value_k,
                                                             cleaning_batch_size,
                                                             do_model_train_and_score,
                                                             num_iterations)

    print("Done!")
    return pd.DataFrame(iteration_results)


def measure_shapley_opt_exec_time(corruption_fraction, num_iterations, use_shapley_weighting, shapley_value_k,
                                  cleaning_batch_size, do_model_train_and_score, repeats=10):
    result = timeit.repeat(stmt=cleandoc(f"""
    iteration_results = execute_image_pipeline_w_shapley_opt(corrupted_row_ids,
                                                            {use_shapley_weighting},
                                                            {shapley_value_k},
                                                            {cleaning_batch_size},
                                                            {do_model_train_and_score},
                                                            {num_iterations})
    print("Done!")
    """),
                           setup=cleandoc(f"""
    from whatif.example_pipelines.product_images_shapley_naive import create_corrupt_data
    from whatif.example_pipelines.product_images_shapley_opt import execute_image_pipeline_w_shapley_opt
    import pandas as pd
    
    corrupted_row_ids = create_corrupt_data({corruption_fraction})
    """),
                           repeat=repeats, number=1)
    return pd.DataFrame({"runtimes": result})
