import numpy as np
import pandas as pd
import sys

from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, label_binarize
from sklearn.pipeline import Pipeline

from whatif.refinements import _data_valuation
from whatif.utils.utils import get_project_root


def execute_image_pipeline_w_shapley(corrupted_row_ids: pd.DataFrame, label_corrections: pd.DataFrame,
                                     total_updates: int, shapley_value_cleaning=True):
    def decode_image(img_str):
        return np.array([int(val) for val in img_str.split(':')])

    # TODO change this to pyarrow + parquet, which can handle numpy arrays well
    train_data = pd.read_csv(f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/'
                             f'product_images_corrupted.csv',
                             converters={'image': decode_image})

    # We need some way to identify fact table rows throughout the pipeline and across runs
    train_data['image_lineage_id'] = range(0, len(train_data))

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
    #

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
    if len(sys.argv) > 1:
        random_seed_for_splitting = int(sys.argv[1])

    train, test = train_test_split(images_of_interest, test_size=0.2, random_state=random_seed_for_splitting)

    y_train = label_binarize(train['category_name'], classes=categories_to_distinguish)
    y_test = label_binarize(test['category_name'], classes=categories_to_distinguish)

    # Let us record the fact table identifiers so we can build the label corrections map
    image_lineage_ids = train["image_lineage_id"]
    x_train = pipeline_without_model.fit_transform(train[['image']])
    model_without_pipeline.fit(x_train, y_train)

    x_test = pipeline_without_model.transform(test[['image']])
    print(model_without_pipeline.score(x_test, y_test))

    print("Shapley values")
    shapley_values = _data_valuation._compute_shapley_values(x_train,
                                                             np.squeeze(y_train),
                                                             x_test,
                                                             np.squeeze(y_test),
                                                             10)
    df_with_id_and_shapley_value = pd.DataFrame(
        {"image_lineage_id": image_lineage_ids, "shapley_value": shapley_values})
    if shapley_value_cleaning is True:
        rows_to_fix = df_with_id_and_shapley_value.nsmallest(50, "shapley_value")
    else:
        rows_to_fix = df_with_id_and_shapley_value.sample(n=50, replace=False)
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
    print(f"Already cleaned rows: {already_cleaned_rows}#")
    total_corrupted_rows = len(corrupted_row_ids)
    print(f"Total corrupted rows: {total_corrupted_rows}#")

    # Get rows that are still corrupted
    corrupted_row_ids = corrupted_row_ids.merge(label_corrections, how='outer', on='image_lineage_id', indicator=True)
    corrupted_row_ids = corrupted_row_ids[corrupted_row_ids['_merge'] == 'left_only']
    corrupted_row_ids = corrupted_row_ids.drop(columns=['_merge'])
    assert (total_corrupted_rows - len(corrupted_row_ids)) == len(label_corrections)

    corrupted_row_ids.image_lineage_id = corrupted_row_ids.image_lineage_id.astype(int)
    corrections_and_corrupted = joined_rows_to_fix.merge(corrupted_row_ids, how='outer', on="image_lineage_id", indicator=True)
    # Rows to fix that were not corrupted

    false_corruption_alarm = len(corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'left_only'])
    print(f"False alarm for {false_corruption_alarm}# rows that were not corrupted")
    correct_corruption_alarm = len(corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'both'])
    print(f"Correctly detected {correct_corruption_alarm}# rows that were corrupted in iteration")
    corruption_not_detected_yet = len(corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'right_only'])
    print(f"Did not yet detect {corruption_not_detected_yet}# rows that were corrupted in iteration")

    new_label_corrections = corrections_and_corrupted[corrections_and_corrupted['_merge'] == 'both']
    if len(new_label_corrections) != 0:
        new_label_corrections.loc[new_label_corrections['category_name'] == 'Sneaker', 'category_id'] = 9
        new_label_corrections.loc[new_label_corrections['category_name'] == 'Ankle boot', 'category_id'] = 7
    else:
        new_label_corrections['category_id'] = None
    new_label_corrections = new_label_corrections[['image_lineage_id', 'category_id']]

    label_corrections = pd.concat([label_corrections, new_label_corrections])
    label_corrections.image_lineage_id = label_corrections.image_lineage_id.astype(int)
    label_corrections.category_id = label_corrections.category_id.astype(int)

    total_updates += correct_corruption_alarm
    fraction_data_cleaned = total_updates / total_corrupted_rows
    print(f"Fraction of cleaned data: {fraction_data_cleaned}")

    return label_corrections, total_updates


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


corrupted_row_ids = create_corrupt_data(0.2)
label_corrections = pd.DataFrame({'image_lineage_id': [], "category_id": []})
total_updates = 0
for _ in range(40):
    label_corrections, total_updates = execute_image_pipeline_w_shapley(corrupted_row_ids, label_corrections,
                                                                        total_updates, True)
