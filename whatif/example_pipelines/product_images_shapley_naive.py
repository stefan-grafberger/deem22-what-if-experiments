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


def execute_image_pipeline_w_shapley():
    def decode_image(img_str):
        return np.array([int(val) for val in img_str.split(':')])

    # TODO change this to pyarrow + parquet, which can handle numpy arrays well
    train_data = pd.read_csv( f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/product_images.csv', converters={'image': decode_image})
    # We need some way to identify fact table rows throughout the pipeline and across runs
    train_data['image_lineage_id'] = range(0, len(train_data))

    product_categories = pd.read_csv( f'{str(get_project_root())}/whatif/example_pipelines/datasets/sneakers/product_categories.csv')
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
    image_lineage_ids = train['image_lineage_id']
    x_train = pipeline_without_model.fit_transform(train[['image']])
    model_without_pipeline.fit(x_train, y_train)

    x_test = pipeline_without_model.transform(test[['image']])
    print(model_without_pipeline.score(x_test, y_test))

    print("Shapley values")
    shapley_values = _data_valuation._compute_shapley_values(x_train,
                                                             np.squeeze(y_train),
                                                             x_test,
                                                             np.squeeze(y_test))
    df_with_id_and_shapley_value = pd.DataFrame({"lineage_image_id": image_lineage_ids, "shapley_value": shapley_values})
    rows_to_fix = df_with_id_and_shapley_value.nsmallest(50, "shapley_value")
    joined_rows_to_fix = train.merge(rows_to_fix, left_index=True, right_on="lineage_image_id")
    for row_index, row in list(joined_rows_to_fix.iterrows())[0:3]:
        from matplotlib import pyplot as plt
        print(f"""row["category_name"]: {row["category_name"]}""")
        plt.imshow(np.reshape(row["image"], (28, 28, 1)), interpolation='nearest')
        plt.show()

    print(rows_to_fix)
    label_corrections = {}  # TODO

execute_image_pipeline_w_shapley()
