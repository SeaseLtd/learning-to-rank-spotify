import pandas as pd
from part1.dataPreprocessing import utils

def generate_id_from_columns(input_data_frame, query_features, id_column):
    str_id = input_data_frame[query_features[0]].astype(str)
    for feature in query_features[1:]:
        str_id = str_id + '_' + input_data_frame[feature].astype(str)
    input_data_frame[id_column] = pd.factorize(str_id)[0]


def preprocessing_query_id_hashing(output_dir, dataset_path, encoding):
    # Load the dataset with NaN filled (spotify_NaN_filled.csv)
    new_dataset = pd.read_csv(dataset_path)

    # Start Features Engineering
    # Split the 'Date' feature in day, month and year and extract day of the week
    split_date = new_dataset['Date'].str.split(pat='-', expand=True).astype(int)
    new_dataset['Date'] = pd.to_datetime(new_dataset['Date'], format='%Y-%m-%d')
    new_dataset['Weekday'] = new_dataset['Date'].dt.dayofweek
    new_dataset['Year'] = split_date[0]
    new_dataset['Month'] = split_date[1]
    new_dataset['Day'] = split_date[2]
    new_dataset.drop(columns=['Date'], inplace=True)
    new_dataset.drop(columns=['Year'], inplace=True)

    # Generate the query Id as an hash of multiple query-level features
    query_level_features = ['Region', 'Day', 'Month', 'Weekday']
    generate_id_from_columns(new_dataset, query_level_features, id_column='query_ID')
    unique_query_ID = new_dataset['query_ID'].unique()
    query_ID_value_counts = new_dataset['query_ID'].value_counts()
    new_dataset['Region'] = pd.factorize(new_dataset['Region'])[0]

    # Leave one out encoding for 'Artist' feature
    new_dataset = utils.leave_one_out_encoding(new_dataset, "Artist_NaN_filled", "Position")

    # Choose encoding for 'Title' feature
    if encoding == "hash":
        # Hash encoding
        new_dataset_encoded = utils.hash_encoding(new_dataset, "Title_NaN_filled")
        utils.write_set(new_dataset_encoded, output_dir, 'new_dataset_title_encoded_with_' + encoding)
    elif encoding == "d2v":
        # Doc2vec encoding
        new_dataset_encoded = utils.doc2vec_encoding(new_dataset, "Title_NaN_filled")
        utils.write_set(new_dataset_encoded, output_dir, 'new_dataset_title_encoded_with_' + encoding)
    else:
        print("No encoding found")

