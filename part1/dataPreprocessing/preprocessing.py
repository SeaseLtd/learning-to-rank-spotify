import pandas as pd
from part1.dataPreprocessing import utils

def preprocessing(output_dir, dataset_path, encoding):
    # Load the csv file (Spotify dataset)
    dataset = pd.read_csv(dataset_path)

    # Rename column from Track Name to Title (no space)
    dataset = dataset.rename(columns={"Track Name": "Title"})

    # Remove the first part of the URL
    dataset['URL'] = dataset['URL'].str.replace('https://open.spotify.com/track/', '')
    # Factorize the URL
    dataset['ID'] = pd.factorize(dataset['URL'])[0]
    dataset.drop(columns=['URL'], inplace=True)

    # Check NaN values
    print('\nCount total NaN in each column:\n',
          dataset.isnull().sum())
    # Create a dictionary ID-Track Name to fill NaN values in the Title (Track Name) column
    title_NaN_filled = utils.fillnan_single_column_using_dictionary(dataset, "ID", "Title", "Title_NaN_filled")
    # Create a dictionary ID-Artist to fill NaN values in the Artist column
    artist_NaN_filled = utils.fillnan_single_column_using_dictionary(dataset, "ID", "Artist", "Artist_NaN_filled")
    features_NaN_filled = pd.concat([title_NaN_filled, artist_NaN_filled], axis=1)
    dataset.drop(columns=['Title'], inplace=True)
    dataset.drop(columns=['Artist'], inplace=True)

    # new_dataset is the dataset without NaN values
    new_dataset = dataset.join(features_NaN_filled.set_index(dataset.index))
    print("\nCount total NaN in each column: \n",
          new_dataset.isnull().sum())

    # Save the new dataset (without NaN values) to csv
    new_dataset.to_csv(output_dir+'/spotify_NaN_filled.csv', index=False)

    # Create the query Id from the Region column
    new_dataset['query_ID'] = pd.factorize(new_dataset['Region'])[0]
    new_dataset.drop(columns=['Region'], inplace=True)

    # Split the 'Date' column in day, month and year and extract day of the week feature
    split_date = new_dataset['Date'].str.split(pat='-', expand=True).astype(int)
    new_dataset['Date'] = pd.to_datetime(new_dataset['Date'], format='%Y-%m-%d')
    new_dataset['Weekday'] = new_dataset['Date'].dt.dayofweek
    new_dataset['Year'] = split_date[0]
    new_dataset['Month'] = split_date[1]
    new_dataset['Day'] = split_date[2]
    new_dataset.drop(columns=['Date'], inplace=True)
    new_dataset.drop(columns=['Year'], inplace=True)

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





