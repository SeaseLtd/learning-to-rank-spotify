import pandas as pd
from part1.dataPreprocessing import utils

def training_set_builder_subset(output_dir):
    # Create a Subset from the full dataset to avoid intersections

    # Load the full training and test sets
    training_set_store = pd.HDFStore(output_dir+'/training_set.h5', 'r')
    training_set_data_frame = training_set_store['training_set']
    training_set_store.close()

    test_set_store = pd.HDFStore(output_dir+'/test_set.h5', 'r')
    test_set_data_frame = test_set_store['test_set']
    test_set_store.close()

    # Extract a subset containing Daily Song Charts from Position 1 to 21
    training_set_subset = training_set_data_frame[training_set_data_frame['Position'] <= 21]
    test_set_subset = test_set_data_frame[test_set_data_frame['Position'] <= 21]

    # Delete Ranking column and create it again (from Position values) with the right mapping
    training_set_subset.drop(columns=['Ranking'], inplace=True)
    test_set_subset.drop(columns=['Ranking'], inplace=True)

    # Group the Position values to Relevance labels from 0 to 20
    d = {range(1, 2): 20, range(2, 3): 19, range(3, 4): 18, range(4, 5): 17, range(5, 6): 16, range(6, 7): 15,
         range(7, 8): 14, range(8, 9): 13, range(9, 10): 12,
         range(10, 11): 11, range(11, 12): 10, range(12, 13): 9, range(13, 14): 8, range(14, 15): 7, range(15, 16): 6,
         range(16, 17): 5, range(17, 18): 4, range(18, 19): 3,
         range(19, 20): 2, range(20, 21): 1, range(21, 22): 0}

    training_set_subset['Ranking'] = training_set_subset['Position'].apply(
        lambda x: next((v for k, v in d.items() if x in k), 0))

    test_set_subset['Ranking'] = test_set_subset['Position'].apply(
        lambda x: next((v for k, v in d.items() if x in k), 0))


    training_set_subset_query_counts = training_set_subset["query_ID"].value_counts()
    test_set_subset_query_counts = test_set_subset["query_ID"].value_counts()

    training_set_subset_relevance_counts = training_set_subset.groupby("query_ID")["Ranking"].value_counts()
    test_set_subset_relevance_counts = test_set_subset.groupby("query_ID")["Ranking"].value_counts()

    # Save the Subset sets
    utils.write_set(training_set_subset, output_dir + '/subset', 'training_set')
    utils.write_set(test_set_subset, output_dir + '/subset', 'test_set')