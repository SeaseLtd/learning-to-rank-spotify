import pandas as pd
from part1.dataPreprocessing import utils
from part1.trainingSetBuilder import training_set_builder as tsb

def write_training_test_set(data_frame, output_dir, test_set_size=10000, query_id_sample_threshold=25):
    query_ids_sample_counts = pd.DataFrame(data_frame['query_ID'].value_counts())
    query_ids_sample_counts.index.rename('query_ID', inplace=True)
    query_ids_sample_counts.rename(columns={'query_ID': "queryIdCount"}, inplace=True)
    interactions_data_frame = pd.merge(data_frame, query_ids_sample_counts,
                                       on='query_ID', how='outer', validate="many_to_one")

    interactions_data_frame = clean_data_frame_from_single_label(interactions_data_frame)

    percentage = test_set_size / len(interactions_data_frame)
    percentage = min(percentage, 0.2)
    test_set_size = min(test_set_size, int(len(interactions_data_frame) * percentage))


    # Split in training and test set
    interactions_train, interactions_test = training_test_set_split(
       interactions_data_frame, query_id_sample_threshold, percentage)
    interactions_train = interactions_train.sort_values('query_ID')
    interactions_test = interactions_test.sort_values('query_ID')
    utils.write_set(interactions_train, output_dir, 'training_set')
    utils.write_set(interactions_test, output_dir, 'test_set')


def clean_data_frame_from_single_label(data_frame):
    # Add count of num different relevance label per query id
    relevance_counts = data_frame.groupby("query_ID")["Ranking"].value_counts()
    different_relevance_counts_per_query_id = pd.DataFrame(relevance_counts.index.get_level_values(0).value_counts())
    different_relevance_counts_per_query_id.index.rename('query_ID', inplace=True)
    different_relevance_counts_per_query_id.rename(columns={"query_ID": "relevance_count"}, inplace=True)
    interactions_data_frame = pd.merge(data_frame, different_relevance_counts_per_query_id,
                                       on='query_ID', how='outer', validate="many_to_one")

    # Separate query id with just one relevance label (we don't want it in the test set)
    interactions_data_frame = interactions_data_frame[interactions_data_frame.relevance_count > 1]
    interactions_data_frame.drop(columns=['relevance_count'], inplace=True)
    interactions_data_frame = interactions_data_frame.reset_index(drop=True)
    return interactions_data_frame


def training_test_set_split(interactions_data_frame, query_id_sample_threshold, percentage):

    under_sampled_only_train = interactions_data_frame[interactions_data_frame.queryIdCount <
                                                       query_id_sample_threshold/percentage]


    to_steal_from = interactions_data_frame[interactions_data_frame.queryIdCount >=
                                            query_id_sample_threshold/percentage]


    # Randomly shuffled the data frame and then split it into training and test sets, ensuring that 20% of the observations for each query Id move to the Test Set, independently of the Relevance Labels
    to_steal_from_mixed = to_steal_from.sample(frac=1).reset_index(drop=True)
    to_steal_from_group = to_steal_from_mixed.groupby(['query_ID'])
    flags = (to_steal_from_group.cumcount() + 1) <= to_steal_from_group['query_ID'].transform(
        'size') * percentage
    to_steal_from_mixed = to_steal_from_mixed.assign(to_steal=flags)
    interactions_test = to_steal_from_mixed[to_steal_from_mixed.to_steal == True]
    interactions_train = to_steal_from_mixed[to_steal_from_mixed.to_steal == False]

    interactions_train = pd.concat([interactions_train, under_sampled_only_train], sort=False)
    interactions_train = interactions_train.drop(columns=['queryIdCount', 'to_steal'])
    interactions_test = interactions_test.drop(columns=['queryIdCount', 'to_steal'])
    return interactions_train, interactions_test


def training_set_builder(output_dir, dataset_name, relevance_label_number, test_set_size, query_id_sample_threshold):
    # Load the dataset obtained after the preprocessing part
    new_dataset_store = pd.HDFStore(output_dir+'/'+dataset_name+'.h5', 'r')
    new_dataset = new_dataset_store[dataset_name]
    new_dataset_store.close()

    # Group the position values to generate 'bins' to use as Relevance Labels
    bins = tsb.mapping(relevance_label_number)
    # Generate Relevance Labels from 'Position' values and store them in the 'Ranking' column
    new_dataset['Ranking'] = new_dataset['Position'].apply(lambda x: next((v for k, v in bins.items() if x in k), 0))

    # create the training set and the tests set to train the model
    write_training_test_set(new_dataset, output_dir, test_set_size, query_id_sample_threshold)

