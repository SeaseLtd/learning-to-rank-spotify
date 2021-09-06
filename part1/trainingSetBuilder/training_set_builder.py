import pandas as pd
from part1.dataPreprocessing import utils


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
       interactions_data_frame, query_id_sample_threshold, percentage, test_set_size)
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


def training_test_set_split(interactions_data_frame, query_id_sample_threshold, percentage, test_set_size):

    under_sampled_only_train = interactions_data_frame[interactions_data_frame.queryIdCount <
                                                       query_id_sample_threshold/percentage]
    to_steal_from = interactions_data_frame[interactions_data_frame.queryIdCount >=
                                            query_id_sample_threshold/percentage]
    # Separate in training and test set
    to_steal_from_group = to_steal_from.groupby(['query_ID', 'Ranking'])
    # Take 20% of the observations for each relevance labels in each query and put it in the test set
    flags = (to_steal_from_group.cumcount() + 1) <= to_steal_from_group['Ranking'].transform(
        'size') * percentage
    to_steal_from = to_steal_from.assign(to_steal=flags)
    interactions_test = to_steal_from[to_steal_from.to_steal == True]
    interactions_train = to_steal_from[to_steal_from.to_steal == False]
    # If we don't achieve test size we add an entire query id group from the only train
    if len(interactions_test) < test_set_size and len(
            under_sampled_only_train[under_sampled_only_train.queryIdCount >= query_id_sample_threshold]) > 0:
        to_steal_from = under_sampled_only_train[under_sampled_only_train.queryIdCount >= query_id_sample_threshold]
        to_steal_from_group = to_steal_from.groupby('query_ID')
        to_steal_from_group_ordered = to_steal_from_group.size().sort_values(ascending=False)
        i = 0
        while len(interactions_test) < test_set_size and i < len(to_steal_from_group_ordered):
            to_move = to_steal_from[to_steal_from.query_ID == to_steal_from_group_ordered.index.values[i]]
            interactions_test = interactions_test.append(to_move, sort=False)
            under_sampled_only_train = pd.concat([under_sampled_only_train, to_move], sort=False)
            under_sampled_only_train = under_sampled_only_train.reset_index().drop_duplicates(
                under_sampled_only_train.columns, keep=False).set_index('index')
            i = i + 1
    interactions_train = pd.concat([interactions_train, under_sampled_only_train], sort=False)
    interactions_train = interactions_train.drop(columns=['queryIdCount', 'to_steal'])
    interactions_test = interactions_test.drop(columns=['queryIdCount', 'to_steal'])
    return interactions_train, interactions_test


def mapping(relevance_label_number):
    if relevance_label_number == "10":
        # Group the position values to relevance labels from 0 to 10
        bins = {range(0, 2): 10, range(2, 3): 9, range(3, 4): 8, range(4, 6): 7,
             range(6, 11): 6, range(11, 21): 5, range(21, 36): 4, range(36, 56): 3,
             range(56, 81): 2, range(81, 131): 1, range(131, 201): 0}
        return bins
    elif relevance_label_number == "20":
        # Group the position values to relevance labels from 0 to 20
        bins = {range(1, 2): 20, range(2, 3): 19, range(3, 4): 18, range(4, 5): 17, range(5, 6): 16, range(6, 7): 15,
             range(7, 8): 14, range(8, 9): 13, range(9, 10): 12,
             range(10, 11): 11, range(11, 14): 10, range(14, 19): 9, range(19, 26): 8, range(26, 36): 7,
             range(36, 46): 6,
             range(46, 61): 5, range(61, 76): 4, range(76, 96): 3,
             range(96, 116): 2, range(116, 151): 1, range(151, 201): 0}
        return bins
    else:
        print("mapping not found")


def training_set_builder(output_dir, dataset_name, relevance_label_number, test_set_size, query_id_sample_threshold):
    # Load the dataset obtained after the preprocessing part
    new_dataset_store = pd.HDFStore(output_dir+'/'+dataset_name+'.h5', 'r')
    new_dataset = new_dataset_store[dataset_name]
    new_dataset_store.close()

    # Group the position values to generate 'bins' to use as Relevance Labels
    bins = mapping(relevance_label_number)
    # Generate Relevance Labels from 'Position' values and store them in the 'Ranking' column
    new_dataset['Ranking'] = new_dataset['Position'].apply(lambda x: next((v for k, v in bins.items() if x in k), 0))

    # create the training set and the tests set to train the model
    write_training_test_set(new_dataset, output_dir, test_set_size, query_id_sample_threshold)

