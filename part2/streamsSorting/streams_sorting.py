import pandas as pd
import numpy as np
import statistics

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def ndcg_dataframe(data_frame, queries, relevance_label):
    # Calculate the NDCG at k on a dataframe
    query_groups = data_frame.query_ID.unique()
    ndcg_scores_list = []
    for n in query_groups:
        grouped = data_frame[data_frame[queries] == n]
        #grouped_index = np.arange(1, len(grouped) + 1)
        ndcg = ndcg_at_k(grouped[relevance_label], 10)
        ndcg_scores_list.append(ndcg)
    final_ndcg = statistics.mean(ndcg_scores_list)
    print("The final ndcg is: " + str(final_ndcg))


def keep_only_maximum_streams(data_frame, queries, ids):
    # Create a dataframe only with the maximum number of streams for each song for each query
    query_groups = data_frame[queries].unique()
    ids_groups = data_frame[ids].unique()
    column_names = data_frame.columns
    df = pd.DataFrame(columns=column_names)
    for n in query_groups:
        grouped_queries = data_frame[data_frame[queries] == n].reset_index(drop=True)
        for i in ids_groups:
            if (grouped_queries[ids] == i).any():
                grouped_ids = grouped_queries[grouped_queries[ids] == i].iloc[[0]]
                df = df.append(grouped_ids, ignore_index=True)
            else:
                continue
    return df

def streams_sorting(dataset_path, dataset_name, highest_streams_dataset_path):
    # Load the dataset obtained after the data preprocessing part
    new_dataset_store = pd.HDFStore(dataset_path + '/' + dataset_name + '.h5', 'r')
    new_dataset = new_dataset_store[dataset_name]
    new_dataset_store.close()

    # Group the position values to relevance labels from 0 to 20
    d = {range(1, 2): 20, range(2, 3): 19, range(3, 4): 18, range(4, 5): 17, range(5, 6): 16, range(6, 7): 15,
         range(7, 8): 14, range(8, 9): 13, range(9, 10): 12,
         range(10, 11): 11, range(11, 14): 10, range(14, 19): 9, range(19, 26): 8, range(26, 36): 7, range(36, 46): 6,
         range(46, 61): 5, range(61, 76): 4, range(76, 96): 3,
         range(96, 116): 2, range(116, 151): 1, range(151, 201): 0}

    # Generate Relevance Labels from 'Position' values and store them in the 'Ranking' column
    new_dataset['Ranking'] = new_dataset['Position'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))

    # Correlation matrix
    correlation_matrix = new_dataset.corr()

    # New dataframe, sorted by Streams values per each query
    new_dataset_sorted = new_dataset.groupby(['query_ID']).apply(lambda x: x.sort_values(["Streams"], ascending=False)).reset_index(
        drop=True)

    # Check correlation between Position and Streams values counts
    print("Correlation between Position and Streams is: ", new_dataset_sorted['Streams'].corr(new_dataset_sorted['Position']))

    # Check correlation between Ranking and Streams
    print("Correlation between Ranking and Streams is: ", new_dataset_sorted['Streams'].corr(new_dataset_sorted['Ranking']))

    # Calcutate the NDCG
    print("Calculate the ndcg@10 on the data frame sorted (descending) on the 'Streams' feature:")
    ndcg_dataframe(new_dataset_sorted, 'query_ID', 'Ranking')

    # Dataframe with the highest streams, sorted by Streams values per each query
    dataset_highest_streams = keep_only_maximum_streams(new_dataset_sorted, 'query_ID', 'ID')
    dataset_highest_streams.to_csv(highest_streams_dataset_path + '/spotify_highest_streams.csv', index=False)

    dataset_highest_streams_sorted = dataset_highest_streams.groupby(['query_ID']).apply(lambda x: x.sort_values(["Streams"], ascending=False)).reset_index(
        drop=True)

    print("Calculate the ndcg@10 on the data frame that contains the highest streams only:")
    ndcg_dataframe(dataset_highest_streams_sorted, 'query_ID', 'Ranking')


