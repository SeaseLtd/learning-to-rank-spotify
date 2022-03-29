import pandas as pd
import xgboost

def assigning_new_query_id_to_undersampled_queries_single_feature(dataframe, single_query_feature):
    unique_query = dataframe['query_ID'].unique().tolist()
    unique_feature = dataframe[single_query_feature].unique().tolist()
    column_names = dataframe.columns
    new_dataframe = pd.DataFrame(columns=column_names, dtype=int)

    i = unique_query[0]
    for value in unique_feature:
        feature_dataframe = dataframe[dataframe[single_query_feature] == value]
        feature_dataframe['query_ID'] = i
        new_dataframe =  pd.concat([new_dataframe, feature_dataframe], ignore_index=True, sort=False)
        i = i + 1

    query_ids_sample_counts = pd.DataFrame(new_dataframe['query_ID'].value_counts())
    query_ids_sample_counts.index.rename('query_ID', inplace=True)
    query_ids_sample_counts.rename(columns={'query_ID': "queryIdCount"}, inplace=True)
    under_sampled_data_frame = pd.merge(new_dataframe, query_ids_sample_counts,
                                       on='query_ID', how='outer', validate="many_to_one")
    return under_sampled_data_frame


def assigning_new_query_id_to_undersampled_queries_multiple_feature(input_data_frame, multiple_query_feature, id_column):
    unique_query = input_data_frame['query_ID'].unique().tolist()

    input_data_frame.drop(columns=['query_ID'], inplace=True)

    str_id = input_data_frame[multiple_query_feature[0]].astype(str)
    for feature in multiple_query_feature[1:]:
        str_id = str_id + '_' + input_data_frame[feature].astype(str)
    input_data_frame[id_column] = str_id
    input_data_frame[id_column] = pd.factorize(str_id)[0]
    i = unique_query[0]

    column_names = input_data_frame.columns
    new_dataframe = pd.DataFrame(columns=column_names, dtype=int)
    unique_value = input_data_frame[id_column].unique().tolist()
    for x in unique_value:
        feature_dataframe = input_data_frame[input_data_frame[id_column] == x]
        feature_dataframe[id_column] = i
        new_dataframe = pd.concat([new_dataframe, feature_dataframe], ignore_index=True, sort=False)
        i = i + 1

    query_ids_sample_counts = pd.DataFrame(new_dataframe['query_ID'].value_counts())
    query_ids_sample_counts.index.rename('query_ID', inplace=True)
    query_ids_sample_counts.rename(columns={'query_ID': "queryIdCount"}, inplace=True)
    under_sampled_data_frame = pd.merge(new_dataframe, query_ids_sample_counts,
                                        on='query_ID', how='outer', validate="many_to_one")

    return under_sampled_data_frame


def train_model(output_dir, training_set, test_set, under_sampled_only_train, model_name, eval_metric, query_level_feature):
    # Load the training and the test sets
    training_set_store = pd.HDFStore(training_set, 'r')
    training_set_data_frame = training_set_store['training_set']
    training_set_store.close()

    test_set_store = pd.HDFStore(test_set, 'r')
    test_set_data_frame = test_set_store['test_set']
    test_set_store.close()

    under_sampled_only_train_store = pd.HDFStore(under_sampled_only_train, 'r')
    under_sampled_data_frame = under_sampled_only_train_store['under_sampled_only_train']
    under_sampled_only_train_store.close()

    # Drop under_sampled_only_train from training_set
    under_sampled_data_frame = under_sampled_data_frame.drop(columns='queryIdCount')
    training_set_data_frame = training_set_data_frame.append(under_sampled_data_frame).drop_duplicates(keep=False)

    query_feature_grouping = query_level_feature.split(',')
    if len(query_feature_grouping) > 1:
        # Group under_sampled_only_train by Region and Month
        under_sampled_data_frame = assigning_new_query_id_to_undersampled_queries_multiple_feature(
            under_sampled_data_frame, query_feature_grouping, id_column='query_ID')
        #unique_query_ID = under_sampled_data_frame['query_ID'].unique()
        #query_id_count = under_sampled_data_frame['query_ID'].value_counts()
        under_sampled_data_frame.drop(columns='queryIdCount', inplace=True)
        training_set_data_frame = training_set_data_frame.append(under_sampled_data_frame)
    else:
        # Group under_sampled_only_train by Region only
        under_sampled_data_frame = assigning_new_query_id_to_undersampled_queries_single_feature(
            under_sampled_data_frame, ''.join(query_feature_grouping))
        #unique_query_ID = under_sampled_data_frame['query_ID'].unique()
        #query_id_count = under_sampled_data_frame['query_ID'].value_counts()
        under_sampled_data_frame.drop(columns='queryIdCount', inplace=True)
        training_set_data_frame = training_set_data_frame.append(under_sampled_data_frame)


    training_set_data_frame.drop(columns=['Position'], inplace=True)
    test_set_data_frame.drop(columns=['Position'], inplace=True)

    training_data_set = training_set_data_frame[
        training_set_data_frame.columns.difference(
            ['Ranking', 'ID', 'query_ID'])]
    training_query_id_column = training_set_data_frame['query_ID']
    training_query_groups = training_query_id_column.value_counts(sort=False)
    training_label_column = training_set_data_frame['Ranking']

    test_data_set = test_set_data_frame[
        test_set_data_frame.columns.difference(
            ['Ranking', 'ID', 'query_ID'])]
    test_query_id_column = test_set_data_frame['query_ID']
    test_query_groups = test_query_id_column.value_counts(sort=False)
    test_label_column = test_set_data_frame['Ranking']

    training_xgb_matrix = xgboost.DMatrix(training_data_set, label=training_label_column)
    training_xgb_matrix.set_group(training_query_groups)
    test_xgb_matrix = xgboost.DMatrix(test_data_set, label=test_label_column)
    test_xgb_matrix.set_group(test_query_groups)
    params = {'objective': 'rank:ndcg', 'eval_metric': eval_metric, 'verbosity': 2}
    watch_list = [(training_xgb_matrix, 'train'), (test_xgb_matrix, 'eval')]

    xgb_model = xgboost.train(params, training_xgb_matrix, num_boost_round=999, evals=watch_list)
    predictions = xgb_model.predict(test_xgb_matrix)
    print(predictions)

    # Saving  XGBoost model
    xgboost_model_json = output_dir + "/xgboost-" + model_name + ".json"
    xgb_model.dump_model(xgboost_model_json, fmap='', with_stats=True, dump_format='json')


