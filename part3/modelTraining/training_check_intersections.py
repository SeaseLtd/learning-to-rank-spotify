import pandas as pd
import xgboost

def check_intersections(training_set, test_set, column_to_exclude):
    cols = training_set.columns.tolist()
    cols.remove(column_to_exclude)
    intersections = pd.merge(training_set, test_set, on=cols)
    intersections.drop(columns=[column_to_exclude + "_y"], inplace=True)
    intersections = intersections.rename(columns={column_to_exclude + "_x" : column_to_exclude})
    drop_intersections_from_training_set = pd.concat([training_set, intersections], ignore_index=True)
    drop_intersections_from_training_set.drop_duplicates(keep=False, inplace=True)
    check_no_intersections = pd.merge(drop_intersections_from_training_set, test_set, on=cols)
    if check_no_intersections.empty:
        print('No more intersections')
        return drop_intersections_from_training_set
    print('There are still intersections')


def train_model(output_dir, training_set, test_set, model_name, eval_metric):
    # Load the training and the test sets
    training_set_store = pd.HDFStore(training_set, 'r')
    training_set_data_frame = training_set_store['training_set']
    training_set_store.close()

    test_set_store = pd.HDFStore(test_set, 'r')
    test_set_data_frame = test_set_store['test_set']
    test_set_store.close()

    training_set_data_frame = check_intersections(training_set_data_frame, test_set_data_frame, column_to_exclude="query_ID")

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
    params = {'objective': 'rank:ndcg', 'eval_metric': eval_metric, 'verbosity': 2, 'early_stopping_rounds': 10}
    watch_list = [(test_xgb_matrix, 'eval'), (training_xgb_matrix, 'train')]

    xgb_model = xgboost.train(params, training_xgb_matrix, num_boost_round=999, evals=watch_list)
    predictions = xgb_model.predict(test_xgb_matrix)
    print(predictions)

    # Saving  XGBoost model
    xgboost_model_json = output_dir + "/xgboost-" + model_name + ".json"
    xgb_model.dump_model(xgboost_model_json, fmap='', with_stats=True, dump_format='json')


