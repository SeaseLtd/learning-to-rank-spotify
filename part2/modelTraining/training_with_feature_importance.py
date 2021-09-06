import pandas as pd
import shap
import xgboost
import matplotlib.pyplot as plt


def train_model(output_dir, training_set, test_set, model_name, eval_metric, images_path, feature_to_analyze):
    # Train a LTR model with XGBoost
    training_set_store = pd.HDFStore(training_set, 'r')
    training_set_data_frame = training_set_store['training_set']
    training_set_store.close()

    test_set_store = pd.HDFStore(test_set, 'r')
    test_set_data_frame = test_set_store['test_set']
    test_set_store.close()

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

    # Compute the feature importance
    feature_importance(xgb_model, training_data_set, images_path, feature_to_analyze)

def feature_importance(xgb_model, training_data_set, images_path, feature_to_analyze):
    # Explain the model prediction using SHAP library
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(training_data_set)

    # SHAP plots
    # SUMMARY PLOT
    shap.summary_plot(shap_values, training_data_set, show=False)
    plt.savefig(images_path + '/summary_plot.png', bbox_inches='tight')
    plt.close()

    # SUMMARY PLOT with bars
    shap.summary_plot(shap_values, training_data_set, plot_type="bar", show=False)
    plt.savefig(images_path + '/summary_plot_bars.png', bbox_inches='tight')
    plt.close()

    # DECISION PLOT
    # one observation
    shap.decision_plot(explainer.expected_value, shap_values[0], training_data_set.iloc[0],
                       feature_names=training_data_set.columns.tolist(), show=False, ignore_warnings=True)
    plt.savefig(images_path + '/decision_plot_0.png', bbox_inches='tight')
    plt.close()

    # 500000 observations only
    shap.decision_plot(explainer.expected_value, shap_values[0:500000, :], training_data_set.iloc[0:500000, :],
                       feature_names=training_data_set.columns.tolist(), show=False, ignore_warnings=True)
    plt.savefig(images_path + '/decision_plot_500000.png', bbox_inches='tight')
    plt.close()

    # total
    shap.decision_plot(explainer.expected_value, shap_values, training_data_set,
                       feature_names=training_data_set.columns.tolist(), show=False, ignore_warnings=True)
    plt.savefig(images_path + '/decision_plot.png', bbox_inches='tight')
    plt.close()


    # FORCE PLOT
    # one observation
    shap.force_plot(explainer.expected_value, shap_values[0], training_data_set.iloc[0], show=False,
                    matplotlib=True)
    plt.savefig(images_path + '/force_plot_0.png', bbox_inches='tight')
    plt.close()

    # total
    html_img = shap.force_plot(explainer.expected_value, shap_values, training_data_set, show=False)
    shap.save_html(images_path + '/full_prediction_explanation.html', html_img)


    # Create a DEPENDENCE PLOT to show the effect of a single feature
    # Specify only one feature (in feature_to_analyze)
    shap.dependence_plot(feature_to_analyze, shap_values, training_data_set, show=False)
    plt.savefig(images_path + '/dependence_plot_' + feature_to_analyze + '.png', bbox_inches='tight')
    plt.close()

    # Specify both features to analyze (e.g. feature_to_analyze and Month)
    if 'Month' in training_data_set.columns:
        shap.dependence_plot(feature_to_analyze, shap_values, training_data_set,
                             interaction_index='Month', show=False)
        plt.savefig(images_path + '/dependence_plot_' + feature_to_analyze + '_with_Month.png', bbox_inches='tight')
        plt.close()

    # Specify both features to analyze (e.g. feature_to_analyze and Streams)
    if 'Streams' in training_data_set.columns:
        shap.dependence_plot(feature_to_analyze, shap_values, training_data_set,
                             interaction_index='Streams', show=False)
        plt.savefig(images_path + '/dependence_plot_' + feature_to_analyze + '_with_Streams.png', bbox_inches='tight')
        plt.close()













