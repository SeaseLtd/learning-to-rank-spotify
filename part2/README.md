## A Learning to Rank Project on a Daily Song Ranking Problem - part 2

This folder contains the code related to the second part of the project described in [this blog post](https://sease.io/2021/02/a-learning-to-rank-project-on-a-daily-song-ranking-problem-part-2.html)

Check it out for more details!


### streamsOrdering
It contains the files to sort the full dataset in descending order based on the ‘Streams‘ feature and manually calculate the NDCG@10 on the dataframe.

##### Explanation of parameters

-d | Dataset path | Contains the directory where the file after data preprocessing is saved

-n | Dataset name | Contains the name of the dataset obtained after the data preprocessing part

-s | Highest streams File | Contains the directory where to save the dataframe with the highest streams


### trainingSetBuilder
It contains the files for the creation of the Subset from the training and test set.

##### Explanation of parameters

-o | Output Directory | Contains the directory where the training and the test set are saved


### modelTraining 
It contains the files to train the model using the LambdaMART method (XGBoost) and to perform the feature importance using SHAP.

##### Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-t | Training Set File | Path to the training set file

-s | Test Set File | Path to the test set file

-n | Model Name | Name of the model we are creating

-e | Evaluation Metric | The metric to use for the model generation (e.g. ndcg@10)

-i | Images Directory | Contains the directory where to save the SHAP images/plots

-f | Feature Name | The feature to analyze with the dependence plot

