## A Learning to Rank Project on a Daily Song Ranking Problem - part 4

This folder contains the code related to the fourth part of the project described in [this blog post](https://sease.io/2021/05/a-learning-to-rank-project-on-a-daily-song-ranking-problem-part-4.html)

Check it out for more details!


### trainingSetBuilder
It contains the files for the creation of the training and test set applying a new split (random).
It also contains a method to "remove" (fill with nan values) the features used to create the query id hashing (query level features).


##### Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-d | Dataset name | Contains the name of the dataset obtained after the data preprocessing part

-r | Relevance labels | The mapping to apply to group the position values in relevance labels (e.g. 10 or 20)

-s | Test Set Size | Contains the size of the test set (around 20% of the entire dataset)

-t | Query id sample threshold | Contains the threshold set to avoid small query id samples

-q | Remove query level features | Pass this parameter, if you want to remove them otherwise exclude it 


### modelTraining 
It contains the files for the training of the model using the LambdaMART method (XGBoost); in this case 2 different approaches can be applied:
- drop all the under-sampled queries (*training_drop_undersampled_queries.py*)
- group the under-sampled queries based on one OR two features (*training_group_undersampled_queries.py*)

##### Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-t | Training Set File | Path to the training set file

-s | Test Set File | Path to the test set file

-u | Under-sampled File | Path to the dataset that contains the under-sampled queries only

-n | Model Name | Name of the model we are creating

-e | Evaluation Metric | The metric to use for the model generation (e.g. ndcg@10)

-f | Query level features | The name of the feature (features) to use to group the observations (e.g. Region OR Region,Month)



