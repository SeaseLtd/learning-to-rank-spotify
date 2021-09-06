## A Learning to Rank Project on a Daily Song Ranking Problem - part 3

This folder contains the code related to the third part of the project described in [this blog post](https://sease.io/2021/03/a-learning-to-rank-project-on-a-daily-song-ranking-problem-part-3.html)

Check it out for more details!


### dataPrepocessing
It contains the files to load the Spotify CSV file and apply data preprocessing techniques.
It includes a method to create the query id hashing (i.e. from multiple columns).

##### Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-d | Path of the dataset | Contains path of the Spotify dataset to load

-e | Encoding | Contains the categorical encoding to apply to the 'Track Name' feature (hash or d2v)


### trainingSetBuilder
It contains the files for the creation of the training and test set applying a new split (random)

##### Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-d | Dataset name | Contains the name of the dataset obtained after the data preprocessing part

-r | Relevance labels | The mapping to apply to group the position values in relevance labels (e.g. 10 or 20)

-s | Test Set Size | Contains the size of the test set (around 20% of the entire dataset)

-t | Query id sample threshold | Contains the threshold to set to avoid small query id samples

### modelTraining 
It contains the files for the training of the model using the LambdaMART method (XGBoost).
It includes a method to check the intersections (same rows) between training and test set when comparing different models on the same test set.

##### Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-t | Training Set File | Path to the training set file

-s | Test Set File | Path to the test set file

-n | Model Name | Name of the model we are creating

-e | Evaluation Metric | The metric to use for the model generation (e.g. ndcg@10)

