## A Learning to Rank Project on a Daily Song Ranking Problem - part 1

This folder contains the code related to the first part of the project described in this blog post:

https://sease.io/2020/12/a-learning-to-rank-project-on-a-daily-song-ranking-problem.html

Check it out for more details!

###dataPrepocessing
It contains the files to load the Spotify file and apply several data preprocessing techniques.

#####Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-d | Path of the dataset | Contains path of the Spotify dataset to load

-e | Encoding | Contains the categorical encoding to apply to the 'Track Name' feature (hash or d2v)

e.g. 

python3 -m part1.dataPreprocessing.main_preprocessing -o /Users/spotify_project/outputs -d /Users/spotify_project/spotify_dataset.csv -e hash

###trainingSetBuilder
It contains the files for the creation of the training and test set.

#####Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-d | Dataset name | Contains the name of the dataset obtained after the data preprocessing part

-r | Relevance labels | The mapping to apply to group the position values in relevance labels (e.g. 10 or 20)

-s | Test Set Size | Contains the size of the test set (around 20% of the entire dataset)

-t | Query id sample threshold | Contains the threshold set to avoid small query id samples


###modelTraining 
It contains the files for the training of the model using the LambdaMART method (XGBoost).

#####Explanation of parameters

-o | Output Directory | Contains the directory where to save the output files

-t | Training Set File | Path to the training set file

-s | Test Set File | Path to the test set file

-n | Model Name | Name of the model we are creating

-e | Evaluation Metric | The metric to use for the model generation (e.g. ndcg@10)
