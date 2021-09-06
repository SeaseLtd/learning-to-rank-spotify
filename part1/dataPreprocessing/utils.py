import pandas as pd
import string
import re
import category_encoders as ce
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def clean_data(text):
    text = re.sub('[' + string.punctuation + ']', '', str(text).lower())
    text = re.sub(r'\n', r' ', text)
    return text

def write_set(dataset, output_dir, set_name):
    # Binary save
    store = pd.HDFStore(output_dir+'/'+set_name+'.h5')
    store[set_name] = dataset
    store.close()

def leave_one_out_encoding(dataset, variable_to_encode, target_variable):
    # Leave One Out encoding - scikit-learn - category_encoders
    encoder = ce.LeaveOneOutEncoder(cols=[variable_to_encode]).fit(dataset, dataset[target_variable])
    new_dataset = encoder.transform(dataset)
    return new_dataset

def hash_encoding(dataset, variable_to_encode):
    # Hash encoding - scikit-learn - category_encoders
    encoder = ce.HashingEncoder(cols=[variable_to_encode], n_components=8)
    new_dataset = encoder.fit_transform(dataset)
    return new_dataset

def doc2vec_encoding(dataset, variable_to_encode):
    # Doc2Vec encoding - gensim
    dataset[variable_to_encode] = dataset[variable_to_encode].map(lambda x: clean_data(x))
    lines = dataset[variable_to_encode]
    token = []
    for line in lines:
        line = line.split()
        token.append(line)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(token)]
    model = Doc2Vec(documents, min_count=1)
    print(model.wv.vocab)
    vec = model.docvecs.doctag_syn0
    model.save('d2v_model')

    lines_vector = pd.DataFrame(vec)
    new_dataset = pd.concat([dataset, lines_vector.set_index(dataset.index)], axis=1)
    new_dataset.drop(columns=[variable_to_encode], inplace=True)
    return new_dataset


def fillnan_single_column_using_dictionary(dataset, key, value, newvalue):
    key_value_dataframe = dataset[[key, value]].drop_duplicates().dropna()
    key_column = key_value_dataframe[key]
    value_column = key_value_dataframe[value]
    dictionary = dict(zip(key_column, value_column))
    # Fill NaN in a column using the dictionary
    key_value_dataframe_filled = dataset[[key, value]]
    key_value_dataframe_filled.set_index([key], drop=False, inplace=True)
    key_value_dataframe_filled[key].update(pd.Series(dictionary))
    key_value_dataframe_filled = key_value_dataframe_filled.rename(columns={key: newvalue})
    print('\nCount total NaN in ' + value + ':\n',
          key_value_dataframe_filled.isnull().sum())
    newvalue_column = key_value_dataframe_filled[newvalue]
    return newvalue_column




