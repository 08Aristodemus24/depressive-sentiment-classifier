import pickle
import json
import pandas as pd

# for NLP data
def load_corpus(path: str):
    """
    reads a text file and returns the text
    """

    with open(path, 'r', encoding='utf-8') as file:
        corpus = file.read()

    return corpus

def get_chars(corpus: str):
    """
    returns a list of all unique characters found
    in given corpus
    """

    chars = sorted(list(set(corpus)))

    return chars

def load_lookup_array(path: str):
    """
    reads a text file containing a list of all unique values
    and returns this
    """

    with open(path, 'rb') as file:
        char_to_idx = pickle.load(file)
        file.close()

    return char_to_idx

def save_lookup_array(path: str, uniques: list):
    """
    saves and writes all the unique list of values to a
    a file for later loading by load_lookup_array()
    """

    with open(path, 'wb') as file:
        pickle.dump(uniques, file)
        file.close()

def save_meta_data(path: str, meta_data: dict):
    """
    saves dictionary of meta data such as hyper 
    parameters to a .json file
    """

    with open(path, 'w') as file:
        json.dump(meta_data, file)
        file.close()

def load_meta_data(path: str):
    """
    loads the saved dictionary of meta data such as
    hyper parameters from the created .json file
    """

    with open(path, 'r') as file:
        meta_data = json.load(file)
        file.close()

    return meta_data

def save_model(model, path: str):
    """
    saves partcularly an sklearn model in a .pkl file
    for later testing
    """

    with open(path, 'wb') as file:
        pickle.dump(model, file)
        file.close()

def load_model(path: str):
    """
    loads the sklearn model, scaler, or encoder stored
    in a .pkl file for later testing and deployment
    """

    with open(path, 'rb') as file:
        model = pickle.load(file)
        file.close()

    return model

def get_top_models(models_train, models_cross, pool_size: int=10, model_type: str="regressor"):
    """
    takes in the dataframes returned by either LazyClassifier or LazyPredict
    e.g. clf = LazyRegressor(
        verbose=0, 
        ignore_warnings=True, 
        custom_metric=None, 
        regressors=[LinearRegression, Ridge, Lasso, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR])
    models_train, predictions_train = clf.fit(ch_X_trains, ch_X_trains, ch_Y_trains, ch_Y_trains)
    models_cross, predictions_cross = clf.fit(ch_X_trains, ch_X_cross, ch_Y_trains, ch_Y_cross)

    args:
        models_train - 
        models_cross - 
        pool_size - number of rows to take into consideration when merging the
        dataframes of model train and cross validation metric values
    """

    # rename columns for each dataframe to avoid duplication during merge
    for col in models_train.columns:
        models_train.rename(columns={f"{col}": f"Train {col}"}, inplace=True)
        models_cross.rename(columns={f"{col}": f"Cross {col}"}, inplace=True)

    # merge both first pool_size rows of training and cross 
    # validation model dataframes
    models_train = models_train[:pool_size].reset_index()
    models_cross = models_cross[:pool_size].reset_index()
    
    # merge model dataframes on 'Model' column
    top_models = models_train.merge(models_cross, how='inner', left_on='Model', right_on='Model')
    top_models.sort_values(by="Cross Adjusted R-Squared" if model_type == "regressor" else "Cross F1 Score", inplace=True, ascending=False)

    return top_models

def create_metrics_df(train_metric_values, 
                      test_metric_values, 
                      metrics=['accuracy', 'precision', 'recall', 'f1-score']):
    """
    creates a metrics dataframe
    """

    metrics_dict = {
        'data_split': ['training', 'testing']
    }

    for index, metric in enumerate(metrics):
        metrics_dict[metric] = [
            train_metric_values[index], 
            test_metric_values[index]
        ]

    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df

def create_classified_df(train_conf_matrix, val_conf_matrix, test_conf_matrix, train_labels, val_labels, test_labels):
    """
    creates a dataframe that represents all classified and 
    misclassified values
    """

    num_right_cm_train = train_conf_matrix.trace()
    num_right_cm_val = val_conf_matrix.trace()
    num_right_cm_test = test_conf_matrix.trace()

    num_wrong_cm_train = train_labels.shape[0] - num_right_cm_train
    num_wrong_cm_val = val_labels.shape[0] - num_right_cm_val
    num_wrong_cm_test = test_labels.shape[0] - num_right_cm_test

    classified_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'classified': [num_right_cm_train, num_right_cm_val, num_right_cm_test], 
        'misclassified': [num_wrong_cm_train, num_wrong_cm_val, num_wrong_cm_test]}, 
        index=["training set", "validation set", "testing set"])
    
    return classified_df
