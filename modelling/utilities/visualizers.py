import numpy as np
import pandas as pd

import matplotlib as mplt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
font = {'fontname': 'Helvetica'}

import matplotlib.cm as cm
import seaborn as sb

from sklearn.metrics import (
  accuracy_score, 
  precision_score, 
  recall_score, 
  f1_score, 
  roc_auc_score, 
  mean_squared_error, 
  mean_absolute_error)
from sklearn.manifold import TSNE

import itertools



def data_split_metric_values(Y_true, Y_pred, Y_pred_prob, metrics_to_use: list=['accuracy', 'precision', 'recall', 'f1', 'roc-auc']):
    """
    args:
        Y_true - a vector of the real Y values of a data split e.g. the 
        training set, validation set, test

        Y_pred - a vector of the predicted Y values of an ML model given 
        a data split e.g. a training set, validation set, test set

        given these arguments it creates a bar graph of all the relevant
        metrics in evaluating an ML model e.g. accuracy, precision,
        recall, and f1-score.
    """

    metrics = {
        'accuracy': accuracy_score(y_true=Y_true, y_pred=Y_pred),
        'precision': precision_score(y_true=Y_true, y_pred=Y_pred, average='weighted'),
        'recall': recall_score(y_true=Y_true, y_pred=Y_pred, average='weighted'),
        'f1': f1_score(y_true=Y_true, y_pred=Y_pred, average='weighted'),
        'roc-auc': Y_pred_prob if Y_pred_prob.all() == None else roc_auc_score(y_true=Y_true, y_score=Y_pred_prob, average='weighted', multi_class='ovr')
    }

    # create metric_values dictionary
    metric_values = {}
    for index, metric in enumerate(metrics_to_use):
      metric_values[metric] = metrics[metric]

    return metric_values

def view_words(word_vec: dict, word_range: int, title: str="untitled", save_img: bool=True, style: str='dark'):
    """
    suitable for all discrete input

    args:
        word_vec - key value pairs of the words and respective embeddings

        len_to_show - the limit in which each word vector is only allowed to show

        word range - if false then all words are shown but if a value 
        is given then number words shown are up to that value only
        
        word_range: int | bool=50
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # slice the dictionary to a particular range
    sliced_word_vec = dict(itertools.islice(word_vec.items(), word_range))

    # separate all word keys and their respective 
    # embeddings from each other and place in separate arrays
    words, embeddings = zip(*sliced_word_vec.items())
    words = np.array(words)
    embeddings = np.array(embeddings)

    # reduce length/dimensions of embeddings from 300 to 2
    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, random_state=0)

    # because there are 21624 words dimensionality of emb_red will go from 21624 x 300 to 21624 x 2
    emb_red = tsne_model.fit_transform(embeddings)

    # populate a new dictionary with new reduced embeddings with 2 dimensions
    word_vec_red = {}
    for index, key in enumerate(words):
        # extract x and ys in emb_red array
        x, y = emb_red[index]

        # populate dictionary with x and y coordinates
        if key not in word_vec_red:
            word_vec_red[key] = (x, y)


    # build and visualize
    fig = plt.figure(figsize=(15, 15))
    axis = fig.add_subplot()

    # plot the points
    axis.scatter(emb_red[:, 0], emb_red[:, 1], c=np.random.randn(emb_red.shape[0]), marker='p',alpha=0.75, cmap='magma')

    # annotate the points
    for iter, (word, coord) in enumerate(word_vec_red.items()):
        x, y = coord
        axis.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    axis.set_xlabel('x', )
    axis.set_ylabel('y', )
    axis.set_title(title, )
    
    if save_img:
        plt.savefig(f'./figures & images/{title}.png')
        plt.show()

def view_value_frequency(word_counts, colormap: str="plasma", title: str="untitled", save_img: bool=True, kind: str='barh', limit: int=6, asc: bool=True, style: str='dark'):
    """
    suitable for all discrete input

    plots either a horizontal bar graph to display frequency of words top 'limit' 
    words e.g. top 20 or a pie chart to display the percentages of the top 'limit' 
    words e.g. top 20, specified by the argument kind which can be either
    strings barh or pie

    main args:
        words_counts - is actually a the returned value of the method
        of a pandas series, e.g.
            hom_vocab = pd.Series(flat_hom)
            hom_counts = hom_vocab.value_counts()

        limit - is the number of values to only consider showing in
        the horizontal bar graph and the pie chart

        colormap - can be "viridis" | "crest" also
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # get either last few words or first feww words
    data = word_counts[:limit].sort_values(ascending=asc)

    cmap = cm.get_cmap(colormap)
    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()
    
    if kind == 'barh':        
        axis.barh(data.index, data.values, color=cmap(np.linspace(0, 1, len(data))))
        axis.set_xlabel('frequency', )
        axis.set_ylabel('value', )
        axis.set_title(title, )
        
    elif kind == 'pie':
        axis.pie(data, labels=data.index, autopct='%.2f%%', colors=cmap(np.linspace(0, 1, len(data))))
        axis.axis('equal')
        axis.set_title(title, )
    
    if save_img:
        plt.savefig(f'./figures & images/{title}.png')
        plt.show()

def multi_class_heatmap(conf_matrix, img_title: str="untitled", cmap: str='YlGnBu', save_img: bool=True, labels: list=["Non-Depressive", "Depressive"], style: str='dark'):
    """
    takes in the confusion matrix returned by the confusion_matrix()
    function from sklearn e.g. conf_matrix_train = confusion_matrix(
        Y_true_train, Y_pred_train, labels=np.unique(Y_true_train)
    )

    other args:
        cmap - the color map you want the confusion matrix chart to have.
        Other values can be 'flare'
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    axis = sb.heatmap(conf_matrix, cmap=cmap, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    axis.set_title(img_title, )

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_metric_values(metrics_df, img_title: str="untitled", save_img: bool=True, colormap: str='mako', style: str='dark'):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values

    calculate accuracy, precision, recall, and f1-score for every 
    data split using the defined data_split_metric_values() function 
    above:

    train_acc, train_prec, train_rec, train_f1 = data_split_metric_values(Y_true_train, Y_pred_train)
    val_acc, val_prec, val_rec, val_f1 = data_split_metric_values(Y_true_val, Y_pred_val)
    test_acc, test_prec, test_rec, test_f1 = data_split_metric_values(Y_true_test, Y_pred_test)

    metrics_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'accuracy': [train_acc, val_acc, test_acc], 
        'precision': [train_prec, val_prec, test_prec], 
        'recall': [train_rec, val_rec, test_rec], 
        'f1-score': [train_f1, val_f1, test_f1]
    })
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    colors = []

    # excludes the data split column
    n_metrics = metrics_df.shape[1] - 1
    rgb_colors = cm.get_cmap(colormap, n_metrics)
    for i in range(rgb_colors.N):
        rgb_color = rgb_colors(i)
        colors.append(str(mplt.colors.rgb2hex(rgb_color)))
    colors = np.array(colors)

    # sample n ids based on number of metrics of metrics df
    sampled_ids = np.random.choice(list(range(colors.shape[0])), size=n_metrics, replace=False)
    sampled_colors = colors[sampled_ids]

    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # uses the given array of the colors you want to use
    sb.set_palette(sb.color_palette(sampled_colors))

    # create accuracy, precision, recall, f1-score of training group
    # create accuracy, precision, recall, f1-score of validation group
    # create accuracy, precision, recall, f1-score of testing group
    df_exp = metrics_df.melt(id_vars='data_split', var_name='metric', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='metric', ax=axis)
    axis.set_title(img_title, )
    axis.set_yscale('log')
    axis.legend()

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_classified_labels(df, img_title: str="untitled", save_img: bool=True, colors: list=['#db7f8e', '#b27392'], style: str='dark'):
    """
    given a each list of the training, validation, and testing set
    groups accuracy, precision, recall, and f1-score, plot a bar
    graph that separates these three groups metric values

    calculates all misclassified vs classified labels for training,
    validation, and testing sets by taking in a dataframe called
    classified_df created with the following code:

    num_right_cm_train = conf_matrix_train.trace()
    num_right_cm_val = conf_matrix_val.trace()
    num_right_cm_test = conf_matrix_test.trace()

    num_wrong_cm_train = train_labels.shape[0] - num_right_cm_train
    num_wrong_cm_val = val_labels.shape[0] - num_right_cm_val
    num_wrong_cm_test = test_labels.shape[0] - num_right_cm_test

    classified_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'classified': [num_right_cm_train, num_right_cm_val, num_right_cm_test], 
        'misclassified': [num_wrong_cm_train, num_wrong_cm_val, num_wrong_cm_test]}, 
        index=["training set", "validation set", "testing set"])
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    fig = plt.figure(figsize=(15, 10))
    axis = fig.add_subplot()

    # uses the given array of the colors you want to use
    sb.set_palette(sb.color_palette(colors))

    # create classified and misclassified of training group
    # create classified and misclassified of validation group
    # create classified and misclassified of testing group
    df_exp = df.melt(id_vars='data_split', var_name='status', value_name='score')
    
    axis = sb.barplot(data=df_exp, x='data_split', y='score', hue='status', ax=axis)
    axis.set_title(img_title, )
    axis.legend()

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def view_label_freq(label_freq, img_title: str="untitled", save_img: bool=True, labels: list | pd.Series | np.ndarray=["DER", "NDG", "OFF", "HOM"], horizontal: bool=True, style: str='dark'):
    """
    suitable for all discrete input

    main args:
        label_freq - is actually a the returned value of the method
        of a pandas series, e.g.
            label_freq = df['label'].value_counts()
            label_freq

        labels - a list of all the labels we want to use in the 
        vertical bar graph
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    # plots the unique labels against the count of these unique labels

    axis = sb.barplot(x=label_freq.values, y=labels, palette="flare") \
        if horizontal == True else sb.barplot(x=labels, y=label_freq.values, palette="flare")
    x_label = "frequency" if horizontal == True else "value"
    y_label = "value" if horizontal == True else "frequency"
    axis.set_xlabel(x_label, )
    axis.set_ylabel(y_label, )
    axis.set_title(img_title, )

    if save_img:
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

def plot_all_features(X, hue=None, colormap: str='mako', style: str='dark'):
    """
    suitable for: all discrete inputs, both discrete and continuous inputs,
    and all continuous inputs

    args:
        X - the dataset we want all features to be visualized whether
        discrete or continuous

        hue - a string that if provided will make the diagonals
        of the pairplot to be bell curves of the provided string feature
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    sb.set_palette(colormap)
    sb.pairplot(X, hue=hue, plot_kws={'marker': 'p', 'linewidth': 1})

# for recommendation
def describe_col(df: pd.DataFrame, column: str):
    """
    args:
        df - pandas data frame
        column - column of data frame to observe unique values and frequency of each unique value
    """

    print(f'count/no. of occurences of each unique {column} out of {df.shape[0]}: \n')

    unique_ids = df[column].unique()
    print(f'total unique values: {len(unique_ids)}')

class ModelResults:
    def __init__(self, history, epochs):
        """
        args:
            history - the history dictionary attribute extracted 
            from the history object returned by the self.fit() 
            method of the tensorflow Model object 

            epochs - the epoch list attribute extracted from the history
            object returned by the self.fit() method of the tensorflow
            Model object
        """
        self.history = history
        self.epochs = epochs

    def _build_results(self, metrics_to_use: list):
        """
        builds the dictionary of results based on history object of 
        a tensorflow model

        returns the results dictionary with the format {'loss': 
        [24.1234, 12.1234, ..., 0.2134], 'val_loss': 
        [41.123, 21.4324, ..., 0.912]} and the number of epochs 
        extracted from the attribute epoch of the history object from
        tensorflow model.fit() method

        args:
            metrics_to_use - a list of strings of all the metrics to extract 
            and place in the dictionary
        """

        # extract the epoch attribute from the history object
        epochs = self.epochs
        results = {}
        for metric in metrics_to_use:
            if metric not in results:
                # extract the history attribute from the history object
                # which is a dictionary containing the metrics as keys, and
                # the the metric values over time at each epoch as the values
                results[metric] = self.history[metric]

        return results, epochs
    
    def export_results(self, dataset_id: str="untitled", metrics_to_use: list=['loss', 
                                            'val_loss', 
                                            'binary_crossentropy', 
                                            'val_binary_crossentropy', 
                                            'binary_accuracy', 
                                            'val_binary_accuracy', 
                                            'precision', 
                                            'val_precision', 
                                            'recall', 
                                            'val_recall', 
                                            'f1_m', 
                                            'val_f1_m', 
                                            'auc', 
                                            'val_auc',
                                            'categorical_crossentropy',
                                            'val_categorical_crossentropy'], save_img: bool=True):
        """
        args:
            metrics_to_use - a list of strings of all the metrics to extract 
            and place in the dictionary, must always be of even length
        """

        # extracts the dictionary of results and the number of epochs
        results, epochs = self._build_results(metrics_to_use)
        results_items = list(results.items())

        # we want to leave the user with the option to 
        for index in range(0, len(metrics_to_use) - 1, 2):
            # say 6 was the length of metrics to use
            # >>> list(range(0, 6 - 1, 2))
            # [0, 2, 4]
            metrics_indeces = (index, index + 1)
            curr_metric, curr_metric_perf = results_items[metrics_indeces[0]]
            curr_val_metric, curr_val_metric_perf = results_items[metrics_indeces[1]]
            print(curr_metric)
            print(curr_val_metric)
            curr_result = {
                curr_metric: curr_metric_perf,
                curr_val_metric: curr_val_metric_perf
            }
            print(curr_result)

            self.view_train_cross_results(
                results=curr_result,
                epochs=epochs, 
                curr_metrics_indeces=metrics_indeces,
                save_img=save_img,
                img_title="model performance using {} dataset for {} metric".format(dataset_id, curr_metric)
            )

    def view_train_cross_results(self, results: dict, epochs: list, curr_metrics_indeces: tuple, save_img: bool, img_title: str="untitled", style: str='dark'):
        """
        plots the number of epochs against the cost given cost values 
        across these epochs.
        
        main args:
            results - is a dictionary created by the utility preprocessor
            function build_results()
        """
        styles = {
            'dark': 'dark_background',
            'solarized': 'Solarized_Light2',
            '538': 'fivethirtyeight',
            'ggplot': 'ggplot',
        }

        plt.style.use(styles.get(style, 'default'))

        figure = plt.figure(figsize=(15, 10))
        axis = figure.add_subplot()

        styles = [
            ('p:', '#f54949'), 
            ('h-', '#f59a45'), 
            ('o--', '#afb809'), 
            ('x:','#51ad00'), 
            ('+:', '#03a65d'), 
            ('8-', '#035aa6'), 
            ('.--', '#03078a'), 
            ('>:', '#6902e6'),
            ('p-', '#c005e6'),
            ('h--', '#fa69a3'),
            ('o:', '#240511'),
            ('x-', '#052224'),
            ('+--', '#402708'),
            ('8:', '#000000')]

        for index, (key, value) in enumerate(results.items()):
            # value contains the array of metric values over epochs
            # e.g. [213.1234, 123.43, 43.431, ..., 0.1234]

            if key == "loss" or key == "val_loss":
                # e.g. loss, val_loss has indeces 0 and 1
                # binary_cross_entropy, val_binary_cross_entropy 
                # has indeces 2 and 3
                axis.plot(
                    np.arange(len(epochs)), 
                    value, 
                    styles[curr_metrics_indeces[index]][0], 
                    color=styles[curr_metrics_indeces[index]][1], 
                    alpha=0.5, 
                    label=key, 
                    markersize=10, 
                    linewidth=3)
            else:
                # here if the metric value is not hte loss or 
                # validation loss each element is rounded by 2 
                # digits and converted to a percentage value 
                # which is why it is multiplied by 100 in order
                # to get much accurate depiction of metric value
                # that is not in decimal format
                metric_perc = [round(val * 100, 2) for val in value]
                axis.plot(
                    np.arange(len(epochs)), 
                    metric_perc, 
                    styles[curr_metrics_indeces[index]][0], 
                    color=styles[curr_metrics_indeces[index]][1], 
                    alpha=0.5, 
                    label=key, 
                    markersize=10, 
                    linewidth=3)

        # annotate end of lines
        for index, (key, value) in enumerate(results.items()):        
            if key == "loss" or key == "val_loss":
                last_loss_rounded = round(value[-1], 2)
                axis.annotate(last_loss_rounded, xy=(epochs[-1], value[-1]), color='black', alpha=1)
            else: 
                last_metric_perc = round(value[-1] * 100, 2)
                axis.annotate(last_metric_perc, xy=(epochs[-1], value[-1] * 100), color='black', alpha=1)

        axis.set_ylabel('metric value', )
        axis.set_xlabel('epochs', )
        axis.set_title(img_title, )
        axis.legend()

        if save_img == True:
            plt.savefig(f'./figures & images/{img_title}.png')
            plt.show()

        # delete figure
        del figure

def view_all_splits_results(history_dict: dict, save_img: bool=True, img_title: str="untitled", style: str='dark'):
    """
    
    """
    styles = {
        'dark': 'dark_background',
        'solarized': 'Solarized_Light2',
        '538': 'fivethirtyeight',
        'ggplot': 'ggplot',
    }

    plt.style.use(styles.get(style, 'default'))

    history_df = pd.DataFrame(history_dict)
    print(history_df)

    palettes = np.array(['#f54949', '#f59a45', '#afb809', '#51ad00', '#03a65d', '#035aa6', '#03078a', '#6902e6', '#c005e6', '#fa69a3', '#240511', '#052224', '#402708', '#000000'])
    markers = np.array(['o', 'v', '^', '8', '*', 'p', 'h', ])#'x', '+', '>', 'd', 'H', '3', '4'])

    sampled_indeces = np.random.choice(list(range(len(markers))), size=history_df.shape[1], replace=False)

    print(palettes[sampled_indeces])
    print(markers[sampled_indeces])

    figure = plt.figure(figsize=(15, 10))
    axis = sb.lineplot(data=history_df, 
        palette=palettes[sampled_indeces].tolist(),
        markers=markers[sampled_indeces].tolist(), 
        linewidth=3.0,
        markersize=9,
        alpha=0.75)
    
    axis.set_ylabel('metric value', )
    axis.set_xlabel('epochs', )
    axis.set_title(img_title, )
    axis.legend()

    if save_img == True:
        print(save_img)
        plt.savefig(f'./figures & images/{img_title}.png')
        plt.show()

"""Here are all the available colormaps in matplotlib
['magma',
 'inferno',
 'plasma',
 'viridis',
 'cividis',
 'twilight',
 'twilight_shifted',
 'turbo',
 'Blues',
 'BrBG',
 'BuGn',
 'BuPu',
 'CMRmap',
 'GnBu',
 'Greens',
 'Greys',
 'OrRd',
 'Oranges',
 'PRGn',
 'PiYG',
 'PuBu',
 'PuBuGn',
 'PuOr',
 'PuRd',
 'Purples',
 'RdBu',
 'RdGy',
 'RdPu',
 'RdYlBu',
 'RdYlGn',
 'Reds',
 'Spectral',
 'Wistia',
 'YlGn',
 'YlGnBu',
 'YlOrBr',
 'YlOrRd',
 'afmhot',
 'autumn',
 'binary',
 'bone',
 'brg',
 'bwr',
 'cool',
 'coolwarm',
 'copper',
 'cubehelix',
 'flag',
 'gist_earth',
 'gist_gray',
 'gist_heat',
 'gist_ncar',
 'gist_rainbow',
 'gist_stern',
 'gist_yarg',
 'gnuplot',
 'gnuplot2',
 'gray',
 'hot',
 'hsv',
 'jet',
 'nipy_spectral',
 'ocean',
 'pink',
 'prism',
 'rainbow',
 'seismic',
 'spring',
 'summer',
 'terrain',
 'winter',
 'Accent',
 'Dark2',
 'Paired',
 'Pastel1',
 'Pastel2',
 'Set1',
 'Set2',
 'Set3',
 'tab10',
 'tab20',
 'tab20b',
 'tab20c',
 'grey',
 'gist_grey',
 'gist_yerg',
 'Grays',
 'magma_r',
 'inferno_r',
 'plasma_r',
 'viridis_r',
 'cividis_r',
 'twilight_r',
 'twilight_shifted_r',
 'turbo_r',
 'Blues_r',
 'BrBG_r',
 'BuGn_r',
 'BuPu_r',
 'CMRmap_r',
 'GnBu_r',
 'Greens_r',
 'Greys_r',
 'OrRd_r',
 'Oranges_r',
 'PRGn_r',
 'PiYG_r',
 'PuBu_r',
 'PuBuGn_r',
 'PuOr_r',
 'PuRd_r',
 'Purples_r',
 'RdBu_r',
 'RdGy_r',
 'RdPu_r',
 'RdYlBu_r',
 'RdYlGn_r',
 'Reds_r',
 'Spectral_r',
 'Wistia_r',
 'YlGn_r',
 'YlGnBu_r',
 'YlOrBr_r',
 'YlOrRd_r',
 'afmhot_r',
 'autumn_r',
 'binary_r',
 'bone_r',
 'brg_r',
 'bwr_r',
 'cool_r',
 'coolwarm_r',
 'copper_r',
 'cubehelix_r',
 'flag_r',
 'gist_earth_r',
 'gist_gray_r',
 'gist_heat_r',
 'gist_ncar_r',
 'gist_rainbow_r',
 'gist_stern_r',
 'gist_yarg_r',
 'gnuplot_r',
 'gnuplot2_r',
 'gray_r',
 'hot_r',
 'hsv_r',
 'jet_r',
 'nipy_spectral_r',
 'ocean_r',
 'pink_r',
 'prism_r',
 'rainbow_r',
 'seismic_r',
 'spring_r',
 'summer_r',
 'terrain_r',
 'winter_r',
 'Accent_r',
 'Dark2_r',
 'Paired_r',
 'Pastel1_r',
 'Pastel2_r',
 'Set1_r',
 'Set2_r',
 'Set3_r',
 'tab10_r',
 'tab20_r',
 'tab20b_r',
 'tab20c_r']
"""