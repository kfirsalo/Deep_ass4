import numpy as np
import torch
# import matplotlib.pyplot as plt
import pickle
# import matplotlib as mpl

#
# mpl.rcParams['xtick.labelsize'] = 10
# mpl.rcParams['ytick.labelsize'] = 10
# mpl.rcParams['axes.titlesize'] = 18
# mpl.rcParams['axes.labelsize'] = 14
# mpl.rcParams['legend.loc'] = "lower right"
# mpl.rcParams['grid.color'] = "black"


def get_device(device):
    return device if torch.cuda.is_available() else 'cpu'


def change_words(word):
    # For the data_loaders.py file. The words of the vocab need to be lowercase and digits should be
    # converted to DG format like the pre-trained vocab (0.05 -> DG.DGDG) (explained in the pdf)
    word = word.lower()  # lower case
    new_word = ''
    for letter in word:
        if letter.isnumeric():  # convert the letter to 'DG' if it is a digit
            new_word += 'DG'
        else:
            new_word += letter
    return new_word


def add_key_to_dict(dict_, key, idx):
    # Add a key to a given dict and forward the index
    dict_.update({key: idx})
    idx += 1
    return dict_, idx


def build_prefix(word):
    # Function to build the 3 length prefix of a word
    pref = ""
    for i in range(3):
        try:
            letter = word[i]
            if letter.isnumeric():  # convert the letter to 'DG' if it is a digit
                pref += "DG"
            else:
                pref += letter
        except IndexError:
            pref += "_"  # padding
    return pref


def build_suffix(word):
    # Function to build the 3 length suffix of a word
    suff = ''
    for i in range(-1, -4, -1):
        try:
            letter = word[i]
            if letter.isnumeric():
                suff = 'DG' + suff  # convert the letter to 'DG' if it is a digit
            else:
                suff = letter + suff
        except IndexError:
            suff = '_' + suff  # padding
    return suff


def accuracy_ner(preds, labels, index_to_tag):
    # Calculate accuracy for POS and NER tasks. If we are in POS task it is calculate regularly.
    # If we are in NER task, we should ignore the words that are labeled as 'O' and are also predicted
    # to be 'O'.
    preds = preds.tolist()
    labels = labels.tolist()
    acc = 0
    num = 0
    for i in range(len(preds)):
        p, l = preds[i], labels[i]
        if index_to_tag[p] == index_to_tag[l] and index_to_tag[l] == 'O':
            continue
        else:
            num += 1
            acc += 1 if index_to_tag[p] == index_to_tag[l] else 0
    return acc, num


# def plot_accuracies(accuracies, mode):
#     """
#     Receives the accuracies of all four representations and the mode (=task, 'pos' or 'ner') and plots the learning
#     curves.
#     The accuracies are assumed to be collected in the order requested in the question (we assume, for example, that
#     the results were collected from a dataset that has a number of sentences divisible by 500).
#     """
#     sentences_seen = 5 * (1 + np.arange(len(accuracies[0])))
#     plt.figure()
#     for acc in accuracies:
#         plt.plot(sentences_seen, acc)
#     plt.legend(["a", "b", "c", "d"])
#     plt.xlabel("Seen sentences / 100")
#     plt.ylabel("Accuracy")
#     plt.title("Learning curves - %s data" % mode)
#     with open("accuracies_%s.pkl" % mode, "wb") as f:
#         pickle.dump(accuracies, f)
#     plt.savefig("accuracies_%s.png" % mode)

