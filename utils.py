import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from pathlib import Path

mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.loc'] = "lower right"
mpl.rcParams['grid.color'] = "black"


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


def plot_measure(measures, mode="Accuracy", gradient=False, weight_decay=0):
    """
    Receives the accuracies of all four representations and the mode (=task, 'pos' or 'ner') and plots the learning
    curves.
    The accuracies are assumed to be collected in the order requested in the question (we assume, for example, that
    the results were collected from a dataset that has a number of sentences divisible by 500).
    """
    epochs = 1 + np.arange(len(measures[0]))
    epochs = epochs.astype(int)
    plt.figure()
    for measure in measures:
        plt.plot(epochs, measure)
    plt.legend(["train", "dev", "test"])
    plt.xlabel("Number Of Epochs")
    plt.ylabel(f"{mode}")
    plt.title(f"{mode} Per Epoch")
    plt.locator_params(axis='x', nbins=len(epochs))
    pkl_path = Path(f"pkl/{mode}_gradient={gradient}_weight_decay={weight_decay}.pkl")
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(mode, f)
    plot_path = Path(f"plots/{mode}_gradient={gradient}_weight_decay={weight_decay}.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)


# def plot_measure_pkl(x_len=10):
#     """
#     Receives the accuracies of all four representations and the mode (=task, 'pos' or 'ner') and plots the learning
#     curves.
#     The accuracies are assumed to be collected in the order requested in the question (we assume, for example, that
#     the results were collected from a dataset that has a number of sentences divisible by 500).
#     """
#     epochs = 1 + np.arange(x_len)
#     epochs = epochs.astype(int)
#     plt.figure()
#     pkl_path = Path("pkl")
#     plot_path = Path("plots")
#     acc_trial_name = ["Accuracy_gradient=False_weight_decay=0.0", "Accuracy_gradient=False_weight_decay=3e-05",
#                       "Accuracy_gradient=True_weight_decay=0"]
#     loss_trial_name = ["Loss_gradient=False_weight_decay=0.0", "Loss_gradient=False_weight_decay=3e-05",
#                        "Loss_gradient=True_weight_decay=0"]
#     for acc_name, loss_name in zip(acc_trial_name, loss_trial_name):
#         acc_path = Path.joinpath(pkl_path, acc_name+".pkl")
#         acc_plot_path = Path.joinpath(plot_path, acc_name+".png")
#         loss_path = Path.joinpath(pkl_path, loss_name+".pkl")
#         loss_plot_path = Path.joinpath(plot_path, loss_name+".png")
#         with open(acc_path, 'rb') as f:
#             acc = pickle.load(f)
#         with open(loss_path, 'rb') as f:
#             loss = pickle.load(f)
#         plt.figure(0)
#         plt.plot(epochs, acc[:x_len])
#     plt.legend(["train", "dev", "test"])
#     plt.xlabel("Number Of Epochs")
#     plt.ylabel("Accuracy")
#     plt.title(f"Accuracy Per Epoch")
#     plt.locator_params(axis='x', nbins=len(epochs))
#     plot_path = Path(f"plots/{mode}_gradient={gradient}_weight_decay={weight_decay}.png")
#     plot_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(plot_path)


def plot_loss_acc(scores, mode_task="POS", mode_plot="Loss", part="1"):
    plt.figure(figsize=(9, 7))
    epochs = [i+1 for i in range(len(scores))]
    plt.plot(epochs, scores, color="red", linewidth=3, label=f"Dev {mode_plot}")
    plt.title(f"Part {part} - {mode_task} Task: Dev {mode_plot} per Epoch", color="red")
    plt.xlabel("Epoch", color="grey")
    plt.ylabel(f"{mode_plot}", color="grey")
    plt.xticks(range(0, len(epochs)+1, 5))
    plt.grid()
    plt.savefig(f"dev_{mode_plot}_{mode_task}_{part}")


def clip_grads(model, clip_c):
    """
    Apply gradient clipping as the writers did in the paper's implementation:
    If a gradient is larger than clip_c ** 2, divide it by the 2-norm of all gradients vector times clip_c
    """
    total_norm = 0
    for p in filter(lambda x: x.requires_grad, model.parameters()):
        total_norm += (p.grad.data ** 2).sum()
    total_norm = total_norm ** 0.5
    for p in filter(lambda x: x.requires_grad, model.parameters()):
        p.grad.data = torch.where(p.grad.data > clip_c ** 2, p.grad.data.mul(1 / (clip_c * total_norm)), p.grad.data)

