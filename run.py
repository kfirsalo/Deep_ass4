import os
import pickle

from data_loader import SNLIDataset
from torch.utils.data import DataLoader
from model import SNLIModel
from utils import get_device, plot_measure
from torch import nn, optim
from time import time
import torch
import argparse

GPU = 0


def train(num_epochs, train_data_loader, dev_data_loader, test_data_loader, model, device, lr, weight_decay, wait_n,
          epochs_print=True):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_train_loss, total_dev_loss, total_test_loss = [], [], []
    final_acc_train, final_acc_dev, final_acc_test = [], [], []
    wait_counter = 0  # Number of epochs in which the dev accuracy is not the minimal seen.
    bad_counter = 0  # Number of updates
    for epoch in range(num_epochs):
        model.train()
        current_num_samples = 0
        train_words_seen = 0
        total_train_current_loss = 0.
        true_current_preds = 0
        train_samples_for_acc = 0.
        epoch_num_samples = 0
        total_train_current_loss = 0
        for batch in train_data_loader:
            optimizer.zero_grad()

            chars_seq1, words_seq1, len_seq1, len_words1, chars_seq2, words_seq2, len_seq2, len_words2, tags = batch
            chars_seq1, words_seq1, chars_seq2, words_seq2, tags = chars_seq1.to(device), words_seq1.to(device), \
                                                                   chars_seq2.to(device), words_seq2.to(
                device), tags.to(device)
            preds = model(chars_seq1, words_seq1, chars_seq2, words_seq2, len_seq1, len_seq2, len_words1, len_words2)
            loss = criterion(preds, tags)
            loss.backward()
            optimizer.step()
            total_train_current_loss += loss.item()
            preds = torch.argmax(preds, dim=1)
            epoch_num_samples += batch[0].shape[0]
            current_num_samples += batch[0].shape[0]
            train_words_seen += preds.size(0)
            true_current_preds += (preds == tags).float().sum().item()
            acc_samples = len(preds)
            train_samples_for_acc += acc_samples

            if current_num_samples >= 1280 and not epochs_print:
                train_loss = total_train_current_loss / train_words_seen
                train_accuracy = true_current_preds / train_samples_for_acc
                total_train_loss.append(train_loss)
                final_acc_train.append(train_accuracy)
                dev_loss, acc_dev = evaluate(dev_data_loader, model, device, criterion)
                test_loss, acc_test = evaluate(test_data_loader, model, device, criterion)
                final_acc_dev.append(acc_dev)
                total_dev_loss.append(dev_loss)
                wait_counter, bad_counter, lr, optimizer = update_optimizer(acc_dev, final_acc_dev, wait_counter,
                                                                            bad_counter, lr, optimizer, wait_n)
                model.train()
                train_words_seen = 0.
                true_current_preds = 0.
                total_train_current_loss = 0.
                train_samples_for_acc = 0.
                current_num_samples = 0
                print(
                    f"Dataset: SNLI | Epoch {epoch} | Num Sentences: {epoch_num_samples} | Train Acc {train_accuracy} | Train Loss: {train_loss} |"
                    f" Dev Acc {acc_dev} | Dev Loss: {dev_loss} | Test Acc {acc_test} | Test Loss: {test_loss} ")
        if epochs_print:
            train_loss = total_train_current_loss / train_words_seen
            train_accuracy = true_current_preds / train_samples_for_acc
            total_train_loss.append(train_loss)
            final_acc_train.append(train_accuracy)
            dev_loss, acc_dev = evaluate(dev_data_loader, model, device, criterion)
            test_loss, acc_test = evaluate(test_data_loader, model, device, criterion)
            total_dev_loss.append(dev_loss)
            final_acc_dev.append(acc_dev)
            final_acc_test.append(acc_test)
            total_test_loss.append(test_loss)
            wait_counter, bad_counter, lr, optimizer = update_optimizer(acc_dev, final_acc_dev, wait_counter,
                                                                        bad_counter, lr, optimizer, wait_n)
            model.train()
            print(
                f"Dataset: SNLI | Epoch {epoch} | Num Sentences: {current_num_samples} | Train Acc {train_accuracy} | Train Loss: {train_loss} |"
                f" Dev Acc {acc_dev} | Dev Loss: {dev_loss} | Test Acc {acc_test} | Test Loss: {test_loss}  ")
    # final_t = time() - t
    # print(f"All run took {final_t} seconds")
    # if save:
    #     path_to_save_model = Path(path_to_save + f"_{mode}_{task}.pt")
    #     path_to_save_model.parent.mkdir(parents=True, exist_ok=True)
    #     torch.save(model.state_dict(), path_to_save_model)
    # final_acc_train, final_acc_dev = [], []
    # t = time()
    final_acc = [final_acc_train, final_acc_dev, final_acc_test]
    final_loss = [total_train_loss, total_dev_loss, total_test_loss]

    return final_acc, final_loss, model


def update_optimizer(dev_accuracy, dev_accuracies, wait_counter, bad_counter, lr, optimizer, wait_n):
    if dev_accuracy > max(dev_accuracies):
        wait_counter = 0
    elif dev_accuracy < max(dev_accuracies):
        wait_counter += 1

    if wait_counter >= wait_n:  # Waited enough epochs for improvement - updating the learning rate.
        print("Updating learning rate")
        bad_counter += 1
        wait_counter = 0
        lr *= 0.5

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return wait_counter, bad_counter, lr, optimizer


def evaluate(data_loader, model, device, criterion):
    model.eval()
    total_test_epoch_loss = 0.
    true_epoch_preds = 0
    test_samples_for_acc = 0.
    for batch in data_loader:
        with torch.no_grad():
            chars_seq1, words_seq1, len_seq1, len_words1, chars_seq2, words_seq2, len_seq2, len_words2, tags = batch
            chars_seq1, words_seq1, chars_seq2, words_seq2, tags = chars_seq1.to(device), words_seq1.to(device), \
                                                                   chars_seq2.to(device), words_seq2.to(
                device), tags.to(device)
            preds = model(chars_seq1, words_seq1, chars_seq2, words_seq2, len_seq1, len_seq2, len_words1, len_words2)
            loss = criterion(preds, tags)
            total_test_epoch_loss += loss.item()
            preds = torch.argmax(preds, dim=1)
            true_epoch_preds += (preds == tags).float().sum().item()
            acc_samples = len(preds)
            test_samples_for_acc += acc_samples
    acc = true_epoch_preds / test_samples_for_acc
    total_test_epoch_loss = total_test_epoch_loss / test_samples_for_acc
    return total_test_epoch_loss, acc


def run_for_plot(train_dataset, train_loader, dev_loader, test_loader, device, weight_decays, wait_n):
    model = SNLIModel(len(train_dataset.words_to_index), train_dataset.words_to_index,
                      len(train_dataset.chars_to_index),
                      word_embed_dim=300, char_embed_dim=15, char_embed_dim_out=100, hidden_lstm_dim=100,
                      pre_trained_embedding=train_dataset.pre_trained_vocab, dropout=0.5, device=device)
    for weight_decay in weight_decays:
        final_acc, final_loss, _ = train(30, train_loader, dev_loader, test_loader, model, device, 0.0004,
                                         weight_decay=weight_decay, wait_n=wait_n,
                                         epochs_print=True)
        plot_measure(final_acc, mode="Accuracy", gradient=False, weight_decay=weight_decay)
        plot_measure(final_loss, mode="Loss", gradient=False, weight_decay=weight_decay)
    final_acc, final_loss, _ = train(30, train_loader, dev_loader, test_loader, model, device, 0.0004, weight_decay=0,
                                     wait_n=wait_n, epochs_print=True)
    plot_measure(final_acc, mode="Accuracy", gradient=False, weight_decay=0)
    plot_measure(final_loss, mode="Loss", gradient=False, weight_decay=0)
    model = SNLIModel(len(train_dataset.words_to_index), train_dataset.words_to_index,
                      len(train_dataset.chars_to_index),
                      word_embed_dim=300, char_embed_dim=15, char_embed_dim_out=100, hidden_lstm_dim=100,
                      pre_trained_embedding=train_dataset.pre_trained_vocab, dropout=0.5, device=device,
                      is_gradient=True)
    final_acc, final_loss, _ = train(30, train_loader, dev_loader, test_loader, model, device, 0.0004, weight_decay=0,
                                     wait_n=wait_n, epochs_print=True)
    plot_measure(final_acc, mode="Accuracy", gradient=True, weight_decay=0)
    plot_measure(final_loss, mode="Loss", gradient=True, weight_decay=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainFile", type=str, default="snli_1.0/snli_1.0_train.txt",
                        help="Relative path to train data (input file to train on) (str)")
    parser.add_argument("--gloveFile", type=str, default="glove.840B.300d.pkl",
                        help="Path to Glove pretrained word embedding")
    parser.add_argument("--devFile", type=str, default="snli_1.0/snli_1.0_dev.txt",
                        help="Relative path to dev data (str)")
    parser.add_argument("--testFile", type=str, default="snli_1.0/snli_1.0_test.txt",
                        help="Relative path to test data (str)")
    parser.add_argument("--device", type=str, default='cuda:1',
                        help="Device to use when running the model. Either 'cpu' or 'cuda:integer' for an integer "
                             "indicating on which gpu to run, assuming it is possible. For example: cuda:0. (str)")
    parser.add_argument("--plot", type=str, default="False",
                        help="Whether to plot the requested graph. (str)")

    params = {"word_embed_dim": 300,  # Word embedding dimension (always 300 if our GloVe is used)
              "char_embed_dim": 15,  # Character embedding dimension
              "dropout": 0.5,  # Dropout rate in all dropout
              "char_embed_out": 100,  # Output dimension of character embedding from each CNN layer
              "lstm_hid": 600,  # Hidden layer size in LSTMs
              "epochs": 30,  # Maximal number of epochs to run the model
              "batch_size": 32,  # Mini-batch size
              "lr": 0.0004,  # Initial learning rate (might get smaller when running)
              "reg": 0.,  # L2 regularization coefficient
              "wait_n": 1,  # Number of epochs to wait until changing the learning rate to half its value
              "is_grad": True
    }

    args = parser.parse_args()

    print("...Start The Run...")
    glove_path = args.gloveFile
    train_dataset_path = "train_dataset.pt"
    train_path = args.trainFile
    dev_path = args.devFile
    test_path = args.testFile
    if os.path.exists(train_dataset_path):
        train_dataset = torch.load(train_dataset_path)
    else:
        train_dataset = SNLIDataset(train_path, glove_path)
        torch.save(train_dataset, train_dataset_path)

    draw_plot = args.plot
    dev_dataset = SNLIDataset(dev_path, train_word_vocab=train_dataset.words_to_index,
                              train_char_vocab=train_dataset.chars_to_index, exist_tags=train_dataset.tag_to_index)
    test_dataset = SNLIDataset(test_path, train_word_vocab=train_dataset.words_to_index,
                               train_char_vocab=train_dataset.chars_to_index, exist_tags=train_dataset.tag_to_index)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=params["batch_size"], shuffle=True, collate_fn=dev_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, collate_fn=test_dataset.collate_fn)
    device = get_device(device=GPU)
    #if draw_plot:
#   run_for_plot(train_dataset, train_loader, dev_loader, test_loader, device, weight_decays=[0, 3e-5])
    #else:
    model = SNLIModel(len(train_dataset.words_to_index), train_dataset.words_to_index,
                      len(train_dataset.chars_to_index),
                      word_embed_dim=params["word_embed_dim"], char_embed_dim=params["char_embed_dim"],
                      char_embed_dim_out=params["char_embed_out"], hidden_lstm_dim=params["lstm_hid"],
                      pre_trained_embedding=train_dataset.pre_trained_vocab, dropout=params["dropout"], device=device,
                      is_gradient=params["is_grad"])
    final_acc, final_loss, final_model = train(params["epochs"], train_loader, dev_loader, test_loader, model,
                                               device, params["lr"], params["reg"], params["wait_n"], epochs_print=True)
    if draw_plot:
        plot_measure(final_acc, mode="Accuracy", gradient=params["is_grad"], weight_decay=params["reg"])
        plot_measure(final_loss, mode="Loss", gradient=params["is_grad"], weight_decay=params["reg"])
