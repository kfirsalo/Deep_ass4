import os
import pickle

from data_loader import SNLIDataset
from torch.utils.data import DataLoader
from model import SNLIModel
from utils import get_device
from torch import nn, optim
from time import time
import torch

GPU = 1


def train(num_epochs, train_data_loader, dev_data_loader, model, device, lr, weight_decay, epochs_print=True):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_train_loss, total_dev_loss = [], []
    final_acc_train, final_acc_dev = [], []
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
                                                                   chars_seq2.to(device), words_seq2.to(device), tags.to(device)
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
                final_acc_dev.append(acc_dev)
                total_dev_loss.append(dev_loss)
                model.train()
                train_words_seen = 0.
                true_current_preds = 0.
                total_train_current_loss = 0.
                train_samples_for_acc = 0.
                current_num_samples = 0
                print(
                    f"Dataset: SNLI | Epoch {epoch} | Num Sentences: {epoch_num_samples} | Train Acc {train_accuracy} | Train Loss: {train_loss} |"
                    f" Dev Acc {acc_dev} | Dev Loss: {dev_loss} ")
        if epochs_print:
            train_loss = total_train_current_loss / train_words_seen
            train_accuracy = true_current_preds / train_samples_for_acc
            total_train_loss.append(train_loss)
            final_acc_train.append(train_accuracy)
            dev_loss, acc_dev = evaluate(dev_data_loader, model, device, criterion)
            total_dev_loss.append(dev_loss)
            final_acc_dev.append(acc_dev)
            model.train()
            print(
                f"Dataset: SNLI | Epoch {epoch} | Num Sentences: {current_num_samples} | Train Acc {train_accuracy} | Train Loss: {train_loss} |"
                f" Dev Acc {acc_dev} | Dev Loss: {dev_loss} ")
    # final_t = time() - t
    # print(f"All run took {final_t} seconds")
    # if save:
    #     path_to_save_model = Path(path_to_save + f"_{mode}_{task}.pt")
    #     path_to_save_model.parent.mkdir(parents=True, exist_ok=True)
    #     torch.save(model.state_dict(), path_to_save_model)
    final_acc_train, final_acc_dev = [], []
    t = time()

    return final_acc_dev, model


def evaluate(data_loader, model, device, criterion):
    model.eval()
    total_test_epoch_loss = 0.
    true_epoch_preds = 0
    test_samples_for_acc = 0.
    for batch in data_loader:
        with torch.no_grad():
            chars_seq1, words_seq1, len_seq1, len_words1, chars_seq2, words_seq2, len_seq2, len_words2, tags = batch
            chars_seq1, words_seq1, chars_seq2, words_seq2, tags = chars_seq1.to(device), words_seq1.to(device), \
                                                                   chars_seq2.to(device), words_seq2.to(device), tags.to(device)
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


if __name__ == "__main__":
    print("if not its SUS")
    path = "snli_1.0/snli_1.0_train.txt"
    glove_path = "glove.840B.300d.pkl"
    train_dataset_path = "train_dataset.pt"
    dev_dataset_path = "snli_1.0/snli_1.0_dev.txt"
    if os.path.exists(train_dataset_path):
        train_dataset = torch.load(train_dataset_path)
        print("save the dataset mis frydman")
    else:
        train_dataset = SNLIDataset(path, glove_path)
        torch.save(train_dataset, train_dataset_path)
    dev_dataset = SNLIDataset(dev_dataset_path, train_word_vocab=train_dataset.words_to_index, train_char_vocab=train_dataset.chars_to_index, exist_tags=train_dataset.tag_to_index)
    print("if not its SUS but less")
    with open(glove_path, "rb") as f:
        pre_trained_vocab = pickle.load(f)
    keys = list(pre_trained_vocab.keys())
    for i, anomaly in enumerate(keys):
        if len(pre_trained_vocab[anomaly]) < 300:
            del pre_trained_vocab[keys[i]]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True, collate_fn=dev_dataset.collate_fn)
    device = get_device(device=GPU)
    model = SNLIModel(len(train_dataset.words_to_index), train_dataset.words_to_index, train_dataset.unknown_words_in_train, len(train_dataset.chars_to_index),
                      word_embed_dim=300, char_embed_dim=15, char_embed_dim_out=100, hidden_lstm_dim=100, pre_trained_embedding=train_dataset.pre_trained_vocab, dropout=0.5, device=device)
    final_acc_dev, final_model = train(5, train_loader, dev_loader, model, device, 0.0004, 0)
