import os
import pickle

from data_loader import SNLIDataset
from torch.utils.data import DataLoader
from model import SNLIModel
from utils import get_device
from torch import nn, optim
from time import time
import torch

GPU = 3


def train(num_epochs, train_data_loader, dev_data_loader, model, device, lr, weight_decay, idx_to_tag):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_train_loss, total_dev_loss = [], []
    for epoch in range(num_epochs):
        model.train()
        current_num_samples = 0
        train_words_seen = 0
        total_train_current_loss = 0.
        true_current_preds = 0
        train_samples_for_acc = 0.
        epoch_num_samples = 0
        for batch in train_data_loader:
            optimizer.zero_grad()

            chars_seq1, words_seq1, len_seq1, len_words1, chars_seq2, words_seq2, len_seq2, len_words2, tags = batch
            chars_seq1, words_seq1, chars_seq2, words_seq2, tags = chars_seq1.to(device), words_seq1.to(device), \
                                                                   chars_seq2.to(device), words_seq2.to(device), tags.to(device)
            preds = model(chars_seq1, words_seq1, chars_seq2, words_seq2)

    #         new_preds = torch.cat([preds[i, :ln] for i, ln in enumerate(len_seq)])
    #         new_tags = torch.cat([tag[i, :ln] for i, ln in enumerate(len_seq)])
    #         loss = criterion(new_preds, new_tags)
    #         loss.backward()
    #         optimizer.step()
    #         total_train_current_loss += loss.item()
    #         new_preds = torch.argmax(new_preds, dim=1)
    #         epoch_num_samples += seq.size(0)
    #         current_num_samples += seq.size(0)
    #         train_words_seen += new_preds.size(0)
    #
    #         true_current_preds += (new_preds == new_tags).float().sum().item()
    #         acc_samples = len(new_preds)
    #         # current_acc, acc_samples = accuracy_ner(new_preds, new_tags, idx_to_tag)
    #         # true_current_preds += current_acc
    #         train_samples_for_acc += acc_samples
    #         if current_num_samples >= 500 and not epochs_print:
    #             train_loss = total_train_current_loss / train_words_seen
    #             train_accuracy = true_current_preds / train_samples_for_acc
    #             total_train_loss.append(train_loss)
    #             final_acc_train.append(train_accuracy)
    #             dev_loss, acc_dev = evaluate(dev_data_loader, model, device, idx_to_tag, criterion, mode=mode,
    #                                          task=task)
    #             total_dev_loss.append(dev_loss)
    #             final_acc_dev.append(acc_dev)
    #             model.train()
    #             train_words_seen = 0.
    #             true_current_preds = 0.
    #             total_train_current_loss = 0.
    #             train_samples_for_acc = 0.
    #             current_num_samples = 0
    #             print(
    #                 f"Dataset: {mode} | Epoch {epoch} | Num Sentences: {epoch_num_samples} | Train Acc {train_accuracy} | Train Loss: {train_loss} |"
    #                 f" Dev Acc {acc_dev} | Dev Loss: {dev_loss} ")
    #     if epochs_print:
    #         train_loss = total_train_current_loss / train_words_seen
    #         train_accuracy = true_current_preds / train_samples_for_acc
    #         total_train_loss.append(train_loss)
    #         final_acc_train.append(train_accuracy)
    #         dev_loss, acc_dev = evaluate(dev_data_loader, model, device, idx_to_tag, criterion, mode=mode, task=task)
    #         total_dev_loss.append(dev_loss)
    #         final_acc_dev.append(acc_dev)
    #         model.train()
    #         print(
    #             f"Dataset: {mode} | Epoch {epoch} | Num Sentences: {current_num_samples} | Train Acc {train_accuracy} | Train Loss: {train_loss} |"
    #             f" Dev Acc {acc_dev} | Dev Loss: {dev_loss} ")
    # final_t = time() - t
    # print(f"All run took {final_t} seconds")
    # if save:
    #     path_to_save_model = Path(path_to_save + f"_{mode}_{task}.pt")
    #     path_to_save_model.parent.mkdir(parents=True, exist_ok=True)
    #     torch.save(model.state_dict(), path_to_save_model)
    final_acc_train, final_acc_dev = [], []
    t = time()

    return final_acc_dev, model


def evaluate(data_loader, model, device, idx_to_tag, criterion, mode="ner", task="a"):
    model.eval()
    total_test_epoch_loss = 0.
    true_epoch_preds = 0
    test_samples_for_acc = 0.
    for batch in data_loader:
        with torch.no_grad():
            if task == "a":
                seq, tag, len_seq = batch
                seq, tag = seq.to(device), tag.to(device)
                preds = model(seq)
            elif task == "b":
                seq, tag, len_seq, len_words = batch
                seq, tag = seq.to(device), tag.to(device)
                preds = model(seq, len_seq)
            elif task == "c":
                seq, pref, suff, tag, len_seq = batch
                seq, pref, suff, tag = seq.to(device), pref.to(device), suff.to(device), tag.to(device)
                preds = model(seq, pref, suff)
            elif task == "d":
                chars_seq, words_seq, tag, len_seq, len_words = batch
                chars_seq, words_seq, tag = chars_seq.to(device), words_seq.to(device), tag.to(device)
                preds = model(chars_seq, words_seq, len_seq)
            new_preds = torch.cat([preds[i, :ln] for i, ln in enumerate(len_seq)])
            new_tags = torch.cat([tag[i, :ln] for i, ln in enumerate(len_seq)])
            test_loss = criterion(new_preds, new_tags)
            total_test_epoch_loss += test_loss.item()
            new_preds = torch.argmax(new_preds, dim=1)
            if mode == "pos":
                true_epoch_preds += (new_preds == new_tags).float().sum().item()
                acc_samples = len(new_preds)
            else:
                current_acc, acc_samples = accuracy_ner(new_preds, new_tags, idx_to_tag)
                true_epoch_preds += current_acc
            test_samples_for_acc += acc_samples
    acc = true_epoch_preds / test_samples_for_acc
    total_test_epoch_loss = total_test_epoch_loss / test_samples_for_acc
    return total_test_epoch_loss, acc


if __name__ == "__main__":
    print("if not its SUS")
    path = "snli_1.0/snli_1.0_train.txt"
    glove_path = "glove.840B.300d.pkl"
    train_dataset_path = "train_dataset.pt"
    if os.path.exists(train_dataset_path):
        train_dataset = torch.load(train_dataset_path)
    else:
        train_dataset = SNLIDataset(path, glove_path)
    print("if not its SUS but less")
    with open(glove_path, "rb") as f:
        pre_trained_vocab = pickle.load(f)
    keys = list(pre_trained_vocab.keys())
    for i, anomaly in enumerate(keys):
        if len(pre_trained_vocab[anomaly]) < 300:
            del pre_trained_vocab[keys[i]]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = None
    device = get_device(device=GPU)
    model = SNLIModel(len(train_dataset.words_to_index), train_dataset.unknown_words_in_train, len(train_dataset.chars_to_index),
                      word_embed_dim=300, char_embed_dim=100, pre_trained_embedding=pre_trained_vocab, device=device)
    final_acc_dev, final_model = train(5, train_loader, dev_loader, model, device, 0.0004,
                                       0, train_dataset.index_to_tag)
