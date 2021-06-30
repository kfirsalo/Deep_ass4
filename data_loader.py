import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import change_words, add_key_to_dict
import pickle


class SNLIDataset(Dataset):
    """
    For embedding every word as an output of an LSTM of embedded characters, we build a dataset such that every item
    includes:
        1. A padded tensor of the sequences of represented characters.
        2. A tensor of the tags corresponding to each word, padded the same way.
        3. The length of the sentence.
        4. The length of each word in the sentence.
    A batch of sentences will include:
        1. A 3-dimensional tensor, where the first dimension represents the sentence index, the second represents the
        word in the sentence and the third represents the letters in a word. Padding on the two last dimensions is applied.
        2. A tensor of tags, two dimensional (sentence by word in sentence) and padded like in (a).
        3. The lengths of each sentence.
        4. The lengths of each word in the sentences, padded such that the total length of this list is
        batch size * length of longest sentence. A padded word will have a length of 0.
    """

    def __init__(self, path, pre_trained_vocab_path=None, exist_tags=None, train_word_vocab=None, train_char_vocab=None):
        self.unknown_key = '__unknown__'  # define symbol for unknown word
        if train_word_vocab is None:
            with open(pre_trained_vocab_path, "rb") as f:
                self.pre_trained_vocab = pickle.load(f)
            keys = list(self.pre_trained_vocab.keys())
            for i, anomaly in enumerate(keys):
                if len(self.pre_trained_vocab[anomaly]) < 300:
                    del self.pre_trained_vocab[keys[i]]
            keys = list(self.pre_trained_vocab.keys())
            self.words_to_index = {keys[i]: i for i in range(len(self.pre_trained_vocab))}
            self.words_to_index.update({self.unknown_key: len(self.words_to_index)})
            self.chars_to_index, self.tag_to_index = {"__pad__": 0, self.unknown_key: 1}, {"entailment": 0, "contradiction": 1, "neutral": 2}
            self.data, self.index_to_tag, self.index_to_char, self.unknown_words_in_train = self.load_data(path, to_add=True)
        else:
            self.words_to_index, self.chars_to_index, self.tag_to_index = train_word_vocab, train_char_vocab, exist_tags
            self.data, self.index_to_tag, self.index_to_char, self.unknown_words_in_train = self.load_data(path, to_add=False)
        self.index_to_word = {i: w for w, i in self.words_to_index.items()}

    def load_data(self, path, to_add=True):
        final_sequences = []
        unknown_words_in_train = 0
        with open(path, "r") as file:
            next(file)
            for row in file:
                split_row = row.strip().split("\t")
                label, sentence1, sentence2 = split_row[0], split_row[5], split_row[6]
                if label == "-":
                    continue
                sentence_chars1, sentence_words1, sentence_chars_len1, words_len1, unknown_words_in_train = self.sentence_parse(
                    sentence1, unknown_words_in_train, to_add)
                sentence_chars2, sentence_words2, sentence_chars_len2, words_len2, unknown_words_in_train = self.sentence_parse(
                    sentence2, unknown_words_in_train, to_add)
                final_sequences.append([sentence_chars1, sentence_words1, sentence_chars_len1, words_len1,
                                        sentence_chars2, sentence_words2, sentence_chars_len2, words_len2,
                                        self.tag_to_index[label]])
        index_to_tag = {i: t for t, i in self.tag_to_index.items()}
        index_to_char = {i: c for c, i in self.chars_to_index.items()}
        return final_sequences, index_to_tag, index_to_char, unknown_words_in_train

    def sentence_parse(self, current_sentence, unknown_words_in_train, to_add=True):
        current_sentence = current_sentence.split(" ")
        sentence_chars, sentence_words, words_len = [], [], []
        word_index, char_index = len(self.words_to_index), len(self.chars_to_index)
        for word in current_sentence:
            word_as_chars = []
            # unique_words.add(word)
            words_len.append(len(word))
            for c in word:
                if c not in self.chars_to_index:
                    if not to_add:
                        c = self.unknown_key
                    else:
                        chars_to_index, char_index = add_key_to_dict(self.chars_to_index, c, char_index)
                word_as_chars.append(self.chars_to_index[c])
            word = change_words(word)  # lowercase and replace digits with DG as explained in pdf
            if word not in self.words_to_index:
                unknown_words_in_train += 1
                if not to_add:
                    word = self.unknown_key
                else:
                    words_to_index, word_index = add_key_to_dict(self.words_to_index, word, word_index)
            sentence_chars.append(word_as_chars)
            sentence_words.append(self.words_to_index[word])
        if len(sentence_chars) != 0:  # no need for empty sentences
            return sentence_chars, sentence_words, len(sentence_chars), words_len, unknown_words_in_train

    @staticmethod
    def _first_padding(token):
        """
        Pads a sentence by the number of characters.
        We treat the character representation exactly like in (b)
        """
        max_len = max([len(token[0][i]) for i in range(len(token[0]))])
        padded_data = np.zeros((token[2], max_len), dtype=int)
        for i in range(token[2]):
            padded_data[i, 0:token[3][i]] = token[0][i]
        return [padded_data, token[1], token[2], token[3]]

    @staticmethod
    def _second_padding(batch):
        """
        Pads a batch of sentences, like in (a) for the words and like in (b) for the characters.
        """
        sentence_lens = [seq[2] for seq in batch]
        all_word_lens = [seq[3][i] for seq in batch for i in range(len(seq[3]))]
        padded_word_lens = []
        batched_word_lens = [seq[0].shape[1] for seq in batch]
        longest_seq = max(sentence_lens)
        longest_word = max(batched_word_lens)
        batch_len = len(batch)
        padded_chars = torch.zeros((batch_len, longest_seq, longest_word), dtype=torch.long)
        padded_words = torch.zeros(batch_len, longest_seq, dtype=torch.long)
        word_len_idx = 0
        for i, seq_len in enumerate(sentence_lens):
            padded_chars[i, 0:seq_len, 0:batched_word_lens[i]] = batch[i][0]
            padded_words[i, 0:seq_len] = batch[i][1]
            padded_word_lens += (all_word_lens[word_len_idx:word_len_idx + seq_len] + [0] * (longest_seq - seq_len))
            word_len_idx += seq_len
        return [padded_chars, padded_words, sentence_lens, padded_word_lens]

    def collate_fn(self, batch):
        sentence1 = [batch[i][:4] for i in range(len(batch))]
        sentence2 = [batch[i][4:8] for i in range(len(batch))]
        labels = torch.tensor([batch[i][8] for i in range(len(batch))], dtype=torch.long)
        a = self._second_padding(sentence1)
        b = self._second_padding(sentence2)
        return a + b + [labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        padded_sentence = self._first_padding(self.data[i][:4]) + self._first_padding(self.data[i][4:8]) + [
            self.data[i][8]]
        return torch.tensor(padded_sentence[0]), torch.tensor(padded_sentence[1]), torch.tensor(padded_sentence[2],
                                                                                                dtype=torch.int), \
               torch.tensor(padded_sentence[3], dtype=torch.int), torch.tensor(padded_sentence[4]), torch.tensor(
            padded_sentence[5]), \
               torch.tensor(padded_sentence[6], dtype=torch.int), torch.tensor(padded_sentence[7], dtype=torch.int), \
               padded_sentence[8]


# def main():
#     path = "snli_1.0/snli_1.0_train.txt"
#     glove_path = "glove.840B.300d.pkl"
#     dataset = SNLIDataset(path, glove_path)
#     loader = DataLoader(dataset, batch_size=32, shuffle=True,
#                         collate_fn=dataset.collate_fn)
#     for batch in loader:
#         print(batch)
#
#
# main()
