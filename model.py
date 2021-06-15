import torch
from torch import nn

"""
Model description:

The encoding of each sentence is as follows:
1. From a sentence, we get two embeddings:
    a. A simple word embedding (Not trained).
    b. A character embedding, followed by a CNN:
        (1) Conv2d
        (2) ReLU
        (3) max-pooling
       3 times of this, with filter lengths in [1, 3, 5]. The outputs are then concatenated.
   The two embeddings are then concatenated and put through a dropout layer.
2. 3-layered Stacked BiLSTM - The inputs to the first layer are the embeddings; to the second layer - a concatenation of
                              the embedding and the output of the first layer; and to the third layer - a concatenation
                              of the embedding and the output of the second layer.
3. The gated attention - Summarizing the outputs of the last layer to form a constant-sized encoding per sentence
   (see more in the report).

From the encodings, a large vector is created:
From the representations h1, h2, the representation of a pair is a concatenation [h1, h2, |h1-h2|, h1 * h2],
where * denotes element-wise (Hadamard) product.

The vector enters an MLP, implemented as follows:
1. Feed-Forward layer, then ReLU.
2. Dropout
3. Concatenate the input and the output to serve as the input for the next layer.
4. Feed-Forward layer, then ReLU.
5. Dropout
6. Feed-Forward layer
7. Softmax
8. To the CELoss.
"""


class SNLIModel(nn.Module):
    def __init__(self, word_vocab_size, unknown_words_size, char_vocab_size, word_embed_dim, char_embed_dim,
                 pre_trained_embedding, device="cuda:1"):
        """
        :param word_vocab_size: The number of different words we collected in the training corpus.
        :param char_vocab_size: The number of different characters we collected in the training corpus.
        :param word_embed_dim: The dimension of word embedding (before LSTM, concatenation with characters and linear layer).
        :param char_embed_dim: The dimension of character embedding.
        :param lstm_chars_dim: The dimension of the output from the lstm on characters.
        :param words_linear_dim: The dimension of the output from the linear layer that receives the concatenated embeddings.
        :param lstm_dim: The output dimension of each direction from the biLSTM.
        :param num_tags: The number of different tags.
        :param dropout: Dropout between layers
        :param device: Device (cuda or cpu)
        """
        super(SNLIModel, self).__init__()
        # self.lstm_dim = lstm_dim
        # self.lstm_char_dim = lstm_chars_dim
        self.device = device

        self.word_embedding = nn.Embedding(num_embeddings=len(pre_trained_embedding) + unknown_words_size, embedding_dim=word_embed_dim)
        self.word_embedding.weight.data[:len(pre_trained_embedding), :] = torch.tensor(list(pre_trained_embedding.values()))
        self.word_embedding.weight.requires_grad = False

        self.chars_embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embed_dim)
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(1, 100))
        self.cnn_2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100))
        self.cnn_3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, 100))
        self.max_pooling = nn.MaxPool1d(kernel_size=self.longest_word)
        # self.lstm_chars = nn.LSTM(char_embed_dim, lstm_chars_dim, batch_first=True)
        # self.words_linear = nn.Linear(lstm_chars_dim + word_embed_dim, words_linear_dim)
        # self.lstm = nn.LSTM(words_linear_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=2,
        #                     dropout=dropout)
        # self.linear1 = nn.Linear(2 * lstm_dim, num_tags)

    @staticmethod
    def original_size(seq, dim=1):
        return (seq != 0).sum(dim=dim).sort(dim - 1, descending=True)

    def _init_hidden(self, batch_size):
        return (torch.zeros((4, batch_size, self.lstm_dim), device=self.device),
                torch.zeros((4, batch_size, self.lstm_dim), device=self.device))

    def cnn_forward(self, seq_words, seq_chars):
        word_embedding = self.word_embedding(seq_words)

        list_char_embed = []
        char_embedding = self.chars_embedding(seq_chars)
        char_embedding = char_embedding.permute(0, 3, 1, 2)
        # cnn_embedding = None
        for cnn_layer in [self.cnn_1, self.cnn_2, self.cnn_3]:
            embed = cnn_layer(char_embedding)
            embed = torch.relu(embed)
            embed = self.max_pooling(embed)
            list_char_embed.append(embed)
        char_embedding = torch.cat(list_char_embed)
        return char_embedding, word_embedding

    def forward(self, seq_chars1, seq_words1, seq_chars2, seq_words2):
        char_embed1, word_embed1 = self.cnn_forward(seq_words1, seq_chars1)
        char_embed2, word_embed2 = self.cnn_forward(seq_words2, seq_chars2)

            # char_embedding = char_embedding.view(char_embedding.shape[0], -1)

        # First part - character embedding like (b)
        # num_sentence, num_words, num_chars = seq_chars.shape
        # seq_chars = seq_chars.view(num_sentence * num_words, num_chars)  # To work the character LSTM by word.
        #
        # words_len, word_len_sort_ind = self.original_size(seq_chars, dim=1)
        # words = seq_chars[word_len_sort_ind]
        # positive_lens = words_len[words_len > 0]  # To run the word embedding, we need non-empty words.
        # true_size = positive_lens.size(0)
        # cut_words = words[:true_size]
        # embeds = self.chars_embedding(cut_words)
        #
        # packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, words_len[:true_size], batch_first=True)
        # char_lstm_out, _ = self.lstm_chars(packed_embeds)
        # char_lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(char_lstm_out, batch_first=True)
        # char_lstm_out = torch.cat([char_lstm_out[i, j.data - 1] for i, j in enumerate(lengths)]).view(len(lengths),
        #                                                                                               self.lstm_char_dim)
        # char_lstm_out = torch.cat(
        #     (char_lstm_out, torch.zeros(words.shape[0] - true_size, self.lstm_char_dim, device=self.device)))
        #
        # _, original_order = word_len_sort_ind.sort(0)
        # char_lstm_out = char_lstm_out[original_order]
        # char_lstm_out = char_lstm_out.view(num_sentence, num_words, -1)  # Order restored
        #
        # # Second part - word embedding like (a)
        # words_embedding = self.word_embedding(seq_words)
        #
        # # Third part - concatenating and a linear layer
        # final_words_embedding = torch.cat((words_embedding, char_lstm_out), dim=2)
        # final_words_embedding = self.words_linear(final_words_embedding)
        # final_words_embedding = torch.tanh(final_words_embedding)
        #
        # # Fourth part - having the word embedding, the pipeline continues as always
        # sentence_len, sent_len_sort_ind = torch.tensor(sentence_lens).sort(0, descending=True)
        # sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        # sentences = final_words_embedding[sent_len_sort_ind]
        # new_embeds = torch.nn.utils.rnn.pack_padded_sequence(sentences, sentence_len, batch_first=True)
        # lstm_out, _ = self.lstm(new_embeds, self._init_hidden(sentences.size(0)))
        # lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        #
        # _, original_order = sent_len_sort_ind.sort(0)
        # lstm_out = lstm_out[original_order]
        #
        # out = self.linear1(lstm_out)  # linear layer

        return char_embed1
