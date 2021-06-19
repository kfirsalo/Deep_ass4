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
    def __init__(self, word_vocab_size, words_to_index, char_vocab_size, word_embed_dim,
                 char_embed_dim, char_embed_dim_out, hidden_lstm_dim,
                 pre_trained_embedding, dropout, device="cuda:1", is_gradient=False):
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

        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_embed_dim)
        self.chars_embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embed_dim)
        self.word_embedding.weight.data, self.chars_embedding.weight.data = map(
            self._randn_weights, [self.word_embedding.weight.data, self.chars_embedding.weight.data])
        for idx, vector in pre_trained_embedding.items():
            self.word_embedding.weight.data[words_to_index[idx]] = torch.tensor(vector)
        self.word_embedding.weight.requires_grad = is_gradient

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=char_embed_dim_out, kernel_size=(1, char_embed_dim))
        self.cnn_2 = nn.Conv2d(in_channels=1, out_channels=char_embed_dim_out, kernel_size=(3, char_embed_dim))
        self.cnn_3 = nn.Conv2d(in_channels=1, out_channels=char_embed_dim_out, kernel_size=(5, char_embed_dim))
        self.dropout_cnn = nn.Dropout(p=dropout)
        # self.lstm_chars = nn.LSTM(char_embed_dim, lstm_chars_dim, batch_first=True)
        # self.words_linear = nn.Linear(lstm_chars_dim + word_embed_dim, words_linear_dim)
        embed_dim = word_embed_dim + 3 * char_embed_dim_out
        self.bi_lstm1 = nn.LSTM(embed_dim, hidden_lstm_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.bi_lstm2 = nn.LSTM(embed_dim + 2 * hidden_lstm_dim, hidden_lstm_dim, batch_first=True, bidirectional=True,
                                num_layers=1)
        self.bi_lstm3 = nn.LSTM(embed_dim + 2 * hidden_lstm_dim, hidden_lstm_dim, batch_first=True, bidirectional=True,
                                num_layers=1)
        self.linear1 = nn.Linear(12 * (embed_dim + 2 * hidden_lstm_dim), hidden_lstm_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(12 * (embed_dim + 2 * hidden_lstm_dim) + hidden_lstm_dim, hidden_lstm_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hidden_lstm_dim, 3)

        self._init_layers(char_embed_dim)

    def _init_layers(self, char_embed_dim):
        """
        Initialize layer weights and pre-trained embedding.
        All biases are initialized as zeros.
        """
        self.cnn_1.weight.data = self._cnn_weights(self.cnn_1.weight.data, 1, char_embed_dim)
        self.cnn_2.weight.data = self._cnn_weights(self.cnn_2.weight.data, 3, char_embed_dim)
        self.cnn_3.weight.data = self._cnn_weights(self.cnn_3.weight.data, 5, char_embed_dim)

        self.bi_lstm1.weight_ih_l0.data, self.bi_lstm1.weight_ih_l0_reverse.data, self.bi_lstm2.weight_ih_l0.data, \
            self.bi_lstm2.weight_ih_l0_reverse.data, self.bi_lstm3.weight_ih_l0.data, \
            self.bi_lstm3.weight_ih_l0_reverse.data = map(
                self._randn_weights, [self.bi_lstm1.weight_ih_l0.data, self.bi_lstm1.weight_ih_l0_reverse.data,
                                      self.bi_lstm2.weight_ih_l0.data, self.bi_lstm2.weight_ih_l0_reverse.data,
                                      self.bi_lstm3.weight_ih_l0.data, self.bi_lstm3.weight_ih_l0_reverse.data])
        self.bi_lstm1.weight_hh_l0.data, self.bi_lstm1.weight_hh_l0_reverse.data, self.bi_lstm2.weight_hh_l0.data, \
            self.bi_lstm2.weight_hh_l0_reverse.data, self.bi_lstm3.weight_hh_l0.data, \
            self.bi_lstm3.weight_hh_l0_reverse.data = map(
                self._orthogonal_weights, [self.bi_lstm1.weight_hh_l0.data, self.bi_lstm1.weight_hh_l0_reverse.data,
                                           self.bi_lstm2.weight_hh_l0.data, self.bi_lstm2.weight_hh_l0_reverse.data,
                                           self.bi_lstm3.weight_hh_l0.data, self.bi_lstm3.weight_hh_l0_reverse.data])

        self.linear1.weight.data, self.linear2.weight.data, self.linear3.weight.data = map(
            self._randn_weights, [self.linear1.weight.data, self.linear2.weight.data, self.linear3.weight.data])

        self.cnn_1.bias.data, self.cnn_2.bias.data, self.cnn_3.bias.data, self.bi_lstm1.bias_ih_l0.data, \
            self.bi_lstm1.bias_ih_l0_reverse.data, self.bi_lstm2.bias_ih_l0.data, self.bi_lstm2.bias_ih_l0_reverse.data, \
            self.bi_lstm3.bias_ih_l0.data, self.bi_lstm3.bias_ih_l0_reverse.data, self.bi_lstm1.bias_hh_l0.data, \
            self.bi_lstm1.bias_hh_l0_reverse.data, self.bi_lstm2.bias_hh_l0.data, self.bi_lstm2.bias_hh_l0_reverse.data, \
            self.bi_lstm3.bias_hh_l0.data, self.bi_lstm3.bias_hh_l0_reverse.data, self.linear1.bias.data, \
            self.linear2.bias.data, self.linear3.bias.data = map(
                self._biases, [self.cnn_1.bias.data, self.cnn_2.bias.data, self.cnn_3.bias.data, self.bi_lstm1.bias_ih_l0.data,
                               self.bi_lstm1.bias_ih_l0_reverse.data, self.bi_lstm2.bias_ih_l0.data,
                               self.bi_lstm2.bias_ih_l0_reverse.data, self.bi_lstm3.bias_ih_l0.data,
                               self.bi_lstm3.bias_ih_l0_reverse.data, self.bi_lstm1.bias_hh_l0.data,
                               self.bi_lstm1.bias_hh_l0_reverse.data, self.bi_lstm2.bias_hh_l0.data,
                               self.bi_lstm2.bias_hh_l0_reverse.data, self.bi_lstm3.bias_hh_l0.data,
                               self.bi_lstm3.bias_hh_l0_reverse.data, self.linear1.bias.data,
                               self.linear2.bias.data, self.linear3.bias.data])

    @staticmethod
    def _orthogonal_weights(weight_tensor):
        """
        Initialize a weight tensor of random orthogonal weights, as done in the paper's implementation.
        The input tensor is assumed to be square.
        """
        randn_tensor = 0.01 * torch.randn_like(weight_tensor)
        r1, r2, r3, r4 = randn_tensor.split(round(randn_tensor.size(0) / 4))
        (u1, _, _), (u2, _, _), (u3, _, _), (u4, _, _) = map(torch.svd, [r1, r2, r3, r4])
        return torch.cat([u1, u2, u3, u4], dim=0)

    @staticmethod
    def _randn_weights(weight_tensor):
        """
        Initialize a random normal weight tensor, as done in the paper's implementation.
        """
        final_weight_tensor = 0.01 * torch.randn_like(weight_tensor)
        return final_weight_tensor

    @staticmethod
    def _cnn_weights(weight_tensor, filter_length, char_embed_dim):
        """
        Initialize a random uniform tensor for the CNN layers, as done in the paper's implementation.
        """
        final_weight_tensor = (4. / (3 * filter_length * char_embed_dim)) ** 0.5 * torch.rand_like(weight_tensor) - \
            torch.tensor((1. / (3 * filter_length * char_embed_dim)) ** 0.5).type_as(weight_tensor)
        return final_weight_tensor

    @staticmethod
    def _biases(bias_tensor):
        bias_tensor = torch.zeros_like(bias_tensor)
        return bias_tensor

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
        original_shape = char_embedding.shape
        char_embedding = char_embedding.view(original_shape[0] * original_shape[1], 1, original_shape[2],
                                             original_shape[3])
        # cnn_embedding = None
        for cnn_layer in [self.cnn_1, self.cnn_2, self.cnn_3]:
            embed = cnn_layer(char_embedding)
            embed = torch.relu(embed)
            embed = embed.squeeze()
            embed = nn.MaxPool1d(kernel_size=embed.shape[2])(embed)
            embed = embed.view(original_shape[0], original_shape[1], -1)
            list_char_embed.append(embed)
        char_embedding = torch.cat(list_char_embed, dim=2)
        embedding = torch.cat((word_embedding, char_embedding), dim=2)
        embedding = self.dropout_cnn(embedding)
        return embedding

    def bi_lstm_forward(self, embed, sentence_lens):
        bi_lstm_embed = embed.clone().detach().requires_grad_(True)
        for bi_lstm_layer in [self.bi_lstm1, self.bi_lstm2, self.bi_lstm3]:
            # Fourth part - having the word embedding, the pipeline continues as always
            sentence_len, sent_len_sort_ind = torch.tensor(sentence_lens).sort(0, descending=True)
            sentence_len = sentence_len.clone().detach().long()
            sentences = bi_lstm_embed[sent_len_sort_ind]
            new_embeds = torch.nn.utils.rnn.pack_padded_sequence(sentences, sentence_len, batch_first=True)
            lstm_out, _ = bi_lstm_layer(new_embeds)
            lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

            _, original_order = sent_len_sort_ind.sort(0)
            lstm_out = lstm_out[original_order]
            bi_lstm_embed = torch.cat((embed, lstm_out), dim=2)
        return bi_lstm_embed

    @staticmethod
    def gated_attention(lstm_emb, word_lens):
        """
        Gated attention on the input layer, as done in the paper's implementation.
        """
        normed_input_gate = lstm_emb.norm(2, dim=2)
        # We use a mask that indicates where there are words and where we pad.
        mask = torch.tensor([1 if w != 0 else 0 for w in word_lens], device=lstm_emb.device).view(
            lstm_emb.size(0), lstm_emb.size(1))

        v_avg_pool = (lstm_emb * mask.unsqueeze(2)).sum(1) / mask.sum(1).unsqueeze(1)

        v_max_pool, _ = (lstm_emb * mask.unsqueeze(2)).max(1)

        v_gate = (lstm_emb * normed_input_gate.unsqueeze(2) * mask.unsqueeze(2)).sum(1) / (
                normed_input_gate.unsqueeze(2) * mask.unsqueeze(2)).sum(1)

        final_repr = torch.cat([v_avg_pool, v_max_pool, v_gate], dim=1)
        return final_repr

    def feed_forward(self, embed):
        first_embed = self.linear1(embed)
        first_embed = torch.relu(first_embed)
        first_embed = self.dropout1(first_embed)
        second_embed = torch.cat((embed, first_embed), dim=1)
        second_embed = self.linear2(second_embed)
        second_embed = torch.relu(second_embed)
        second_embed = self.dropout2(second_embed)
        out = self.linear3(second_embed)
        return out

    def forward(self, seq_chars1, seq_words1, seq_chars2, seq_words2, len_seq1, len_seq2, len_words1, len_words2):
        embed1 = self.cnn_forward(seq_words1, seq_chars1)
        embed2 = self.cnn_forward(seq_words2, seq_chars2)
        bi_lstm_out1 = self.bi_lstm_forward(embed1, sentence_lens=len_seq1)
        bi_lstm_out2 = self.bi_lstm_forward(embed2, sentence_lens=len_seq2)
        gate_repr1 = self.gated_attention(bi_lstm_out1, len_words1)
        gate_repr2 = self.gated_attention(bi_lstm_out2, len_words2)
        final_repr = torch.cat(
            (gate_repr1, gate_repr2, torch.abs(gate_repr1 - gate_repr2), torch.mul(gate_repr1, gate_repr2)), dim=1)
        really_final_repr = self.feed_forward(final_repr)
        return really_final_repr
