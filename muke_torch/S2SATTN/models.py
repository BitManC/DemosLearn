import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttenDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_len=MAX_LENGTH):
        super(AttenDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_len = max_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 这里怎样定义 attention的维度，得到的系数矩阵对应的是那些部分的输入输出
        self.attn = nn.Linear(self.hidden_size * 2, self.max_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # decoder_input 为翻译语言的对应句子, decoder_hidden 为 encoder 最后一层的 hidden, encoder_outputs 此时的前 7为是 encoder  output 的值
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # embeddend cat with hidden equals to decoder_input cat with hidden
        # 第 i 个 input hiddent 与 第 i 个 decoder input 的attention, 间接表示了 decoder input 与 encoder input 的权值关系
        # 这里的 attention 为linear, 需要学习的是 w * x 中的 w
        atten_weight = F.softmax(self.attn(torch.cat([embedded[0], hidden[0]], 1)), dim=1)
        att_applied = torch.bmm(atten_weight.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat([embedded[0], att_applied[0]], dim=1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)


        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, atten_weight

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# if __name__ == "__main__":
#     encoder_net = EncoderRNN(5000, 256)
#     decoder_net = DecoderRNN(256, 5000)
#     atten_decoder_net = AttenDecoderRNN(256, 5000)
#
#     tensor_in = torch.tensor([S2SATTN, 14, 16, 18], dtype=torch.long).view(-1, 1)
#     hidden_in = torch.zeros(1, 1, 256)
#
#     encoder_out, encoder_hidden = encoder_net(tensor_in[0], hidden_in)
#
#     print(encoder_out)
#     print(encoder_hidden)
#
#     tensor_in = torch.tensor([[100]])
#     hidden_in = torch.zeros(1, 1, 256)
#     encoder_out = torch.zeros(10, 256)
#
#     out1, out2, out3 = atten_decoder_net(tensor_in, hidden_in, encoder_out)
#     print(out1, out2, out3)
#
#     out1, out2 = decoder_net(tensor_in, hidden_in)
#     print(out1, out2)

class BiLSTM_Attention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_hidden, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)  # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X)  # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        hidden_state = torch.tensor(
            torch.zeros(1 * 2, len(X), self.n_hidden))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.tensor(
            torch.zeros(1 * 2, len(X), self.n_hidden))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]

        # 这里进行最后的输出加权
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention  # model : [batch_size, num_classes], attention : [batch_size, n_step]


