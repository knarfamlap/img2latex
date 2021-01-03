import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnDecoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size,
                 encoder_dim_out=512, dropout=0.3):
        super(AttnDecoder, self).__init__()

        self.rnn_decoder = nn.LSTMCell(
            input_size=hidden_size + embed_size, hidden_size=hidden_size)
        self.embedding = nn.Embedding(
            num_embeddings=output_size, embedding_dim=embed_size)

        self.init_wh = nn.Linear(
            in_features=encoder_dim_out, out_features=hidden_size)
        self.init_wc = nn.Linear(
            in_features=encoder_dim_out, out_features=hidden_size)
        self.init_wo = nn.Linear(
            in_features=encoder_dim_out, out_features=hidden_size)

        # Attention Mechanism
        self.beta = nn.Parameter(data=torch.Tensor(encoder_dim_out))
        torch.nn.init.uniform(self.beta, -1e-2, 1e2)
        self.W_1 = nn.Linear(in_features=encoder_dim_out,
                             out_features=encoder_dim_out, bias=False)
        self.W_2 = nn.Linear(in_features=hidden_size,
                             out_features=encoder_dim_out, bias=False)

        self.W_3 = nn.Linear(in_features=hidden_size +
                             encoder_dim_out, out_features=hidden_size,
                             bias=False)
        self.W_out = nn.Linear(in_features=hidden_size,
                               out_features=output_size, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, imgs, formulas, epsilos=1.):
        pass

    def init_decoder(self, enc_out):
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_o = self._init_o(mean_enc_out)

        return (h, c), init_o

    def _get_attn(self, enc_out, h_t):
        alpha = torch.tanh(self.W_1(enc_out) + self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta * alpha, dim=1)
        alpha = F.softmax(alpha, dim=-1)

        context = torch.bmm(alpha.unsqueeze(1), enc_out)
        context = context.squeeze(1)

        return context, alpha

    def _init_h(self, mean_enc_out):
        return torch.tanh(self.init_wh(mean_enc_out))

    def _init_c(self, mean_enc_out):
        return torch.tanh(self.init_wc(mean_enc_out))

    def _init_o(self, mean_enc_out):
        return torch.tanh(self.init_wo(mean_enc_out))
