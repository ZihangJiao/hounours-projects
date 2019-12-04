import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class DCTModel(nn.Module):
    """Baseline DCT model

    Takes three words as input and outputs
    the flatten DCT feature prediction (3 rotation DOF).

    input shape: [batch_size, 3]
    output shape: [batch_size, out_dim]

    This is designed following Yu's model
    """

    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 out_dim,
                 vocab_size,
                 pretrain_weight=None):
        super(DCTModel, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.vocab_size = vocab_size

        if pretrain_weight is None:
            self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        else:
            # use pretrained weight
            self.embedding = nn.Embedding.from_pretrained(pretrain_weight)
            # fine-tune here
            self.embedding.weight.requires_grad = True

        self.linear1 = nn.Linear(embed_dim * 3, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.out_dim)
        self.apply(GlorotUniform_init_fn)

    def forward(self, x):
        out = self.embedding(x).view(x.shape[0], -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out


def GlorotUniform_init_fn(layer):
    # do xavier/glorot initialization on all linear weights
    if type(layer) == nn.Linear:
        # the gain can be modified here
        torch.nn.init.xavier_uniform_(layer.weight, gain=2.**0.5)
        layer.bias.data.fill_(0.)


class Seq2SeqModel(nn.Module):
    """Our proposed Seq2Seq model

    Takes sentence as input to predicts motion trajectory
    in format of quaternion

    input shape: [batch_size, max_sentence_length]
    output shape: [batch_size, max_trajectory_length, 4] (trajectories)

    It consists of encoder and decoder, both can be called separately
    """

    def __init__(self,
                 embed_dim,
                 vocab_size,
                 dof,
                 enc_dim,
                 dec_dim,
                 enc_layers=1,
                 dec_layers=1,
                 bidirectional=True,
                 dropout_prob=0.25,
                 pretrain_weight=None,
                 teacher_forcing_ratio=0.5):

        super(Seq2SeqModel, self).__init__()
        self.encoder = LSTMEncoder(embed_dim=embed_dim,
                                   vocab_size=vocab_size,
                                   enc_dim=enc_dim,
                                   enc_layers=enc_layers,
                                   bidirectional=bidirectional,
                                   dropout_prob=dropout_prob,
                                   pretrain_weight=pretrain_weight).to('cuda')


        out_dim_enc = 2 * enc_dim if bidirectional else enc_dim
        self.decoder = LSTMDecoder(dof=dof,
                                   dec_dim=dec_dim,
                                   out_dim_enc=out_dim_enc,
                                   dec_layers=dec_layers,
                                   dropout_prob=dropout_prob,
                                   teacher_forcing_ratio=teacher_forcing_ratio).to('cuda')

    def forward(self, in_seq, tgt_seq, lengths):

        unpacked_out_enc, enc_mask = self.encoder(in_seq, lengths)
        dec_outputs, attn_weights = self.decoder(tgt_seq, unpacked_out_enc,
                                                 enc_mask)
        return dec_outputs, attn_weights


class LSTMEncoder(nn.Module):
    """Encoder part of Seq2Seq model

    It's essentially a embedding + Uni/Bi-LSTM
    """
    def __init__(self,
                 embed_dim,
                 vocab_size,
                 enc_dim,
                 enc_layers=1,
                 bidirectional=True,
                 dropout_prob=0.25,
                 pretrain_weight=None):

        super(LSTMEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.enc_layers = enc_layers
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob

        if pretrain_weight is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        else:
            # use pretrained weight
            self.embedding = nn.Embedding.from_pretrained(pretrain_weight)
            # no fine-tune
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.enc_dim,
            num_layers=self.enc_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.enc_layers > 1 else 0.,
            bidirectional=self.bidirectional)

    def forward(self, in_seq, lengths):
        # used to mask zeroing-out padded positions in decoder
        enc_mask = in_seq.eq(0)

        #### Embedding part
        embeddings = self.embedding(in_seq.cuda())  # [batch, max_seq_len, embed_dim]
        embeddings = F.dropout(embeddings,
                               p=self.dropout_prob,
                               training=self.training)

        #### LSTM part
        # initialization of hidden/cell states using Orthogonal Matrix
        num_states = self.enc_layers * (2 if self.bidirectional else 1)
        h0 = torch.empty(num_states, len(embeddings), self.enc_dim).cuda()
        c0 = torch.empty(num_states, len(embeddings), self.enc_dim).cuda()
        nn.init.orthogonal_(h0)
        nn.init.orthogonal_(c0)

        # use 'pack' to accelerate LSTM mini-batch training
        packed_embeddings = pack_padded_sequence(embeddings,
                                                 lengths,
                                                 batch_first=True)

        packed_out_enc, (hidd_stat_enc,
                         cell_stat_enc) = self.lstm(packed_embeddings,
                                                    (h0, c0))

        unpacked_out_enc, lengths_enc = pad_packed_sequence(packed_out_enc,
                                                            batch_first=True)
        unpacked_out_enc = F.dropout(unpacked_out_enc,
                                     p=self.dropout_prob,
                                     training=self.training)

        # unpacked_out_enc shape [batch_size, max_seq_len, enc_dim]
        # enc_mask shape [batch_size, max_seq_len]
        return unpacked_out_enc, enc_mask


class AttentionLayer(nn.Module):
    """Defines the attention layer class.

    Uses Luong's global attention with the general scoring function.
    And scoring method here is 'general'
    """

    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()

        self.src_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.context_plus_hidden_projection = \
            nn.Linear(input_dim + output_dim, output_dim, bias=False)

    def forward(self, tgt_input, encoder_out, src_mask):
        # tgt_input shape [batch_size, dec_dim]
        # encoder_out shape [batch_size, max_seq_len_enc, enc_dim]
        # src_mask shape [batch_size, max_seq_len_enc]

        # Computes attention scores
        attn_scores = self.score(tgt_input, encoder_out)

        # apply a mask to the attention scores and
        # calculate attention weight for each encoder time step
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(dim=1)
            attn_scores.masked_fill_(src_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # get 'average' output of encoder using attention weight
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)

        # pass it through a linear layer to get attentional hidden state
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)
        attn_out = torch.tanh(
            self.context_plus_hidden_projection(context_plus_hidden))

        return attn_out, attn_weights.squeeze(dim=1)

    def score(self, tgt_input, encoder_out):
        # 'general' scoring: $h_t^{T} W h_{enc}$
        projected_encoder_out = \
            self.src_projection(encoder_out).transpose(2, 1)
        attn_scores = \
            torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)

        return attn_scores


class LSTMDecoder(nn.Module):
    """Decoder part of Seq2Seq model

    Costomized Unidirectional LSTM, and an attention layer
    linking it with encoder.
    """
    def __init__(self,
                 dof,
                 embed_dim=32,
                 dec_dim=64,
                 out_dim_enc=64,
                 dec_layers=1,
                 dropout_prob=0.25,
                 teacher_forcing_ratio=0.5):
        super(LSTMDecoder, self).__init__()

        self.dof = dof
        self.embed_dim = embed_dim
        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dropout_prob = dropout_prob
        self.out_dim_enc = out_dim_enc
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.lookback = 4

        self.lstm = nn.ModuleList([
            nn.LSTMCell(input_size= \
                                self.dec_dim + self.lookback * self.dof
                                if layer == 0 else self.dec_dim,
                        hidden_size=self.dec_dim)
            for layer in range(self.dec_layers)
        ]).cuda()

        self.attn = AttentionLayer(self.out_dim_enc, self.dec_dim)
        self.final_projection = nn.Linear(self.dec_dim, self.dof)

    def forward(self, tgt_seq, unpacked_out_enc, enc_mask):
        # tgt_seq shape [batch_size, max_seq_len_dec, lookback_len, num_dof]
        # unpacked_out_enc shape [batch_size, max_seq_len_enc, out_dim_enc]
        # enc_mask shape [batch_size, max_seq_len_end]

        assert self.dof == tgt_seq.shape[-1]
        batch_size, time_step_dec = tgt_seq.shape[0], tgt_seq.shape[1]

        # flatten the last two dimensions
        tgt_seq = tgt_seq.view(batch_size, time_step_dec, -1)

        # determine traing method
        use_teacher_forcing = True if (
            torch.rand(1).item() < self.teacher_forcing_ratio) else False

        #### Initialization part
        # creat the lstm input feed and hidden/cell states of (t = -1) step
        input_feed = tgt_seq.data.new(batch_size, self.dec_dim).zero_()
        hidd_stat_dec = [
            torch.zeros(batch_size, self.dec_dim).cuda()
            for i in range(self.dec_layers)
        ]

        cell_stat_dec = [
            torch.zeros(batch_size, self.dec_dim).cuda()
            for i in range(self.dec_layers)
        ]

        # Orthogonal initialization
        nn.init.orthogonal_(input_feed)
        for i in range(self.dec_layers):
            nn.init.orthogonal_(hidd_stat_dec[i])
            nn.init.orthogonal_(cell_stat_dec[i])

        # initialize attention and output node
        attn_weights = tgt_seq.data.new(batch_size, time_step_dec,
                                        enc_mask.shape[1]).zero_()
        dec_outputs = tgt_seq.data.new(batch_size, time_step_dec,
                                       self.dof).zero_()

        #### LSTM part
        for t in range(time_step_dec):
            # input_feed = input_feed.detach()
            if use_teacher_forcing or t < self.lookback:
                # concatenate correct previous motion and
                # attentional hidden states of the last time step
                lstm_input = torch.cat([tgt_seq[:, t, :].cuda(), input_feed.cuda()], dim=-1).cuda()
            else:
                previous_out = dec_outputs[:, t - self.lookback:t, :].detach().cuda()
                previous_out = previous_out.view(batch_size, -1).cuda()
                lstm_input = torch.cat([previous_out, input_feed], dim=-1)

            # Pass tgt_seq input through each layer(s) and
            # update hidden/cell states

            for idx, layer in enumerate(self.lstm):
                layer = layer.cuda()
                hidd_stat_dec[idx], cell_stat_dec[idx] = layer(
                    lstm_input, (hidd_stat_dec[idx], cell_stat_dec[idx]))

                # add dropout layer after each lstm layer
                lstm_input = F.dropout(hidd_stat_dec[idx],
                                       p=self.dropout_prob,
                                       training=self.training)

            #### Attention part, apply attention to lstm output
            input_feed, step_attn_weight = self.attn(hidd_stat_dec[-1],
                                                     unpacked_out_enc,
                                                     enc_mask.cuda())

            # get the final output of attention layer of one time step
            input_feed = F.dropout(input_feed,
                                   p=self.dropout_prob,
                                   training=self.training)

            # final projection
            step_output = self.final_projection(input_feed)

            attn_weights[:, t, :] = step_attn_weight
            dec_outputs[:, t, :] = step_output


        dof_123 = torch.tanh(dec_outputs[:, :, :-1])
        dof_4 = torch.relu(dec_outputs[:, :, -1]).unsqueeze(-1)

        outputs = torch.cat([dof_123, dof_4], dim=-1)

        # output shape [batch_size, max_seq_len_dec, num_dof]
        # attn_weights [batch_size, max_seq_len_dec, max_seq_len_enc]
        return outputs, attn_weights

    def set_teacher_forcing_ratio(self, ratio: float):
        assert 0 <= ratio <= 1
        self.teacher_forcing_ratio = ratio


class nopad_mse_loss(nn.Module):
    """Customized mse loss function for Seq2Seq model

    Same as torch.nn.functional.mse_loss, it has two reduction modes,
    'mean' & 'sum'.  The input is network output and target with the
    same shape and the 'seq_length' tensor which records actual length
    of each sample in batch so that the loss on PAD can be neglected.

    'keep_dof' is used during evaluation to show loss on each dimension.
    This result can not be used for backpropagation.
    """
    def __init__(self, reduction='mean'):
        super(nopad_mse_loss, self).__init__()

        if reduction == 'mean':
            self.calc_fn = torch.mean
        elif reduction == 'sum':
            self.calc_fn = torch.sum
        else:
            raise Exception('reduction type can only be "mean" or "sum"')


    def forward(self, output, target, seq_length, keep_dof=False):
        # output/target shape [batch_size, max_seq_len_dec, num_dof]
        # seq_length shape [batch_size]

        assert len(output) == len(target) == len(seq_length)

        error_list = []
        for i, length in enumerate(seq_length):
            length = length.item()
            err = output[i, :length] - target[i, :length]
            err = self.calc_fn(torch.pow(err, 2), dim=0)
            error_list.append(err)
        loss = self.calc_fn(torch.stack(error_list), dim=0)

        # loss shape [1] or [num_dof](keep_dof)
        return loss if keep_dof else self.calc_fn(loss)
