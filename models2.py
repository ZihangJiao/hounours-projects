import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import weight_init_uniform
import numpy as np
import math

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
                 dropout_prob=0,
                 pretrain_weight=None,
                 look_back = 4):

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
                                   dropout_prob=dropout_prob)
        self.dof = dof
        self.lookback = look_back
        self.enc_dim =enc_dim
        self.enc_layers = enc_layers
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        if pretrain_weight is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        else:
            # use pretrained weight
            self.embedding = nn.Embedding.from_pretrained(pretrain_weight)
            # no fine-tune
            self.embedding.weight.requires_grad = True

    def forward(self, in_seq, tgt_seq, word_time_distribution):
    # print(temp_input.size())
                # print(temp_input.size())
                # input()


        # motion_array = word_time_distribution[0]
        # new_motion_array= []
        #
        # for time_step in motion_array:
        #     if(time_step - int(time_step) >= 0.5):
        #         new_motion_array.append(int(time_step) + 1)
        #     else:
        #         new_motion_array.append(int(time_step))
        #
        # the_sum = math.fsum(motion_array)
        # if(the_sum - int(the_sum) >= 0.5):
        #     the_sum = int(the_sum) + 1
        # else:
        #     the_sum = int(the_sum)
        # new_motion_array[-1] += int(abs(the_sum - math.fsum(new_motion_array)))
        # motion_array = new_motion_array

        print(word_time_distribution)
        previous_outputs = torch.rand(1,4,4)
        # random initilise the previous output
        begining_word = in_seq[:,:3]
        # get the first three word of input sentense, which is the
        # start-of-sentense symbol, the first word amd the second word
        embeddings = self.embedding(in_seq.cuda())
        num_states = self.enc_layers * (2 if self.bidirectional else 1)
        hidd_state = torch.empty(num_states, len(embeddings), self.enc_dim).cuda()
        cell_state = torch.empty(num_states, len(embeddings), self.enc_dim).cuda()
        nn.init.orthogonal_(hidd_state)
        nn.init.orthogonal_(cell_state)
        # hidd_state = self._init_state(hidd_state)
        # cell_state = self._init_state(cell_state)
        # print(hidd_state.size())
        #initilise hidden state and cell state of lstm


        # print(hidd_state.size())
        unpacked_out_enc, encoder_h, encoder_c = self.encoder(begining_word, self.embedding, hidd_state, cell_state)
        # print(unpacked_out_enc)
        # print(encoder_h)
        # print(encoder_c)
        begining_word_outputs, hidd, cell = self.decoder(tgt_seq, unpacked_out_enc,
                                                 encoder_h, encoder_c,
                                                 time_step = motion_array[0],
                                                 previous_out = previous_outputs)

        dec_outputs = begining_word_outputs
        previous_outputs = dec_outputs[:,-self.lookback:,:]
        hidd_state = hidd
        cell_state = cell
        # print(previous_outputs)

        for i in range(1, len(motion_array)-1):
            word = in_seq[:,i-1:i+2]
            # print(word)
            # input()
            unpacked_out_enc, encoder_h, encoder_c = self.encoder(word, self.embedding, hidd_state, cell_state)
            word_outputs , hidd, cell = self.decoder(tgt_seq, unpacked_out_enc,
                                                     encoder_h, encoder_c,
                                                     time_step = motion_array[i],
                                                     previous_out = previous_outputs)

            dec_outputs = torch.cat((dec_outputs, word_outputs), 1)
            previous_outputs = dec_outputs[:,-self.lookback:,:]
            hidd_state = hidd
            cell_state = cell


        end_word = in_seq[:,-3:]
        # print(end_word)
        # input()
        unpacked_out_enc, encoder_h, encoder_c = self.encoder(begining_word, self.embedding, hidd_state, cell_state)
        end_word_outputs, hidd, cell = self.decoder(tgt_seq, unpacked_out_enc,
                                                 encoder_h, encoder_c,
                                                 time_step = motion_array[-1],
                                                 previous_out = previous_outputs)
        dec_outputs = torch.cat((dec_outputs, end_word_outputs), 1)
        previous_outputs = dec_outputs[:,-self.lookback:,:]
        hidd_state = hidd
        cell_state = cell

        return dec_outputs


    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
        # print(dec_outputs.size())
        # print(dec_outputs[:,-1,:].size())
        # input()



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
                 dropout_prob=0,
                 pretrain_weight=None):

        super(LSTMEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.enc_layers = enc_layers
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob

        # if pretrain_weight is None:
        #     self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        # else:
        #     # use pretrained weight
        #     self.embedding = nn.Embedding.from_pretrained(pretrain_weight)
        #     # no fine-tune
        #     self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.enc_dim ,
            num_layers=self.enc_layers,

            batch_first=True,
            dropout=self.dropout_prob if self.enc_layers > 1 else 0.,
            bidirectional=self.bidirectional)

    # def forward(self, in_seq, hidden_state, cell_state):
    def forward(self, in_seq, embedding, hidden_state, cell_state):
        # print(self.embed_dim)
        # used to mask zeroing-out padded positions in decoder
        # print(in_seq)
        # enc_mask = in_seq.eq(0)
        # print(enc_mask)
        # input()
        #### Embedding part
        # print(in_seq)



        embeddings = embedding(in_seq.cuda())  # [batch, max_seq_len, embed_dim]
        # print(embeddings.size())
        # input()
        embeddings = F.dropout(embeddings,
                               p=self.dropout_prob,
                               training=self.training)

        #### LSTM part

        # print(h0.size())
        # use 'pack' to accelerate LSTM mini-batch training
        # packed_embeddings = pack_padded_sequence(embeddings,
        #                                          lengths,
        #                                          batch_first=True)
        # del embeddings

        # packed_out_enc, (hidd_stat_enc,
        #                  cell_stat_enc) = self.lstm(packed_embeddings,
        #                                             (h0, c0))
        # print(embeddings.size())
        # print(self.embed_dim)
        # print(hidden_state)
        # input()
        # embeddings = embedding

        if(hidden_state.size()[0] == 1 and self.bidirectional == True):
            hidden_state = torch.cat((hidden_state[:,:,:self.enc_dim], hidden_state[:,:,self.enc_dim:]), 0)
            cell_state = torch.cat((cell_state[:,:,:self.enc_dim], cell_state[:,:,self.enc_dim:]), 0)
        # print(cell_state.size())
        # print(self.enc_layers)
        output, (hidden, cell) = self.lstm(embeddings, (hidden_state, cell_state))
        # print(hidden)
        # input()
        # del h0
        # del c0
        # del packed_embeddings

        # unpacked_out_enc, lengths_enc = pad_packed_sequence(packed_out_enc,
        #                                                     batch_first=True)
        #
        # del packed_out_enc
        #
        # unpacked_out_enc = F.dropout(unpacked_out_enc,
        #                              p=self.dropout_prob,
        #                              training=self.training)
        # unpacked_out_enc = output

        # print(unpacked_out_enc)
        # unpacked_out_enc shape [batch_size, max_seq_len, enc_dim]
        # enc_mask shape [batch_size, max_seq_len]
        # print(unpacked_out_enc)
        return output, hidden, cell


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
        # print(projected_encoder_out.size())
        attn_scores = \
            torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)
        # print(projected_encoder_out)
        # print(attn_scores)

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
                 dropout_prob=0,
                 bidirectional_encoder=True):
        super(LSTMDecoder, self).__init__()

        self.dof = dof
        self.embed_dim = embed_dim
        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.dropout_prob = dropout_prob
        self.out_dim_enc = out_dim_enc
        self.bidirectional_encoder = bidirectional_encoder
        self.lookback = 4

        self.lstm = nn.ModuleList([
            nn.LSTMCell(input_size=  self.dof * self.lookback + self.dec_dim*2 if bidirectional_encoder else self.dof * self.lookback + self.dec_dim,
                                # self.dec_dim*2 + self.lookback * self.dof if bidirectional_encoder else self.dof * self.lookback + self.dec_dim,
                                # if layer == 0 else self.dec_dim,
                        hidden_size=self.dec_dim*2 if bidirectional_encoder else self.dec_dim)
            for layer in range(self.dec_layers)
        ]).cuda()

        self.attn = AttentionLayer(self.out_dim_enc, self.dec_dim*2 if bidirectional_encoder else self.dec_dim)
        self.final_projection = nn.Linear(self.dec_dim*2 if bidirectional_encoder else self.dec_dim, self.dof)

    def forward(self,tgt_seq,unpacked_out_enc,hidd_stat_enc,cell_stat_enc,time_step,previous_out):
        device = torch.device('cuda')
        assert self.dof == tgt_seq.shape[-1]
        batch_size, time_step_dec = tgt_seq.shape[0], tgt_seq.shape[1]

        tgt_seq = tgt_seq.view(batch_size, time_step_dec, -1)
        input_feed = tgt_seq.data.new(batch_size, self.dec_dim*2 if self.bidirectional_encoder else self.dec_dim).zero_().cuda()

        hidd_stat_dec = self._init_state(hidd_stat_enc)
        cell_stat_dec = self._init_state(cell_stat_enc)

        nn.init.orthogonal_(input_feed)
        for i in range(self.dec_layers):
            nn.init.orthogonal_(hidd_stat_dec[i])
            nn.init.orthogonal_(cell_stat_dec[i])
        # print(previous_out.size())
        dec_outputs = tgt_seq.data.new(batch_size, 1,
                                       self.dof).zero_()
        # print(dec_outputs)

        decoder_input = unpacked_out_enc[:,0].unsqueeze(1)
        previous_out = previous_out.contiguous().view(batch_size,-1)
        temp_input = previous_out

        # print(temp_input)
        # input()

        # print(temp_input.device)
        # print(temp_input.size())
        # for i in range(batch_size):
        #     for k in range(self.lookback):
        #         temp_input[i][k* self.dof] = 1
        # print("previous_out")
        # print(previous_out)

        for t in range(time_step):
            if(t == 0):
                if self.bidirectional_encoder == True:
                    temp_input = previous_out.to(device)
                    # print(hidd_stat_enc.size())
                    # print(temp_input.size())
                    temp_input = torch.cat([temp_input, hidd_stat_enc[0]],dim = 1)
                    temp_input = torch.cat([temp_input, hidd_stat_enc[1]],dim = 1)
                else:
                    temp_input = previous_out.to(device)
                    temp_input = torch.cat([temp_input, hidd_stat_enc[0]],dim = 1)

            else:

                previous_out = dec_outputs[:, t-1, :].detach().cuda()
                previous_out = previous_out.view(batch_size, -1).cuda()
                # print(previous_out)
                # print(previous_out.size())
                # if(t == 80):
                #     print(lstm_input)

                temp_input = temp_input.to(device)
                tgt_seqtemp = tgt_seq
                tgt_seqtemp = tgt_seqtemp.to(device)
                # if(t != 0):
                #     if torch.rand(1).item() < 0.1:
                #     # print(torch.rand(1).item())
                #         temp_input = torch.cat([temp_input[:,4:], tgt_seqtemp[:,t-1,:]],dim = -1)
                #     else:
                #
                # print(temp_input.size())
                # input()
                # print(temp_input[:,:16].size())
                # print(temp_input[:,:12])
                temp_input = torch.cat([temp_input[:,self.dof:self.dof * self.lookback], previous_out],dim = 1)
                # print(temp_input.size())
                # print(temp_input)
                # print(temp_input.size())
                # print(hidd_stat_dec.size())
                if self.bidirectional_encoder == True:
                    temp_input = torch.cat([temp_input, hidd_stat_enc[0]],dim = 1)
                    temp_input = torch.cat([temp_input, hidd_stat_enc[1]],dim = 1)
                else:
                    temp_input = torch.cat([temp_input, hidd_stat_enc[0]],dim = 1)

            # temp_input = torch.cat()
            # print(temp_input.size())
            # input()
            # print(torch.rand(1).item())
            # print("cated_temp")
            # print(temp_input)
            # input()
            del previous_out
            # print(temp_input.size())
            # print(input_feed)
            # print(input_feed.size())

            lstm_input = temp_input
            # print(lstm_input)
            # print(lstm_input.size())
            # lstm_input = torch.cat([temp_input, input_feed], dim=-1).cuda()

            # if(t == 80):
            #     print(temp_input)
            #     print(lstm_input)

            del input_feed
            # print(lstm_input.size())

            # mean = lstm_input.mean()
            # std = lstm_input.std()
            # lstm_input = (lstm_input - mean)/std

            # Pass tgt_seq input through each layer(s) and
            # update hidden/cell states
            # print(hidd_stat_dec[0].size())

            hidd_stat_dec_temp = torch.zeros(self.dec_layers,batch_size,self.dec_dim * 2 if self.bidirectional_encoder else self.dec_dim).cuda()
            cell_stat_dec_temp = torch.zeros(self.dec_layers,batch_size,self.dec_dim * 2 if self.bidirectional_encoder else self.dec_dim).cuda()

            for idx, layer in enumerate(self.lstm):
                # print(layer)
                layer = layer.cuda()
                # print(lstm_input.size())
                # print(lstm_input.size())
                # print("input")
                # print(lstm_input.size())
                # print(lstm_input.size())
                a,b = layer(
                    lstm_input, (hidd_stat_dec[idx], cell_stat_dec[idx]))
                # print("a:")
                # print(a.size())
                # hidd_stat_dec[idx] = a
                # cell_stat_dec[idx] = b
                # del a
                # del b
                # add dropout layer after each lstm layer
                lstm_input = F.dropout(a,
                                       p=self.dropout_prob,
                                       training=self.training)
                # print(lstm_input.size())
                hidd_stat_dec_temp[idx] = a
                cell_stat_dec_temp[idx] = b
                del a
                del b
            #### Attention part, apply attention to lstm output
            # print(hidd_stat_dec[-1].size())
            # print(unpacked_out_enc.size())
            # print(enc_mask.size())
            # if(t == 20):
            #     print(hidd_stat_dec_temp)
            #     print(hidd_stat_dec)
            hidd_stat_dec = hidd_stat_dec_temp
            cell_stat_dec = cell_stat_dec_temp
            del hidd_stat_dec_temp
            del cell_stat_dec_temp
            # print(hidd_stat_dec)
            # input_feed, step_attn_weight = self.attn(hidd_stat_dec[-1],
            #                                          unpacked_out_enc,
            #                                          enc_mask.cuda())
            # print(input_feed)
            # print(hidd_stat_dec[0])
            input_feed = hidd_stat_dec[-1]
            # step_attn_weight = tgt_seq.data.new(batch_size, 1,
            #                                 enc_mask.shape[1]).zero_()
            # print(step_attn_weight)
            # get the final output of attention layer of one time step
            input_feed = F.dropout(input_feed,
                                   p=self.dropout_prob,
                                   training=self.training).cuda()
            # final projection
            # print(input_feed.size())
            step_output = self.final_projection(input_feed)
            # step_output = input_feed
            # print('TEST')
            # print(input_feed)
            # print(step_output)
            # input()

            # if t != 1:
            #     attn_weights = torch.cat((attn_weights, tgt_seq.data.new(batch_size, 1, enc_mask.shape[1]).zero_()),1)
            # attn_weights[:, t, :] = step_attn_weight
            # del step_attn_weight


            # step_output = step_output * std + mean

            # if t == 0:
            #     dec_outputs = torch.cat((dec_outputs, tgt_seq.data.new(batch_size, 1, self.dof).zero_()),1)
            #     dec_outputs[:, t, :] = temp_input[:,self.dof * (self.lookback -1):self.dof * self.lookback]
            # elif t == 1:
            #     dec_outputs[:, t, :] = step_output
            # else:
            #     dec_outputs = torch.cat((dec_outputs, tgt_seq.data.new(batch_size, 1, self.dof).zero_()),1)
            #     dec_outputs[:, t, :] = step_output
            # # print(step_output)
            # # input()
            # del step_output
            # step_output = step_output * std + mean

            if t != 1:
                dec_outputs = torch.cat((dec_outputs, tgt_seq.data.new(batch_size, 1, self.dof).zero_()),1)
            dec_outputs[:, t, :] = step_output
            # print(step_output)
            del step_output

            # print(step_output.size())
            # for i in step_output:
            #     counter = 0
            #     print(i[0].cpu().numpy()[0])
            #     x = torch.tensor(0.0051).cuda()
            #     print(x)
            #     print(torch.equal(i[0],x))
            #     if i == np.array(1.0,1.0,1.0,1.0):
            #         lengths[counter] = t
            #     counter += 1



        outputs = dec_outputs
        # print(outputs[:,0,:])
        # print(outputs)
        # input()

        # return outputs
        # print(hidd_stat_dec)
        return outputs, hidd_stat_dec, cell_stat_dec

    # def set_teacher_forcing_ratio(self, ratio: float):
    #     assert 0 <= ratio <= 1
    #     self.teacher_forcing_ratio = ratio
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


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
        # print(output)
        # print(target)
        # print(seq_length)
        assert len(output) == len(target) == len(seq_length)

        error_list = []
        for i, length in enumerate(seq_length):
            # length = length.item()
            # print(output[i,:length])
            err = output[i, :length] - target[i, :length]
            # print(len(output[0]))
            # print(len(target[0]))
            err = self.calc_fn(torch.pow(err, 2), dim=0)
            # print(err)
            error_list.append(err)
        loss = self.calc_fn(torch.stack(error_list), dim=0)
        # print(output)
        # print(target)
        # print(loss)

        # loss shape [1] or [num_dof](keep_dof)
        return loss if keep_dof else self.calc_fn(loss)
