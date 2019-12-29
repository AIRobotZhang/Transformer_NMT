# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class Transformer(nn.Module):
    "Transformer composed of multi-Encoder multi-Decoder and FC Layer."
    def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len, src_embedding_weights, src_pad_id, bos_id, \
                 tgt_embedding_weights, tgt_pad_id, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(Transformer, self).__init__()

        self.bos_id = bos_id
        self.tgt_max_len = tgt_max_len

        self.encoder = Encoder(src_vocab_size, src_max_len, src_embedding_weights, src_pad_id, num_layers, model_dim,
                               num_heads, ffn_dim, dropout, finetuning=True)

        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, tgt_embedding_weights, src_pad_id, tgt_pad_id, num_layers, 
                    model_dim, num_heads, ffn_dim, dropout, finetuning=True)

        self.linear = nn.Linear(model_dim, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_inputs, tgt_inputs):
        # src_inputs: batch_size, seq_len1
        # tgt_inputs: batch_size, seq_len2
        encoder_output, encoder_self_attn = self.encoder(src_inputs)
        output, decoder_self_attn, enc_dec_attn = self.decoder(tgt_inputs, src_inputs, encoder_output)

        output = self.linear(output)
        # output = self.softmax(output)

        return output, encoder_self_attn, decoder_self_attn, enc_dec_attn

    def translate(self, src_inputs):
        # src_inputs: batch_size, seq_len
        encoder_output, encoder_self_attn = self.encoder(src_inputs)

        tgt_inputs = torch.LongTensor([self.bos_id]).expand((src_inputs.size(0), 1))
        pred_word = torch.zeros((src_inputs.size(0), self.tgt_max_len)).long()
        if torch.cuda.is_available():
            tgt_inputs = tgt_inputs.cuda()
            pred_word = pred_word.cuda()
        # for ii in range(self.max_len):
        #     linear_out, hn, cn = self.decoder(hn, cn, nextword_id, output)
        #     # pred: batch_size
        #     _, pred = torch.max(linear_out, 1)
        #     nextword_id = pred
        #     pred_word[ii] = pred
        for ii in range(self.tgt_max_len):
            output, decoder_self_attn, enc_dec_attn = self.decoder(tgt_inputs, src_inputs, encoder_output)
            output = self.linear(output[:, -1, :])
            # output = self.softmax(output)
            _, pred = torch.max(output, 1)
            pred = pred.unsqueeze(-1) # batch_size, 1
            tgt_inputs = torch.cat((tgt_inputs, pred), -1)

        # tgt_inputs: batch_size, seq_len(include <bos> and <eos>)

        return tgt_inputs[:, 1:]


class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, padding_mask=None):
        # self attention
        # print(inputs.dtype)
        context, attention = self.attention(inputs, inputs, inputs, padding_mask)
        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    "Multi Encoder Layer"
    def __init__(self, vocab_size, max_seq_len, embedding_weights, pad_id, num_layers=6, model_dim=512, \
                 num_heads=8, ffn_dim=2048, dropout=0.0, finetuning=True):
        super(Encoder, self).__init__()

        self.pad_id = pad_id
        self.word_embedding = nn.Embedding(vocab_size, model_dim)
        if isinstance(embedding_weights, torch.Tensor):
            self.word_embedding.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout) 
                                                for _ in range(num_layers)])
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs):
        # inputs: batch_size, seq_len
        batch_size, seq_len = inputs.size(0), inputs.size(1)
        word_input = self.word_embedding(inputs)  # batch_size, seq_len, model_dim
        pos_input = self.pos_embedding(batch_size, seq_len) # batch_size, seq_len, model_dim
        # print(word_input.dtype)
        # print(pos_input.dtype)
        output = word_input+pos_input # batch_size, seq_len, model_dim

        self_attention_mask = padding_mask(inputs, inputs, self.pad_id)

        attentions = []
        for encoder in self.encoder_layers:
            # output: batch_size, seq_len, model_dim
            # attention: batch_size, seq_len, seq_len
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.enc_dec_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, decoder_inputs, encoder_outputs, self_attn_mask=None, context_attn_mask=None):
        # decoder_inputs: batch_size, seq_len1, model_dim
        # encoder_inputs: batch_size, seq_len2, model_dim
        decoder_output, self_attention = self.self_attention(decoder_inputs, decoder_inputs, \
                                                            decoder_inputs, self_attn_mask)
        # context attention
        # query is decoder's outputs, key and value are encoder's outputs
        decoder_output, context_attention = self.enc_dec_attention(encoder_outputs, encoder_outputs, \
                                                            decoder_output, context_attn_mask)

        # batch_size, seq_len, model_dim
        decoder_output = self.feed_forward(decoder_output)

        return decoder_output, self_attention, context_attention


class Decoder(nn.Module):
    "Multi Decoder Layer"
    def __init__(self, vocab_size, max_seq_len, embedding_weights, src_pad_id, tgt_pad_id, num_layers=6, \
                  model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0, finetuning=True):
        super(Decoder, self).__init__()

        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.word_embedding = nn.Embedding(vocab_size, model_dim)
        if isinstance(embedding_weights, torch.Tensor):
            self.word_embedding.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, ffn_dim, dropout) 
                                                for _ in range(num_layers)])
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, decoder_inputs, encoder_inputs, encoder_output):
        # decoder_inputs: batch_size, seq_len1
        # encoder_inputs: batch_size, seq_len2
        # encoder_output: batch_size, seq_len2, model_dim
        batch_size, seq_len = decoder_inputs.size(0), decoder_inputs.size(1)
        word_input = self.word_embedding(decoder_inputs)
        pos_input = self.pos_embedding(batch_size, seq_len)
        output = word_input+pos_input # batch_size, seq_len, model_dim

        self_attention_padding_mask = padding_mask(decoder_inputs, decoder_inputs, self.tgt_pad_id)
        seq_mask = sequence_mask(decoder_inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask+seq_mask), 0)

        context_attn_mask = padding_mask(encoder_inputs, decoder_inputs, self.src_pad_id)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            # output: batch_size, seq_len, model_dim
            output, self_attn, context_attn = decoder(output, encoder_output, \
                                            self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


def padding_mask(seq_k, seq_q, pad_id):
    # seq_k: batch_size, seq_len_k
    # seq_q: batch_size, seq_len_q
    seq_len = seq_q.size(1)
    pad_mask = seq_k.eq(pad_id)
    if torch.cuda.is_available():
        pad_mask = pad_mask.cuda()
    pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)  # batch_size, seq_len, seq_len
    # pad_mask2 = pad_mask.unsqueeze(-1).expand(-1, -1, seq_len)  # batch_size, seq_len, seq_len
    # pad_mask = (pad_mask1+pad_mask2)>0
    return pad_mask

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    if torch.cuda.is_available():
        mask = mask.cuda()
    seq_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # batch_size, seq_len, seq_len
    return seq_mask


class PositionalEncoding(nn.Module):
    
    def __init__(self, model_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        
        position_encoding = np.array([
          [pos/np.power(10000, 2.0*(j//2)/model_dim) for j in range(model_dim)]
          for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        position_encoding = torch.from_numpy(position_encoding).float()
        
        self.position_encoding = nn.Embedding(max_seq_len, model_dim)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, batch_size, input_len):
        # input_len: a scalar after padding
        input_pos = torch.LongTensor(list(range(0, input_len)))
        input_pos = input_pos.expand((batch_size, input_len))
        if torch.cuda.is_available():
            input_pos = input_pos.cuda()

        return self.position_encoding(input_pos)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, scaled=None, attn_mask=None):
        # Q, K, V: batch_size, seq_len, hidden_dim
        # scaled: sqrt(hidden_dim)
        # atten_mask: batch_size, seq_len, seq_len
        
        attention = torch.bmm(Q, K.transpose(1, 2)) # batch_size, seq_len, seq_len
        if scaled:
            attention = attention//scaled
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention) # batch_size, seq_len, seq_len
        context = torch.bmm(attention, V) # batch_size, seq_len, hidden_dim
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.head_dim = model_dim//num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(model_dim, self.head_dim*num_heads)
        self.k_linear = nn.Linear(model_dim, self.head_dim*num_heads)
        self.v_linear = nn.Linear(model_dim, self.head_dim*num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(self.head_dim*num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # key=value: batch_size, seq_len1, hidden_dim(word_dim/model_dim)
        # query: batch_size, seq_len2, hidden_dim(word_dim/model_dim)
        residual = query
        batch_size = key.size(0)
        # print(query.dtype)

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        
        # split by heads
        Q = query.view(batch_size*self.num_heads, -1, self.head_dim)
        K = key.view(batch_size*self.num_heads, -1, self.head_dim)
        V = value.view(batch_size*self.num_heads, -1, self.head_dim)
        
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention
        scaled = (K.size(-1))**0.5
        # context: batch_size*num_heads, seq_len, head_dim
        context, attention = self.dot_product_attention(Q, K, V, scaled, attn_mask)
        # concat heads
        context = context.view(batch_size, -1, self.head_dim*self.num_heads)
        # final linear projection
        output = self.linear_final(context) # batch_size, seq_len, model_dim(hidden_dim=word_dim)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual+output)

        return output, attention


class PositionWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, z):
        # z: batch_size, seq_len, model_dim
        output = self.w2(F.relu(self.w1(z)))
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(z+output)

        return output
