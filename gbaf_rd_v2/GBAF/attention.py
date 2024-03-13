import math
import torch
import torch.nn as nn
import torch.nn.functional as F


############################ Attention layer ##################################################################
class AttentionLayer(nn.Module):
    def __init__(self, input_size, num_head, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.hidden_layer = 4 * input_size

        self.norm1 = nn.LayerNorm(input_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(input_size, eps=1e-5)

        # self attention
        self.self_attn = MultiHeadAttention(num_head, input_size, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        # feedforward net
        self.linear1 = nn.Linear(input_size, self.hidden_layer)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.hidden_layer, input_size)
        self.activation = F.relu

        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_att = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_att, x_att, x_att, attn_mask=mask))

        x_ff = self.norm2(x)
        x_ff = self.dropout2(self.activation(self.linear1(x_ff)))
        x_ff = self.linear2(x_ff)

        x = x + self.dropout3(x_ff)
        return x


############################ Multi Head Attention ##################################################################
def attention(q, k, v, d_k, attn_mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if attn_mask is not None:
        mask = attn_mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    # pdb.set_trace()
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.input_size = input_size
        self.key_dim = input_size // heads
        self.h = heads

        self.q_linear = nn.Linear(input_size, input_size, bias=False)
        self.v_linear = nn.Linear(input_size, input_size, bias=False)
        self.k_linear = nn.Linear(input_size, input_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.FC = nn.Linear(input_size, input_size)

    def forward(self, q, k, v, attn_mask=None, decoding=0):
        bs = q.size(0)
        # perform linear operation and split into N heads
        q = self.q_linear(q).view(bs, -1, self.h, self.key_dim)
        k = self.k_linear(k).view(bs, -1, self.h, self.key_dim)
        v = self.v_linear(v).view(bs, -1, self.h, self.key_dim)

        # transpose to get dimensions bs * heads * sequenceLen * input_size/heads
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next

        scores = attention(q, k, v, self.key_dim, attn_mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.input_size)
        output = self.FC(concat)

        return output
