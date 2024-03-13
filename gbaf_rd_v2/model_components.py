import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import AttentionLayer
from utils import get_layers



################################ Encoder ##############################################
class Encoder(nn.Module):
    def __init__(self, input_size, m, attention_size, num_layers, num_head, dropout, output_size=1):
        super(Encoder, self).__init__()
        self.Num_layers = num_layers
        self.dim_hidden = int(attention_size/4)
        self.m = m
        self.output_size = output_size

        self.fe = FeatureExtracture(input_size, attention_size, scale=3)
        self.attention_layers = get_layers(AttentionLayer(attention_size, num_head, dropout), num_layers)
        self.norm = nn.LayerNorm(attention_size, eps=1e-5)

        self.linear1 = nn.Linear(attention_size, self.dim_hidden)
        self.linear2 = nn.Linear(self.dim_hidden, self.output_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask, position_embedding):
        x = x.float()
        x = self.fe(x)
        x = position_embedding(x)
        for layer in self.attention_layers:
            x = layer(x, mask)
        x = self.norm(x)

        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, m, attention_size, num_layers, num_head, dropout, dropout_output):
        super(Decoder, self).__init__()
        self.Num_layers = num_layers
        self.m = m

        self.fe = FeatureExtracture(input_size, attention_size, scale=3)
        self.attention_layers = get_layers(AttentionLayer(attention_size, num_head, dropout), num_layers)
        self.norm = nn.LayerNorm(attention_size, eps=1e-5)

        self.dropout_output = nn.Dropout(dropout_output)
        self.linear = nn.Linear(attention_size, 2 ** self.m)


    def forward(self, x, mask, position_embedding):
        x = x.float()
        x = self.fe(x)
        x = position_embedding(x)
        for layer in self.attention_layers:
            x = layer(x, mask)
        x = self.norm(x)
        x = self.dropout_output(x)
        x = self.linear(x)
        output = F.softmax(x, dim=-1)
        return output


class FeatureExtracture(nn.Module):
    def __init__(self, input_size, output_size, scale=3):
        super(FeatureExtracture,self).__init__()

        self.FC1 = nn.Linear(input_size, output_size * scale, bias=True)
        self.activation1 = F.relu
        self.FC2 = nn.Linear(output_size * scale, output_size * scale, bias=True)
        self.activation2 = F.relu
        self.FC3 = nn.Linear(output_size * scale, output_size, bias=True)

    def forward(self, x):
        x = self.FC1(x)
        x = self.FC2(self.activation1(x))
        x = self.FC3(self.activation2(x))

        return x








