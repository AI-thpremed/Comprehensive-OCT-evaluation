import os
import sys
import json
import torch
import numpy as np


class LstmFusion(torch.nn.Module):

    def __init__(self, input_dim, output_dim=256):
        super(LstmFusion, self).__init__()
        print("cv fusion: lstm attention...", flush=True)

        input_hidden_dim = 512   # 词向量维度
        self.lstm_hidden = 256  # lstm 隐层维度
        attention_hidden = 256  # attention 隐层维度
        self.attention_num = 32  # attention 观点数
        fc1_dim = 1024    # 全连接维度(第1层)

        self.input_linear = torch.nn.Linear(
            in_features=input_dim, out_features=input_hidden_dim)

        self.lstm = torch.nn.LSTM(input_size=input_hidden_dim, hidden_size=self.lstm_hidden,
                                  num_layers=1, batch_first=True, bidirectional=True)

        self.linear_W_s1 = torch.nn.Linear(
            in_features=2 * self.lstm_hidden, out_features=attention_hidden)
        self.linear_W_s2 = torch.nn.Linear(
            in_features=attention_hidden, out_features=self.attention_num)
        self.linear_fc = torch.nn.Linear(
            in_features=2 * self.lstm_hidden * self.attention_num, out_features=fc1_dim)
        self.linear_output = torch.nn.Linear(
            in_features=fc1_dim, out_features=output_dim)
        self.output_dim = output_dim

    def get_output_dim(self):
        return self.output_dim

    def forward(self, ftr_input):
        """ftr_input shape: batch_size * max_frame * feature_size """

        embedding_input = self.input_linear(ftr_input)
        # selfattention encode
        bilstm_output, _ = self.lstm(
            embedding_input)  # batch * max_frame * 512
        H_s1 = torch.tanh(self.linear_W_s1(bilstm_output)
                          )  # batch * max_frame * 256
        H_s2 = self.linear_W_s2(H_s1)  # batch * max_frame * 32
        A = torch.nn.functional.softmax(input=H_s2, dim=1)
        A_trans = A.transpose(1, 2)  # batch * 32 * max_frame
        M = torch.matmul(A_trans, bilstm_output)  # batch * 32 * 512

        M_flat = torch.reshape(
            input=M, shape=[-1, 2 * self.lstm_hidden * self.attention_num])  # batch * (32*512)
        fc = self.linear_fc(M_flat)
        fc = torch.nn.functional.leaky_relu(input=fc, negative_slope=0.1)
        feat = self.linear_output(fc)
        return feat
