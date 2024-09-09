import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seq_cha = configs.seq_cha
        self.enc_in = configs.enc_in
        self.patch_len = configs.patch_len
        self.patch_pred_len = configs.patch_pred_len
        self.d_model = configs.d_model
        self.patch_num = patch_num = self.seq_len // self.patch_len

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.linear_x1 = nn.Linear(self.seq_len, self.seq_cha)
        self.linear_x2 = nn.Linear(self.enc_in, self.pred_len)
        self.linear_x3 = nn.Linear(self.seq_cha, self.enc_in)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=True
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_pred_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.dropout = nn.Dropout(configs.dropout)

        self.linear_patch_re = nn.Linear(self.d_model * 2, self.patch_pred_len)

        self.point_conv = nn.Conv1d(self.seq_cha, self.enc_in, kernel_size=1, stride=1)
        self.point_activation = nn.ReLU()
        self.point_norm = nn.BatchNorm1d(self.enc_in)
        self.dec_point_conv = nn.Conv1d(self.enc_in, self.seq_cha, kernel_size=1, stride=1)

        self.depth_conv = nn.Conv1d(patch_num, patch_num, kernel_size=3, stride=1, groups=patch_num, padding=1)
        self.depth_activation = nn.ReLU()
        self.depth_norm = nn.BatchNorm1d(patch_num)

    def forward(self, x, x_mark=None, y_true=None, y_mark=None):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x_rest2 = x
        x_rest2 = self.linear_x3(x_rest2)
        x_rest2 = x_rest2.permute(0, 2, 1)

        x = x.permute(0, 2, 1)
        x = self.point_conv(x)
        x = self.point_activation(x)
        x = self.point_norm(x)
        x = x + x_rest2
        x_rest = x
        x_rest = self.linear_x1(x_rest)
        x_rest = x_rest.permute(0, 2, 1)
        x_rest = self.linear_x2(x_rest)
        x_rest = x_rest.permute(0, 2, 1)

        x = x.permute(0, 2, 1)

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_pred_len
        W = self.patch_len
        d = self.d_model

        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)

        xw_res = xw
        xw = self.depth_conv(xw)
        xw = self.depth_activation(xw)
        xw = self.depth_norm(xw) + xw_res

        xd = self.linear_patch(xw)
        enc_in = self.relu(xd)

        enc_out = self.lstm(enc_in)[1]
        enc_out0 = enc_out[0].repeat(1, 1, M).view(2, -1, self.d_model)
        enc_out1 = enc_out[1].repeat(1, 1, M).view(2, -1, self.d_model)
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B * C, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(B, M, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        dec_out = self.lstm(dec_in, (enc_out0, enc_out1))[0]

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)
        y = yw.reshape(B, C, -1).permute(0, 2, 1)

        y = y.permute(0, 2, 1)
        y = self.dec_point_conv(y)
        y = y.permute(0, 2, 1)
        y = y + x_rest
        y = y + seq_last

        return y
