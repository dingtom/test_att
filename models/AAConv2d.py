"""
Pytorch implementation of Attention Augmented Convolution.
Developed by: Myeongjun Kim (not us)
Source code: https://github.com/leaderj1001/Attention-Augmented-Conv2d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class AAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                dk, dv, Nh, 
                shape=0, relative=True):
        super(AAConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.shape = shape   # 输出的长宽
        self.dk = dk
        self.dv = dv
        # self.dk = (out_channels*dk)//1000  # ？？？？？？？？？？？？？？？？？？？
        # self.dv = (out_channels*dv)//1000
        self.Nh = Nh
        self.relative = relative  # 是否加入位置编码
        # self.stride = stride        # ????????????  
        # self.padding = (self.kernel_size - 1) // 2

        # assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"         
        # assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"         
        # assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"         
        # assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed." 
        
        # 这里要减去 dv，因为 conv_out 的输出要和 attn_out 的输出合并
        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels-self.dv, self.kernel_size, padding=1)
        # 这个卷积操作的目的就是得到 k, q, v， 注意卷积操作包含了计算 X * W_q, X * W_k, X * W_v 的过程
        self.qkv_conv = nn.Conv2d(self.in_channels, 2*self.dk+self.dv, kernel_size=1)
         # attention 的结果仍要作为特征层传入卷积层进行特征提取
        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1)

        if self.relative:  # 每个位置的w, h相对位置编码的可学习参数量均为 2 * [w or h] - 1
            self.key_rel_w = nn.Parameter(torch.randn((2*self.shape-1, dk//Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2*self.shape-1, dk//Nh), requires_grad=True))

    def forward(self, x):
        """
        attention augmented conv 的 “主函数”
        :param x: 输入数据，形状为 (batch_size, in_channels, height, width)
        :return: 最终输出，形状为 (batch, out_channels, height, width)
        """
        batch, _, height, width = x.size()
        # conv_out -> (batch_size, out_channels-dv, height, width)
        conv_out = self.conv_out(x)

        # flat_q, flat_k, flat_v -> (batch_size, Nh, dvh/dkh, height*width)
        # dvh = dv/Nh, dkh = dk/Nh
        # q, k, v -> (batch_size, Nh, dv/dk, height, width)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        # logits  => (batch_size, Nh, height*width, height*width)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)    # (B, Nh, height*width, height*width) (B, Nh, height*width, height*width)
            logits += h_rel_logits
            logits += w_rel_logits
        # weights (batch_size, Nh, height*width, height*width)
        weights = F.softmax(logits, dim=-1)  
        # (batch_size, Nh, height*width, dvh/dkh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv//self.Nh, height, width))
        # (batch, dv, height, width)
        attn_out = self.combine_heads_2d(attn_out) 
        # (batch, dv, height, width)
        attn_out = self.attn_out(attn_out)
        # (batch_size, out_channels, height, width)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        """
        计算 q, k, v 以及每个 head 的 q, k, v
        :param x: 输入数据，形状为 (batch_size, in_channels, height, width)
        :param dk: q, k 的维度
        :param dv: v 的维度
        :param Nh: 有多少个 head
        :return: flat_q, flat_k, flat_v, q, k, v
        """

        N, _, height, width = x.size()
        # 利用卷积操作求 q, k, v, 包含了计算 X * W_q, X * W_k, X * W_v 的过程
        qkv = self.qkv_conv(x)
        # 将卷积输出按 channel 划分为 q, k, v  
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)  # 每个小块中的大小按照list中的大小决定，其中list中的数字总和应等于该维度的大小
        # 将single head 改为 multi-head  # (batch, Nh, channels//Nh, height, width)
        q = self.split_heads_2d(q, Nh)  
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk//Nh
        q *= dkh**-0.5
        # 得到每个 head 的 q, k, v
        flat_q = torch.reshape(q, (N, Nh, dk//Nh, height*width))
        flat_k = torch.reshape(k, (N, Nh, dk//Nh, height*width))
        flat_v = torch.reshape(v, (N, Nh, dv//Nh, height*width))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        """
        划分 head
        :param x: q or k or v
        :param Nh: head 的数量，必须要能整除 q, k, v 的 channel 维度数
        :return: reshape 后的 q, k, v
        """
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels//Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        """
        将所有 head 的输出组合到一起
        :param x: 包含所有 head 的输出
        :return: 组合后的输出
        """
        batch, Nh, dv, height, width = x.size()
        ret_shape = (batch, Nh*dv, height, width)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        """
        计算相对位置编码
        :param q: q
        :return: h 和 w 的位置编码
        """
        B, Nh, dk, height, width = q.size()
        # q -> (B, Nh, height, width, dk)
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        # 分别计算 w 与 h 的一维编码
        key_rel_w = nn.Parameter(torch.randn((2*width-1, dk), requires_grad=True)).to(device)
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, height, width, Nh, "w")

        key_rel_h = nn.Parameter(torch.randn((2*height-1, dk), requires_grad=True)).to(device)
        # # q -> (B, Nh, width, height, dk)
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), key_rel_h, width, height, Nh, "h")

        return rel_logits_h, rel_logits_w  # (B, Nh, height*width, height*width) (B, Nh, height*width, height*width)

    def relative_logits_1d(self, q, rel_k, height, width, Nh, case):
        """
        计算一维位置编码
        :param q: q，维度为(B, Nh, height, width, dk)
        :param rel_k: 位置编码的可学习参数，形状为为 (2*[w or h]-1, dk)
        :param height: 输入特征高度
        :param width: 输入特征宽度
        :param Nh: head 数量
        :param case: 区分 w 还是 h 的位置编码
        :return: 位置编码，形状为 (B, Nh, height*width, height*width)
        """
        # 使用爱因斯坦求和约定，实现批量矩阵乘法  [B, Nh, height, width, dk] [2*width-1, dk]
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)  # (B, Nh, height, width, 2*width-1)
        # 因为是一维位置编码 (w or h)，所以另一个维度用不上
        rel_logits = torch.reshape(rel_logits, (-1, Nh*height, width, 2*width-1))    # (B, Nh*height, width, 2*width-1)
        # 加入位置信息
        # 下面的操作都是为了最后能产生形状为 (B, Nh, height*width, height*width) 的输出，以便于与 logit 相加
        rel_logits = self.rel_to_abs(rel_logits)  # (B, Nh*height, width, width)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, height, width, width))  # (B, Nh, height, width, width)
        rel_logits = torch.unsqueeze(rel_logits, dim=3)  # (B, Nh, height, 1, width, width)
        rel_logits = rel_logits.repeat((1, 1, 1, height, 1, 1))  # (B, Nh, height, height, width, width)

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)  # (B, Nh, height, width, height, width)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, height*width, height*width))
        return rel_logits

    def rel_to_abs(self, x):
        """
        相对 to 绝对，在位置编码中加入绝对位置信息
        :param x: 原始位置编码，形状为 (B, Nh*height, width, 2*width-1)
        :return: 位置编码，形状为 (B, Nh*height, width, width)
        """
        B, Nh, L, _ = x.size()
        # '0' 即绝对位置信息，此后所有操作都是为了让同一 [行 or 列] 的每个点的位置编码的 '0' 出现的位置不同
        # 在最后一个维度的末尾，即每隔 2L - 1 的位置加入 0，
        # 这就是为什么 key_rel_[w or h]，即可学习参数有 2 * [w or h] - 1 个
        col_pad = torch.zeros((B, Nh, L, 1)).to(device)
        x = torch.cat((x, col_pad), dim=3)
        # 每个 head 加入 L - 1 个 0， 为了让每一 [行 or 列] 的 '0' 错位
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        # 将 (L * 2 * L) + (L - 1) 个编码元素重新组织，使其形状为为 (L + 1, 2 * L - 1)
        # 目的是让 '0' 错位，这样每一 [行 or 列] 的点的位置编码中 '0' 出现的位置不一样
        # 相当于嵌入了绝对位置信息
        final_x = torch.reshape(flat_x_padded, (B, Nh, L+1, 2*L-1))
        # reshape 以便于后续操作
        final_x = final_x[:, :, :L, L - 1:]
        return final_x



