import torch
import torch.nn as nn


class SumExpert(nn.Module):
    """求和专家：将输入的三个嵌入向量相加。"""

    def __init__(self):
        super(SumExpert, self).__init__()

    def forward(self, x, y, z):
        # 直接对输入的三个嵌入向量求和
        return x + y + z


class MLPExpert(nn.Module):
    """MLP 专家：使用一个简单的 MLP 将三个嵌入融合。"""

    def __init__(self, embed_size):
        super(MLPExpert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )

    def forward(self, x, y, z):
        # 将三个嵌入向量拼接并输入到 MLP 中
        c = torch.cat([x, y, z], dim=-1)  # 将三个嵌入向量拼接
        c = self.mlp(c)

        return c


class MultiHeadAttentionExpert(nn.Module):
    """Multi-Head Attention 专家：使用多头注意力机制将三个嵌入融合。"""

    def __init__(self, embed_size, num_heads=8):
        super(MultiHeadAttentionExpert, self).__init__()

        in_dim = embed_size * 3
        self.attn = nn.MultiheadAttention(in_dim, num_heads)
        self.fc_out = nn.Linear(in_dim, embed_size)

    def forward(self, x, y, z):
        c = torch.cat([x, y, z], dim=-1)  # 将三个嵌入向量拼接

        out, weights = self.attn(c, c, c)

        # 通过线性层输出
        out = self.fc_out(out)  # 将三个头的输出融合成一个输出

        return out


class GateExpert(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(GateExpert, self).__init__()
        self.txt_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Sigmoid()
        )
        self.vis_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Sigmoid()
        )
        self.id_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Sigmoid()
        )

        self.fusion = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, idf, txt, vis):
        id_values = self.id_gate(idf)
        txt_values = self.txt_gate(txt)
        vis_values = self.vis_gate(vis)

        feat = torch.cat([id_values, txt_values, vis_values], dim=1)
        output = self.fusion(feat)

        return output
