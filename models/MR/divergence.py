import torch
import torch.nn as nn
from torch.functional import F


class JSDivergence(nn.Module):

    def __init__(self):
        super(JSDivergence, self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction="batchmean")

        return 0.5 * loss
