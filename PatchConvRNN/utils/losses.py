import torch.nn as nn
import torch
import torch.nn.functional as F


class STD_loss(nn.Module):
    def __init__(self, l1_weight=0.3, l2_weight=0.3, std_weight=0.4):
        super(STD_loss, self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.std_weight = std_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, y_pred, y_trues):
        std_x = torch.std(y_pred, dim=1)
        std_y = torch.std(y_trues, dim=1)

        std_l1 = self.l1_loss(std_x, std_y)
        std_l2 = self.l2_loss(std_x, std_y)
        # std_diff = torch.abs(std_x - std_y)
        loss1 = F.l1_loss(y_pred, y_trues)
        loss2 = F.mse_loss(y_pred, y_trues)
        loss3 = std_l1
        loss4 = std_l2
        loss5 = F.l1_loss(torch.mean(y_pred, dim=1), torch.mean(y_trues, dim=1))
        loss6 = F.mse_loss(torch.mean(y_pred, dim=1), torch.mean(y_trues, dim=1))
        return loss1, loss2, loss3, loss4, loss5, loss6
