import torch.nn as nn
import torch


class cls_pos(nn.Module):
    def __init__(self):
        super(cls_pos, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])

        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]

        fore_weight = positives * (1.0-pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * ((1.0-pos_label[:, 0, :, :])**4.0) * (pos_pred[:, 0, :, :]**2.0)

        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(pos_label[:, 2, :, :])

        cls_loss = 0.01 * torch.sum(focal_weight*log_loss) / max(1.0, assigned_box)

        return cls_loss


class reg_pos(nn.Module):
    def __init__(self):
        super(reg_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :]*self.smoothl1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
        return reg_loss


class offset_pos(nn.Module):
    def __init__(self):
        super(offset_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_label[:, :2, :, :])
        off_loss = 0.1 * torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
        return off_loss
