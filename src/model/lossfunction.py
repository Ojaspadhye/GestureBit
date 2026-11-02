import torch
import torch.nn as nn

class MSELossFuntion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, valid_mask):
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(-1)
            
        loss = self.mse(preds, targets)
        masked_loss = loss * valid_mask

        return masked_loss.sum() / (valid_mask.sum() * preds.size(-2) * preds.size(-1) + 1e-8)

