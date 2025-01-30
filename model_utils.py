from model import LinearRegressionModel
import torch

class StrategicHingeLoss(torch.nn.Module):
    def __init__(self):
        super(StrategicHingeLoss, self).__init__()

    @staticmethod
    def forward(model: LinearRegressionModel, X, B, Y, p_hat):
        hinge = torch.ones_like(Y) - Y * (
                model.product(X) + torch.div(B, p_hat) * model.w_norm())
        return torch.clamp_min(hinge, 0).mean()


def l2_reg(model: LinearRegressionModel):
    return torch.pow(model.w_norm(), 2)

class SoftSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores: torch.Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat

def calc_welfare(u, B, p):
    bpu_full = torch.div(B, u).squeeze()
    pos_cls = torch.logical_or(bpu_full <= 0, bpu_full >= p)
    moved = (bpu_full >= p).float() # 1 = moved, 0 = not moved
    cost = moved * (B - p*u) # cost = 0 for non-movers
    welfare = torch.sum(B * pos_cls - cost) # Welfare = sum_i {b_i * 1{yhat_i=1} – c(x_i,x’_i)}
    
    return welfare

def calc_burden(u, B, p, Y):
    bpu_full = torch.div(B, u).squeeze()
    neg_pred = torch.logical_and(bpu_full > 0, bpu_full < p)
    false_negatives = torch.logical_and(neg_pred, Y == 1)
    burden = torch.sum(u[false_negatives] * p)
    
    return burden
