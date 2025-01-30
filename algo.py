import torch

from model import LinearRegressionModel


# Get optimal price for X and classifier
def calc_optimal_p(X: torch.Tensor, B: torch.Tensor, model: LinearRegressionModel):
    with torch.no_grad():
        u = torch.maximum(model.distance_to_decision_boundary(X), torch.zeros_like(B))
        # positive = [i for i in range(len(B)) if u[i] == 0]
        positive = torch.where(u == 0, 1, 0)
        bpu = torch.where(u > 0, B / u, torch.zeros_like(B))
        sorted_order = torch.argsort(bpu, descending=True)
        max_revenue = 0
        opt_p = torch.inf
        units_sum = 0
        revenues = torch.zeros_like(B)
        if u.sum() == 0:  # no one needs to move
            return 0, 0, [], revenues
        for i in range(len(B)):
            units_sum += u[sorted_order[i]]
            p = bpu[sorted_order[i]]
            revenue = p * units_sum
            if revenue > max_revenue:
                max_revenue = revenue
                opt_p = bpu[sorted_order[i]]
            revenues[sorted_order[i]] = revenue
        move = torch.where(bpu >= opt_p, 1, 0)
        return opt_p.item(), max_revenue.item(), move, positive, revenues
