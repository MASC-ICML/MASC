import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from torch import sigmoid
from torch.utils.data import DataLoader
from torch_utils import torch_temporary_seed


class LinearRegressionModel(nn.Module):
    def __init__(self,
                 n_features: int,
                 tau: float = 1,
                 weights_seed: int = None):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(size=(n_features,), dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=True)
        self.tau = tau
        if isinstance(weights_seed, int):
            with torch_temporary_seed(weights_seed):
                nn.init.normal_(self.weights, mean=0, std=1)
                nn.init.normal_(self.bias, mean=0, std=1)
        else:
            nn.init.normal_(self.weights, mean=0, std=1)
            nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, X: torch.Tensor, tau=None) -> torch.Tensor:
        if tau is None:
            tau = self.tau
        return sigmoid(tau * self.product(X))

    def product(self, X: torch.Tensor):
        features = X.type(torch.float)
        return torch.matmul(features, self.weights) + self.bias

    def set_parameter(self, index, value):
        params = list(self.parameters())
        with torch.no_grad():
            params[0][index].copy_(value)

    def get_weights(self):
        return list(self.parameters())[0]

    def get_parameter(self, index):
        return self.get_weights()[index]

    def set_bias(self, value, relative=False):
        if relative:
            value += self.bias.item()
        params = list(self.parameters())
        with torch.no_grad():
            params[1][0].copy_(value)

    def get_bias(self):
        return list(self.parameters())[-1]

    def set_bias_from_threshold(self, t):
        sig_inverse = torch.log(t / (1 - t))
        self.set_bias(sig_inverse, relative=True)

    def w_norm(self):
        return torch.linalg.vector_norm(self.weights)

    def distance_to_decision_boundary(self, X: torch.Tensor):
        """
        :param X: D x N
        :return: U : N
        """
        product = self.product(X)
        w_norm = self.w_norm()
        return - (product / w_norm)

    def init_with_naive_cls(self,
                            dl_train: DataLoader,
                            dl_test: DataLoader):
        # collate all train batches
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        for batch in dl_train:
            X_train.append(batch['X'])
            Y_train.append(batch['y'])
        for batch in dl_test:
            X_test.append(batch['X'])
            Y_test.append(batch['y'])
        Xtrain_np = torch.cat(X_train).detach().cpu().numpy()
        Ytrain_np = torch.cat(Y_train).detach().cpu().numpy()
        Xval_np = torch.cat(X_test).detach().cpu().numpy()
        Yval_np = torch.cat(Y_test).detach().cpu().numpy()

        # Train LogisticRegression model
        log_reg = LogisticRegression(C=0.01) # `C` for the regularizations
        log_reg.fit(Xtrain_np, Ytrain_np)

        # Make predictions on the validation set
        Yval_pred = log_reg.predict(Xval_np)

        # Calculate accuracy
        val_accuracy = accuracy_score(Yval_np, Yval_pred)

        # Extract weights and bias from the LogisticRegression model
        reg_weights = log_reg.coef_  # Shape: (1, n_features)
        reg_bias = log_reg.intercept_[0]  # Shape: (1,)

        for i in range(len(list(self.get_weights()))):
            self.set_parameter(i, reg_weights[0][i])
        self.set_bias(reg_bias)
        
        return val_accuracy

    def init_with_bad_cls(self, Xtrain: torch.Tensor, Ytrain: torch.Tensor, Xval: torch.Tensor, Yval: torch.Tensor):
        Ytrain = torch.where(Ytrain == 1, -1, 1)
        Yval = torch.where(Yval == 1, -1, 1)
        self.init_with_naive_cls(Xtrain, Ytrain, Xval, Yval)
        
