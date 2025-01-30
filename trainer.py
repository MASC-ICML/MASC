import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Any, Callable, Dict, Literal, Optional
from collections import defaultdict
from tqdm import tqdm

from model import LinearRegressionModel
from model_utils import calc_burden, calc_welfare, l2_reg, SoftSort


class StrategicTrainer:
    def __init__(self,
                 model: LinearRegressionModel,
                 loss_fn: nn.Module,
                 optimizer: Optimizer,
                 device: Optional[torch.device] = None,
                 ss_tau=0.1,
                 sm_tau=2,
                 reg_lambda=1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.ss_tau = ss_tau  # ->0 for hard
        self.sm_tau = sm_tau  # ->inf for hard
        self.reg_lambda = reg_lambda
        self.metrics = defaultdict(list)
        
        if self.device:
            self.model.to(self.device)
    """
    returns u_positive, upb_positive, bpu_full
    """

    def calc_upb(self, X, B):
        u = self.model.distance_to_decision_boundary(X).type(torch.float64)
        pos_indices = (u > 0).nonzero(as_tuple=True)[0]
        return u[pos_indices], torch.div(u[pos_indices], B[pos_indices]), torch.div(B, u).squeeze()

    """
    returns bpu_sorted, revenue 
    """

    @staticmethod
    def calc_revenue(sort, upb, u_pos):
        if len(upb) == 0:
            return torch.Tensor([]), torch.Tensor([0])
        # normalize for sorting
        upb_min = torch.min(upb, dim=0)[0]
        upb = torch.div(upb, upb_min)
        PI = torch.squeeze(sort.forward(-upb.unsqueeze(dim=0)))  # sort increasing
        # de-normalize
        upb = torch.mul(upb, upb_min)
        upb_sorted = torch.matmul(PI, upb)
        bpu_sorted = 1 / upb_sorted
        C = torch.cumsum(torch.matmul(PI, u_pos.unsqueeze(dim=1)),
                         dim=0)
        revenue = bpu_sorted * C.squeeze()
        return bpu_sorted, revenue

    """
    returns p_hat
    """

    def calc_p(self, upb, u_pos):
        if len(upb) == 1:
            return 1 / upb
        if len(upb) == 0:
            inf = torch.Tensor([torch.inf])
            if self.device:
                inf = inf.to(self.device)
            return inf
        # normalize
        ss = SoftSort(tau=self.ss_tau, hard=False)
        bpu_sorted, revenue = self.calc_revenue(ss, upb, u_pos)
        if len(bpu_sorted) == 0:
            return torch.Tensor([0])
        q = torch.softmax(self.sm_tau * revenue, dim=0)
        return torch.dot(q, bpu_sorted)

    """
    returns p_opt
    """

    def calc_p_hard(self, upb, u_pos):
        if len(upb) == 1:
            return 1 / upb
        if len(upb) == 0:
            inf = torch.Tensor([torch.inf])
            if self.device:
                inf = inf.to(self.device)
            return inf

        sort = SoftSort(tau=7, hard=True)
        bpu_sorted, revenue = self.calc_revenue(sort, upb, u_pos)
        q = torch.argmax(revenue)
        return bpu_sorted[q]

    @staticmethod
    def calc_buyers_percentages(Y: torch.Tensor, bpu: torch.Tensor, p):
        """
        Calculate the precentage of the following groups, both for true and false labels:
        - 
        """
        y = Y.squeeze(0)
        assert y.shape[0] == bpu.shape[0], f"y shape: {y.shape}, bpu shape: {bpu.shape}"
        
        y_positive = (y == 1).type(torch.float)
        y_negative = (y != 1).type(torch.float)
        
        y_true_positive = y_positive * (bpu <= 0)
        y_true_negative = y_negative * (bpu > 0)
        y_false_positive = y_negative * (bpu <= 0)
        y_false_negative = y_positive * (bpu > 0)
        
        y_tn_moved = y_true_negative * (bpu >= p)
        y_tn_stayed = y_true_negative * (bpu < p)
        y_fn_moved = y_false_negative * (bpu >= p)
        y_fn_stayed = y_false_negative * (bpu < p)

        total_pos = y_positive.sum().item()
        total_neg = y_negative.sum().item()
        total_tp = y_true_positive.sum().item()
        total_fp = y_false_positive.sum().item()
        total_tn_moved = y_tn_moved.sum().item()
        total_tn_stayed = y_tn_stayed.sum().item()
        total_fn_moved = y_fn_moved.sum().item()
        total_fn_stayed = y_fn_stayed.sum().item()
        
        return {
            'tp': total_tp / total_pos,
            'fp': total_fp / total_neg,
            'tn_moved': total_tn_moved / total_neg,
            'tn_stayed': total_tn_stayed / total_neg,
            'fn_moved': total_fn_moved / total_pos,
            'fn_stayed': total_fn_stayed / total_pos
        }
    
    def _update_metrics(self, train_result, prefix=Literal['train', 'val', 'test']):
        assert prefix in ['train', 'val', 'test'], "Metric's prefix must be train or val"
        
        for k, v in train_result.items():
            self.metrics[f"{prefix}_{k}"].append(v)
    
    def _update_weights_metrics(self):
        for i, w in enumerate(self.model.get_weights()):
            self.metrics["w_" + str(i)].append(w.item())
        self.metrics["w_bias"].append(self.model.get_bias().item())
    
    def fit(self,
            dl_train: DataLoader,
            dl_val: DataLoader,
            dl_test: DataLoader,
            num_epochs: int,
            early_stopping: Optional[int] = None):
        
        best_val_acc = 0
        epochs_without_improvement = 0
        actual_num_epochs = 0
        
        # Record step 0
        train_result = self._foreach_batch(dl_train, self.test_batch) # running dl_train with test_batch for no grads
        train_result['reg'] = self.reg_lambda * l2_reg(self.model).item()
        self._update_metrics(train_result, prefix="train")
        self._update_weights_metrics()
        val_result = self._foreach_batch(dl_val, self.test_batch)
        self._update_metrics(val_result, prefix="val")
        test_result = self._foreach_batch(dl_test, self.test_batch)
        self._update_metrics(test_result, prefix="test")

        for _ in tqdm(range(num_epochs)):
            actual_num_epochs += 1
            
            # Train
            train_result = self._foreach_batch(dl_train, self.train_batch)
            self._update_metrics(train_result, prefix="train")
            
            # Record weights
            self._update_weights_metrics()
            
            # Validation
            val_result = self._foreach_batch(dl_val, self.test_batch)
            self._update_metrics(val_result, prefix="val")
            
            # Test
            test_result = self._foreach_batch(dl_test, self.test_batch)
            self._update_metrics(test_result, prefix="test")
            
            # Early stopping
            val_acc = val_result["acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    break
            
        self.metrics['actual_num_epochs'] = actual_num_epochs
        return self.metrics
    
    def _foreach_batch(self, dl: DataLoader,
                       forward_fn: Callable[[Any], Dict[str, Any]]):
        all_batches_metrics = defaultdict(list)
        epoch_results = defaultdict()
        num_samples = len(dl.sampler)
        
        for batch in dl:
            batch_result = forward_fn(batch)
            for k, v in batch_result.items():
                all_batches_metrics[k].append(v)
        
        # Aggregate batch results
        for k, v in all_batches_metrics.items():
            if k == 'num_correct':
                epoch_results["acc"] = np.sum(v) / num_samples
            else:
                epoch_results[k] = np.mean(v)
        return epoch_results
    
    def train_batch(self, batch):
        batch_result = defaultdict()
        X, Y, B = batch['X'], batch['y'], batch['budget']
        if self.device:
            X, Y, B = X.to(self.device), Y.to(self.device), B.to(self.device)
        
        # Forward pass
        u_pos, upb_pos, bpu = self.calc_upb(X, B)
        p_hat = self.calc_p(upb_pos, u_pos)

        # Compute loss
        loss = self.loss_fn(self.model, X, B, Y, p_hat)
        reg = self.reg_lambda * l2_reg(self.model)
        objective = loss + reg

        # Backward pass and parameter update
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        # Classify and calculate number of correct predictions
        pos_cls = torch.logical_or(bpu <= 0, bpu >= p_hat)
        Y_pred = torch.where(pos_cls > 0, 1, -1)
        momements = self.calc_buyers_percentages(Y, bpu, p_hat)
        num_correct = torch.sum(Y_pred == Y)
        
        batch_result['loss'] = loss.item()
        batch_result['reg'] = reg.item()
        batch_result['num_correct'] = num_correct.item()
        batch_result['price'] = p_hat.item()
        batch_result.update(momements)
        
        return batch_result
    
    def test_batch(self, batch):
        batch_result = defaultdict()
        X, Y, B = batch['X'], batch['y'], batch['budget']
        if self.device:
            X, Y, B = X.to(self.device), Y.to(self.device), B.to(self.device)

        with torch.no_grad():
            # forward
            u_pos, upb_pos, bpu = self.calc_upb(X, B)
            p_opt = self.calc_p_hard(upb_pos, u_pos)
            loss = self.loss_fn(self.model, X, B, Y, p_opt)
            
            # classify
            pos_cls = torch.logical_or(bpu <= 0, bpu >= p_opt)
            movements = self.calc_buyers_percentages(Y, bpu, p_opt)
            Y_pred = torch.where(pos_cls > 0, 1, -1)
            num_correct = torch.sum(Y_pred == Y)
            
            # Calculate welfare and burden
            u = self.model.distance_to_decision_boundary(X).type(torch.float64)
            welfare = calc_welfare(u, B, p_opt)
            burden = calc_burden(u, B, p_opt, Y)
            
            batch_result['loss'] = loss.item()
            batch_result['num_correct'] = num_correct.item()
            batch_result['price'] = p_opt.item()
            batch_result.update(movements)
            batch_result['welfare'] = welfare.item()
            batch_result['burden'] = burden.item()
        
        return batch_result
     
            
class NaiveBaselineTrainer(StrategicTrainer):
    def __init__(self,
                 model: LinearRegressionModel,
                 loss_fn: nn.Module,
                 device: Optional[torch.device] = None):
        super().__init__(model, loss_fn, None, device)

    def fit(self,
            dl_train: DataLoader,
            dl_test: DataLoader):
        
        # Train naively
        self.metrics = defaultdict()
        naive_test_acc = self.model.init_with_naive_cls(dl_train, dl_test)
        self.metrics['naive_test_acc'] = naive_test_acc

        # Test strategically
        with torch.no_grad():
            batch = next(iter(dl_test))
            X_test, Y_test, B_test = batch['X'], batch['y'], batch['budget']
            if self.device:
                X_test, Y_test, B_test = X_test.to(self.device), Y_test.to(self.device), B_test.to(self.device)
            u_pos, upb_pos, bpu = self.calc_upb(X_test, B_test)
            p_opt = self.calc_p_hard(upb_pos, u_pos)
            pos_cls = torch.logical_or(bpu <= 0, bpu >= p_opt)
            Y_pred = torch.where(pos_cls > 0, 1, -1)
            movements = self.calc_buyers_percentages(Y_test, bpu, p_opt)
            
            self.metrics["test_acc"] = 1 - torch.ne(Y_pred, Y_test).type(torch.float).mean().item()
            self.metrics["test_loss"] = self.loss_fn(self.model, X_test, B_test, Y_test, p_opt).item()
            self.metrics["test_ps"] = p_opt.item()
            self.metrics.update(movements)
            
            u = self.model.distance_to_decision_boundary(X_test).type(torch.float64)
            self.metrics['test_welfare'] = calc_welfare(u, B_test, p_opt).item()
            self.metrics['test_burden'] = calc_burden(u, B_test, p_opt, Y_test).item()
            
            for i, w in enumerate(self.model.get_weights()):
                self.metrics["w_" + str(i)] = w.item()
            self.metrics["w_bias"] = self.model.get_bias().item()

        return self.metrics


class UsingBudgetNaiveBaselineTrainer(StrategicTrainer):
    def __init__(self,
                 model: LinearRegressionModel,
                 loss_fn: nn.Module,
                 device: Optional[torch.device] = None):
        super().__init__(model, loss_fn, None, device)

    def _wrap_dataloader(self, dl: DataLoader):
        for batch in dl:
            X, Y, B = batch['X'], batch['y'], batch['budget']
            if self.device:
                X, Y, B = X.to(self.device), Y.to(self.device), B.to(self.device)
            X_with_B = torch.cat((X, B.unsqueeze(1)), dim=1)
            yield {'X': X_with_B, 'y': Y}
    
    def fit(self,
            dl_train: DataLoader,
            dl_test: DataLoader):
        dl_train_new = self._wrap_dataloader(dl_train)
        dl_test_new = self._wrap_dataloader(dl_test)
        test_accuracy = self.model.init_with_naive_cls(dl_train_new, dl_test_new)
        return test_accuracy


class MarketLineSearchBaselineTrainer(StrategicTrainer):
    def __init__(self,
                 model: LinearRegressionModel,
                 loss_fn: nn.Module,
                 device: Optional[torch.device] = None):
        super().__init__(model, loss_fn, None, device)
    def fit(self,
            dl_train: DataLoader,
            dl_val: DataLoader,
            dl_test: DataLoader,
            epochs: int = 100):
        
        # Train naive model
        self.model.init_with_naive_cls(dl_train, dl_test)
        
        # Create held-out set
        val_batch = next(iter(dl_val))
        X_val, Y_val, B_val = val_batch['X'], val_batch['y'], val_batch['budget']
        if self.device:
            X_val, Y_val, B_val = X_val.to(self.device), Y_val.to(self.device), B_val.to(self.device)
        
        # Calculate model induced price on the validation set
        u_pos_val, upb_pos_val, _ = self.calc_upb(X_val, B_val)
        price = self.calc_p_hard(upb_pos_val, u_pos_val)
        
        # Search for the best bias using train set
        min_bias, max_bias = -3, 3
        bias_vals = np.linspace(min_bias, max_bias, epochs)
        best_bias = min_bias
        best_loss = torch.inf

        for current_bias in tqdm(bias_vals, desc="Market Line Search"):
            train_batch = next(iter(dl_train))
            X, Y, B = train_batch['X'], train_batch['y'], train_batch['budget']
            if self.device:
                X, Y, B = X.to(self.device), Y.to(self.device), B.to(self.device)
            self.model.set_bias(current_bias)
            # Calculate loss with the induced price from before
            loss = self.loss_fn(self.model, X, B, Y, price)
            if loss < best_loss:
                best_loss = loss
                best_bias = current_bias

        # Test
        test_batch = next(iter(dl_test))
        X_test, Y_test, B_test = test_batch['X'], test_batch['y'], test_batch['budget']
        if self.device:
            X_test, Y_test, B_test = X_test.to(self.device), Y_test.to(self.device), B_test.to(self.device)
        self.model.set_bias(best_bias)
        u_pos_test, ups_pos_test, bpu_test = self.calc_upb(X_test, B_test)
        p_test = self.calc_p_hard(ups_pos_test, u_pos_test)
        pos_cls = torch.logical_or(bpu_test <= 0, bpu_test >= price)
        Y_pred = torch.where(pos_cls > 0, 1, -1)
        test_acc = 1 - torch.ne(Y_pred, Y_test).type(torch.float).mean()
        
        metrics = {
            'best_bias': best_bias,
            'min_bias': min_bias,
            'max_bias': max_bias,
            'test_acc': test_acc.item(),
            'price': price.item(),
        }
        
        return metrics

