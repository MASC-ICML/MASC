import argparse
import json
import os
import pandas as pd
import torch
import uuid

from datetime import datetime

from dataloader import MSCDataset, create_dataloaders
from trainer import MarketLineSearchBaselineTrainer, StrategicTrainer, NaiveBaselineTrainer, UsingBudgetNaiveBaselineTrainer
from plots import plot_metrics
from model_utils import StrategicHingeLoss
from model import LinearRegressionModel

def run_adult_data(
    # experiment parameters
    job_id=None,
    exp_id='tmp',
    plot=False,
    save_model=False,
    # data parameters
    budget_scale_max=32,
    # optimization parameters
    lr=1e-3,
    ss_tau=1e-3,
    sm_tau=1e-2,
    reg_lambda=0.1,
    # training parameters
    batch_size=500,
    epochs=100,
    early_stopping=None,
    # seeds
    data_seed=0,
    weights_seed=None,
    ):
    
    signature = job_id if job_id else datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(uuid.uuid4())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("datasets/adult/adult-imputed-budget-balanced-new.csv")
    Y = df['target']
    B = df['budget']
    X = df.drop(['target', 'budget'], axis=1)
    
    # Scale budget
    if budget_scale_max: # scale budget to [1, budget_scale_max]
        a, b = 1, budget_scale_max
        B = a + (b - a) * (B - B.min()) / (B.max() - B.min())
    
    ds = MSCDataset(X, Y, B)
    dl_train, dl_val, dl_test = create_dataloaders(ds, batch_size=batch_size, test_ratio=0.2, data_seed=data_seed)

    loss = StrategicHingeLoss()
    
    # Baselines
    naive_model = LinearRegressionModel(n_features=X.shape[1], weights_seed=weights_seed).to(device)
    naive_trainer = NaiveBaselineTrainer(naive_model, loss, device)
    naive_metrics = naive_trainer.fit(dl_train, dl_test)
    
    using_budget_model = LinearRegressionModel(n_features=X.shape[1], weights_seed=weights_seed).to(device)
    using_budget_trainer = UsingBudgetNaiveBaselineTrainer(using_budget_model, loss, device)
    using_budget_val_acc = using_budget_trainer.fit(dl_train, dl_test)

    linesearch_model = LinearRegressionModel(n_features=X.shape[1], weights_seed=weights_seed).to(device)
    linesearch_base = MarketLineSearchBaselineTrainer(linesearch_model, loss, device)
    linesearch_metrics = linesearch_base.fit(dl_train, dl_val, dl_test)
    
    model = LinearRegressionModel(n_features=X.shape[1], weights_seed=weights_seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    if not weights_seed: # if no seed, init with naive model
        model.init_with_naive_cls(dl_train, dl_test)
        
    trainer = StrategicTrainer(model, loss, opt, device,
                               ss_tau=ss_tau, sm_tau=sm_tau, reg_lambda=reg_lambda)
        
    metrics = trainer.fit(dl_train, dl_val, dl_test,
                          num_epochs=epochs,
                          early_stopping=early_stopping)
    
    
    # Extra metrics we need for normalizing welfare and burden per budget scale
    test_batch = next(iter(dl_test))
    _, Y_test, B_test = test_batch['X'], test_batch['y'], test_batch['budget']
    test_total_budget = B_test.sum().item()
    test_pos_total_budget = B_test[Y_test == 1].sum().item()
    train_size = len(dl_train.sampler)
    valid_size = len(dl_val.sampler)
    test_size = len(dl_test.sampler)
    
    metrics.update({
        'lr': lr,
        'ss_tau': ss_tau,
        'sm_tau': sm_tau,
        'batch_size': batch_size,
        'reg_lambda': reg_lambda,
        'epochs': epochs,
        'early_stopping': early_stopping,
        'data_seed': data_seed,
        'budget_scale_max': budget_scale_max,
        'weights_seed': weights_seed,
        'baseline': naive_metrics,
        'linesearch_baseline': linesearch_metrics,
        'using_budget_baseline_acc': using_budget_val_acc,
        'test_total_budget': test_total_budget,
        'test_pos_total_budget': test_pos_total_budget,
        'train_size': train_size,
        'valid_size': valid_size,
        'test_size': test_size,
    })
    
    # Save model and metrics
    exp_dir = os.path.join("experiments",exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    metrics_path = os.path.join(exp_dir, f'{signature}.json')
    json.dump(metrics, open(metrics_path, 'w'))
    if save_model:
        models_dir = os.path.join("experiments", exp_id, "models")
        os.makedirs(models_dir, exist_ok=True)
        models_path = os.path.join(models_dir, f'{signature}.pkl')
        with open(models_path, 'wb') as f:
            torch.save(model.cpu(), f)

    if plot:
        plot_metrics(metrics)
    
    return metrics

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    # experiment parameters
    parser.add_argument('--j', type=str, default=None) # job id
    parser.add_argument('--ex', type=str, default=str(uuid.uuid1())) # experiment id (for dir name)
    parser.add_argument('--plot', action='store_true') # plot metrics when job is done
    parser.add_argument('--save-model', action='store_true') # save model to experiment dir
    
    # data parameters
    parser.add_argument('--bsm', type=int, default=None) # budget scale max
    
    # optimization parameters
    parser.add_argument('--lr', type=float) # learning rate
    parser.add_argument('--ss', type=float) # softsort temperature
    parser.add_argument('--sm', type=float) # softmax temperature
    parser.add_argument('--reg', type=float) # regularization coefficient
    
    # training parameters
    parser.add_argument('--bs', type=int) # batch size
    parser.add_argument('--e', type=int) # number of epochs
    parser.add_argument('--es', type=int, default=None) # early stopping
    
    # seeds
    parser.add_argument('--ds', type=int, default=0) # seed for data split
    parser.add_argument('--ws', type=int, default=None) # seed for model weights initialization
    
    args = parser.parse_args()
    print(args)
    
    run_adult_data(
        # experiment parameters
        job_id=args.j,
        exp_id=args.ex,
        plot=args.plot,
        save_model=args.save_model,
        # data parameters
        budget_scale_max=args.bsm,
        # optimization parameters
        lr=args.lr,
        ss_tau=args.ss,
        sm_tau=args.sm,
        reg_lambda=args.reg,
        # training parameters
        batch_size=args.bs,
        epochs=args.e,
        early_stopping=args.es,
        # seeds
        data_seed=args.ds,
        weights_seed=args.ws,
    )