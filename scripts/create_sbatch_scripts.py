import itertools

# SLRUM params
available_nodes: list[str] = ...
partition_name: str = ...
account_name: str = ...

# script params
script_path = "scripts/run_adult_script.sh"
batches_path = "scripts/run_adult_batches.sh"

# HP const params
all_budget_scales = [2**i for i in range(11)]
low_scales = [1, 2, 4, 8, 16, 32]
high_scales = [64, 128, 256, 512, 1024]
ten_data_seeds = list(range(10))

if __name__ == '__main__':
    
    arguments = {
        'ex': 'budget_scale', # experiment_id
        'plot': False, # plot
        'save-model': False, # save_model
        'bsm': [1, 2, 4, 8], # budget_scale_max
        'lr': 0.0001, # learning_rate
        'ss': 0.001, # ss_tau
        'sm': 0.01, # sm_tau
        'reg': 0.1, # reg_lambda
        'bs': 500, # batch_size
        'e': 1000, # epochs
        'es': None, # early_stopping
        'ds': ten_data_seeds, # data_seed
        'ws': None, # weights_seed
    }
    
    prefix = ["sbatch", "-p", partition_name, "-A", account_name]
    flags = {k: v for k, v in arguments.items() if isinstance(v, bool)}
    params = {k: v for k, v in arguments.items() if isinstance(v, list)}
    constants = {k: v for k, v in arguments.items() if k not in flags and k not in params and v is not None}

    # Generate all combinations of parameters
    combinations = list(itertools.product(*params.values()))
    print(f'Creating {len(combinations)} tasks')

    with open(batches_path, 'w') as f:
        for i, combo in enumerate(combinations):
            server = ["-w", f"{available_nodes[i % len(available_nodes)]}"]
            args = constants.copy()
            args.update(dict(zip(params.keys(), combo)))
            args = {k: v for k, v in args.items() if v is not None}
            args = [[f'--{k}', f'{v}'] for k, v in args.items()] + [[f'--{k}'] for k, v in flags.items() if v]
            args = list(itertools.chain(*args))
            command = prefix + server + [script_path] + args
            f.write(' '.join(command) + '\n')
        