import numpy as np
import pandas as pd

def remove_examples_by_condition(X: pd.DataFrame,
                                 B: pd.Series,
                                 Y: pd.Series,
                                 amount_to_leave,
                                 X_cond=None,
                                 B_cond=None,
                                 Y_cond=None,
                                 percent=False,
                                 seed=42):
    np.random.seed(seed)
    assert (not percent or amount_to_leave < 1)
    X.reset_index(inplace=True, drop=True)
    B.reset_index(inplace=True, drop=True)
    all_indices = X.index.values.tolist()
    X_indices = all_indices if X_cond is None else np.asarray(X_cond(X)).nonzero()[0]
    Z_indices = all_indices if B_cond is None else np.asarray(B_cond(B)).nonzero()[0]
    Y_indices = all_indices if Y_cond is None else np.asarray(Y_cond(Y)).nonzero()[0]
    # Y_indices = all_indices if Y_cond is None else np.where(Y_cond(Y))[0].tolist()
    indices_intersect = sorted(list(set(X_indices) & set(Z_indices) & set(Y_indices)))
    amount_to_remove = len(indices_intersect) - amount_to_leave if not percent else int(
        len(indices_intersect) * (1 - amount_to_leave))
    assert (amount_to_remove > 0)
    sample = np.random.choice(indices_intersect, amount_to_remove, replace=False)
    X.drop(index=sample, axis=0, inplace=True)
    B.drop(index=sample, axis=0, inplace=True)
    Y.drop(index=sample, axis=0, inplace=True)
    X.reset_index(inplace=True, drop=True)
    B.reset_index(inplace=True, drop=True)
    Y.reset_index(inplace=True, drop=True)
    # Y = np.delete(Y, sample, axis=0)
    return X, B, Y
