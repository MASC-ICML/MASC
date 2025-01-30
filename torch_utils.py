import torch
from contextlib import contextmanager


@contextmanager
def torch_temporary_seed(seed: int):
    """
    A context manager which temporarily sets torch's random seed, then sets the random
    number generator state back to its previous state.
    :param seed: The temporary seed to set.
    """
    if seed is None:
        yield
    else:
        state = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            yield
        finally:
            torch.random.set_rng_state(state)