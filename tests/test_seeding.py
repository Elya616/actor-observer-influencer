"""Tests for reproducibility helpers."""

from __future__ import annotations

import numpy as np
import torch

from actor_observer_rl.utils.seeding import set_global_seed


def test_set_global_seed_reproducibly_seeds_numpy_and_torch() -> None:
    set_global_seed(123)
    torch_values_a = torch.rand(3)
    numpy_values_a = np.random.rand(3)

    set_global_seed(123)
    torch_values_b = torch.rand(3)
    numpy_values_b = np.random.rand(3)

    assert torch.allclose(torch_values_a, torch_values_b)
    assert np.allclose(numpy_values_a, numpy_values_b)
