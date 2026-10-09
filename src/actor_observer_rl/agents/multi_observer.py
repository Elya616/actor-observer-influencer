"""Multi-observer selection for imperfect-state experiments.

The imperfect-state track keeps a single Actor but introduces `N` Observers,
each with its own compression `f_phi_i`. The compression can be fixed, as in
paper Section 5.1, or learnable, as in the later joint-learning stages. At
every step the system samples a single observer index `k_t` from the
categorical distribution

```
rho = softmax(lambda)
k_t ~ Categorical(rho)
m_t = f_phi_{k_t}(s_t)
```

and the Actor sees `concat(o_t, m_t)`. Only the message of the sampled
observer enters the autograd graph, so only `phi_{k_t}` can receive a gradient
on a given step when the Observer is learnable. The selection logits are
updated through the REINFORCE term `log rho_{k_t}`.

`FollowingRates` keeps the learnable logits in a tiny PyTorch module that
exposes both the differentiable softmax weights and a `Categorical`
distribution. `MultiObserver` wraps the observers and the following rates
without hiding the math.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Optional

import torch

from actor_observer_rl.agents.observer import Observer


class FollowingRates(torch.nn.Module):
    """Learnable categorical distribution over `N` observers.

    `forward()` returns the differentiable probabilities `softmax(lambda)`
    used for inspection and for the marginalised expected-reward evaluation.
    `distribution()` returns a `torch.distributions.Categorical` whose
    `log_prob(k)` keeps the autograd path into `lambda`. `current_weights()`
    returns detached probabilities for logging without retaining the graph.
    """

    def __init__(
        self,
        num_observers: int,
        init_logits: Optional[Sequence[float]] = None,
    ) -> None:
        if num_observers <= 1:
            raise ValueError("num_observers must be greater than one")
        super().__init__()
        if init_logits is None:
            initial = torch.zeros(num_observers)
        else:
            initial = torch.as_tensor(list(init_logits), dtype=torch.float32)
            if initial.shape != torch.Size([num_observers]):
                raise ValueError(
                    "init_logits length must match num_observers"
                )
        self.num_observers = int(num_observers)
        self.logits = torch.nn.Parameter(initial)

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def distribution(self) -> torch.distributions.Categorical:
        """Return a Categorical distribution backed by `lambda`."""
        return torch.distributions.Categorical(logits=self.logits)

    def current_weights(self) -> torch.Tensor:
        """Return detached softmax weights for logging or evaluation."""
        with torch.no_grad():
            return torch.softmax(self.logits, dim=-1)


class MultiObserver:
    """Hold `N` observers and sample one of them per step.

    Reward routing follows the single-observer convention: the first observer
    is treated as the primary reward channel and its `report_reward` controls
    what the actor learns from. The other observers only contribute messages
    when sampled.
    """

    def __init__(
        self,
        observers: Sequence[Observer],
        following_rates: FollowingRates,
    ) -> None:
        observers = list(observers)
        if len(observers) <= 1:
            raise ValueError("MultiObserver requires at least two observers")
        if len(observers) != following_rates.num_observers:
            raise ValueError(
                "following_rates.num_observers must match len(observers)"
            )
        for index, observer in enumerate(observers):
            if observer.message_module is None:
                raise ValueError(
                    f"observer at index {index} is missing a message_module"
                )
        self.observers = observers
        self.following_rates = following_rates

    @property
    def num_observers(self) -> int:
        return len(self.observers)

    def sample_observer_and_message(
        self,
        full_state: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample `k_t ~ softmax(lambda)` and return `(k_t, m_t, log rho_{k_t})`.

        Only `observers[k_t].message(full_state)` is called, so only
        `phi_{k_t}` enters the autograd graph and only its parameters can
        receive a gradient from `loss.backward()`. The returned `log_prob`
        keeps the autograd path into `lambda` so the REINFORCE term in the
        loss reaches the following-rate logits.
        """
        distribution = self.following_rates.distribution()
        sampled = distribution.sample()
        observer_index = int(sampled.item())
        log_prob = distribution.log_prob(sampled)
        message = self.observers[observer_index].message(full_state)
        return observer_index, message, log_prob

    def report_reward(self, reward: float | torch.Tensor) -> float:
        return self.observers[0].report_reward(reward)

    def observer_parameters(self) -> Iterator[torch.nn.Parameter]:
        for observer in self.observers:
            yield from observer.parameters()

    def following_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.following_rates.parameters()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.observer_parameters()
        yield from self.following_parameters()
