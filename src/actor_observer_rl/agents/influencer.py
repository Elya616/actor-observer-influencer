"""One learned Actor selector with an unchanged message relay."""

import torch


class Influencer(torch.nn.Module):
    def __init__(self, num_actors: int) -> None:
        super().__init__()
        if num_actors < 1:
            raise ValueError("num_actors must be positive")
        self.logits = torch.nn.Parameter(torch.zeros(num_actors))

    def probabilities(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def select_actor(self) -> tuple[int, torch.Tensor]:
        distribution = torch.distributions.Categorical(probs=self.probabilities())
        actor = distribution.sample()
        return int(actor.item()), distribution.log_prob(actor)

    def relay(self, message: torch.Tensor) -> torch.Tensor:
        # Do not detach: reported rewards must still train the selected Actor.
        return message
