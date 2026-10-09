"""Structured 2-D grid bandit for the scaling experiments.

The professor's scaling advice is to avoid an i.i.d.-random best action across
states, because that makes the optimal-action map a discontinuous, high-variance
function that can fail to learn for reasons unrelated to the Observer-Actor
mechanism. Instead we use a structured 2-D grid: the state is a cell, the actor
sees the cell's normalized ``(x1, x2)`` coordinates as continuous features, and
the grid is divided into ``num_actions`` regions whose centers define the best
action.

For ``num_actions = 4`` this is exactly the recommended quadrant task:

    region 0: lower-left,  region 1: lower-right,
    region 2: upper-left,  region 3: upper-right.

The reward is smooth in the state: for a fixed action it is a Gaussian bump
centered on that action's region center, so neighboring states have similar
rewards and the best action only flips across region boundaries. The Gaussian
width ``reward_sigma`` controls the contrast between the correct action and the
others; it is kept narrow enough that the correct action clearly dominates
(otherwise baseline-free REINFORCE separates the actions only very slowly),
while still being smooth in the state. The best action of any cell is the region
its center falls in, which keeps verification easy.

Scaling is done purely by increasing ``grid_size`` (the number of states is
``grid_size ** 2``) while the feature dimension stays 2 and the action count
stays fixed, so the policy's parameter count does not blow up with the state
count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from actor_observer_rl.envs.base import StepResult


@dataclass
class GridWorldBanditEnv:
    """I.i.d. 2-D grid bandit with structured region-based rewards."""

    grid_size: int
    num_actions: int = 4
    reward_sigma: float = 0.25
    _region_centers: torch.Tensor = field(init=False, repr=False)
    _generator: torch.Generator = field(init=False, repr=False)
    _current_state_index: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.grid_size < 1:
            raise ValueError("grid_size must be at least 1")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if self.reward_sigma <= 0.0:
            raise ValueError("reward_sigma must be positive")
        self._region_centers = self._make_region_centers(self.num_actions)
        self._generator = torch.Generator()

    @staticmethod
    def _make_region_centers(num_actions: int) -> torch.Tensor:
        """Lay out ``num_actions`` region centers on a near-square grid.

        Action ``k`` owns the Voronoi cell of center ``k``; the best action of a
        point is the nearest center. ``num_actions = 4`` gives the 2x2 quadrant
        task; larger counts give a finer but still structured tiling.
        """

        cols = math.ceil(math.sqrt(num_actions))
        rows = math.ceil(num_actions / cols)
        centers: list[tuple[float, float]] = []
        for k in range(num_actions):
            row = k // cols
            col = k % cols
            centers.append(((col + 0.5) / cols, (row + 0.5) / rows))
        return torch.tensor(centers, dtype=torch.float32)

    @property
    def num_states(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def state_dim(self) -> int:
        return 2

    def feature(self, state_index: int) -> torch.Tensor:
        """Continuous ``(x1, x2)`` cell-center coordinates in ``[0, 1]^2``."""

        if state_index < 0 or state_index >= self.num_states:
            raise IndexError("state_index outside valid range")
        col = state_index % self.grid_size
        row = state_index // self.grid_size
        x1 = (col + 0.5) / self.grid_size
        x2 = (row + 0.5) / self.grid_size
        return torch.tensor([x1, x2], dtype=torch.float32)

    def optimal_action(self, state_index: int) -> int:
        """Best action = region whose center is nearest the cell."""

        point = self.feature(state_index)
        distances = torch.linalg.vector_norm(self._region_centers - point, dim=1)
        return int(torch.argmin(distances).item())

    def optimal_actions(self) -> list[int]:
        return [self.optimal_action(index) for index in range(self.num_states)]

    def representative_states(self) -> list[int]:
        """One representative cell per action: the cell nearest its center."""

        representatives: list[int] = []
        for action in range(self.num_actions):
            center = self._region_centers[action]
            best_index = 0
            best_distance = math.inf
            for index in range(self.num_states):
                distance = float(
                    torch.linalg.vector_norm(self.feature(index) - center).item()
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            representatives.append(best_index)
        return representatives

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, object]]:
        if seed is not None:
            self._generator.manual_seed(seed)
        self._current_state_index = int(
            torch.randint(
                low=0,
                high=self.num_states,
                size=(1,),
                generator=self._generator,
            ).item()
        )
        return self.get_actor_observation(), {
            "seed": seed,
            "state_index": self._current_state_index,
        }

    def step(self, action: int) -> StepResult:
        if self._current_state_index is None:
            raise RuntimeError("reset must be called before step")
        state_index = self._current_state_index
        reward = self.get_reward(state_index, action)
        return StepResult(
            state=self.feature(state_index),
            observation=self.get_actor_observation(),
            reward=reward,
            done=True,
            info={"action": int(action), "state_index": state_index},
        )

    def get_full_state(self) -> torch.Tensor:
        return self.feature(self._require_state_index())

    def get_actor_observation(self) -> torch.Tensor:
        return self.feature(self._require_state_index())

    def get_reward(self, state: object, action: int) -> float:
        state_index = self._state_to_index(state)
        action_index = int(action)
        if action_index < 0 or action_index >= self.num_actions:
            raise IndexError(
                f"action {action_index} is outside valid range [0, {self.num_actions})"
            )
        point = self.feature(state_index)
        squared_distance = float(
            torch.sum((point - self._region_centers[action_index]) ** 2).item()
        )
        return math.exp(-squared_distance / (2.0 * self.reward_sigma**2))

    def _require_state_index(self) -> int:
        if self._current_state_index is None:
            raise RuntimeError("reset must be called before reading state")
        return self._current_state_index

    def _state_to_index(self, state: object) -> int:
        if isinstance(state, int):
            return state
        if isinstance(state, torch.Tensor):
            if state.numel() == 1:
                return int(state.item())
            if state.numel() == 2:
                x1 = float(state[0].item())
                x2 = float(state[1].item())
                col = min(self.grid_size - 1, max(0, int(x1 * self.grid_size)))
                row = min(self.grid_size - 1, max(0, int(x2 * self.grid_size)))
                return row * self.grid_size + col
        raise TypeError("state must be an int index or a 2-D feature tensor")
