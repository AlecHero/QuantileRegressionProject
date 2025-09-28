from typing import Any
import numpy as np

from gymnasium import spaces
from cliffcustom import CliffCustomEnv

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {UP: [-1, 0], RIGHT: [0, 1], DOWN: [1, 0], LEFT: [0, -1]}


class WindyRoomsEnv(CliffCustomEnv):
    def __init__(self, render_mode: str | None = None, p_random: float = 0.1, shape=(14, 10), step_reward: int = -1, goal_reward: int = 10):
        self.p_random = p_random
        self.shape = shape
        self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)

        self.step_reward = step_reward
        self.goal_reward = goal_reward
    
        self.nS = np.prod(self.shape)
        self.nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[:, 5] = True
        self._cliff[3, 5] = False

        # Wind Location
        self._wind = np.zeros(self.shape, dtype=int)
        self._wind[:, 1] = 1
        self._wind[:, (2, 3, 4)] = 2

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, UP)
            self.P[s][RIGHT] = self._calculate_transition_prob(position, RIGHT)
            self.P[s][DOWN] = self._calculate_transition_prob(position, DOWN)
            self.P[s][LEFT] = self._calculate_transition_prob(position, LEFT)

        # Calculate initial state distribution
        # We always start in state (3, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

        # pygame utils
        self.cell_size = (60, 60)
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None

    def _get_wind_state(self, state: int) -> int:
        position = np.unravel_index(state, self.shape)
        winded_position = position + np.array(POSITION_MAPPING[UP]) * self._wind[tuple(position)]
        winded_position = self._limit_coordinates(winded_position).astype(int)
        winded_state = np.ravel_multi_index(tuple(winded_position), self.shape)
        return winded_state

    def _calculate_transition_prob(
        self, current: list[int] | np.ndarray, move: int
    ) -> list[tuple[float, Any, int, bool]]:

        if not self.p_random:
            deltas = [POSITION_MAPPING[move]]
        else:
            deltas = [POSITION_MAPPING[act] for act in [move, (move + 1) % 4, (move + 2) % 4, (move + 3) % 4]]
        
        outcomes = []
        for delta in deltas:
            position = np.array(current)
            new_position = position + np.array(delta)
            new_position = self._limit_coordinates(new_position).astype(int)

            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            old_state = np.ravel_multi_index(tuple(position), self.shape)

            p_trans = self.p_random / len(deltas)

            if self._cliff[tuple(new_position)]:
                outcomes.append((p_trans, old_state, self.step_reward, False))
                if delta == deltas[0]:
                    outcomes.append((1.0 - self.p_random, self._get_wind_state(old_state), self.step_reward, False))
            
            elif tuple(new_position) == (self.shape[0] - 1, self.shape[1] - 1): # Goal state
                outcomes.append((p_trans, new_state, self.goal_reward, True))
                if delta == deltas[0]:
                    _wind_state = self._get_wind_state(new_state)
                    if new_state == _wind_state:
                        outcomes.append((1.0 - self.p_random, _wind_state, self.goal_reward, True))
                    else:
                        outcomes.append((1.0 - self.p_random, _wind_state, self.step_reward, False))
            
            else:
                outcomes.append((p_trans, new_state, self.step_reward, False))
                if delta == deltas[0]:
                    outcomes.append((1.0 - self.p_random, self._get_wind_state(new_state), self.step_reward, False))
        
        return outcomes