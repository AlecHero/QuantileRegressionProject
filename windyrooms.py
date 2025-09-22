from contextlib import closing
from io import StringIO
from os import path
from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {UP: [-1, 0], RIGHT: [0, 1], DOWN: [1, 0], LEFT: [0, -1]}


class WindyRoomsEnv(Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

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

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

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

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(path.dirname(__file__), "img/elf_up.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(path.dirname(__file__), "img/mountain_bg1.png"),
                path.join(path.dirname(__file__), "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(path.dirname(__file__), "img/mountain_near-cliff1.png"),
                path.join(path.dirname(__file__), "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(path.dirname(__file__), "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if row < self.shape[0] - 1 and self._cliff[row + 1, col]:
                self.window_surface.blit(self.near_cliff_img[check_board_mask], pos)
            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        outfile = StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (self.shape[0] - 1, self.shape[1] - 1):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            elif self._wind[position]:
                output = f" {self._wind[position]} "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()