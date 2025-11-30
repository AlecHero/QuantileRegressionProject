from typing import Any
import numpy as np

from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv, POSITION_MAPPING, UP, RIGHT, DOWN, LEFT
from gymnasium import spaces 
from importlib import resources

class GridWorld(CliffWalkingEnv):
    def __init__(self, p_random = 0.1, is_slippery: bool = False, is_windy: bool = False, shape=(4, 12), rewards={"step":-1,"goal":10,"fail":-100}, MoG=None, render_mode: str | None = "rgb_array"):
        self.shape = shape
        self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)
        self.goal_pos = self.shape[0]-1, self.shape[1]-1
        
        self.nS = np.prod(self.shape)
        self.nA = 4

        self.MoG = MoG
        self.rewards = rewards
        self.is_slippery = is_slippery
        self.is_windy = is_windy
        self.p_random = p_random

        self._cliff = getattr(self, "_cliff")
        self._is_terminal = self._cliff.copy()
        self._is_terminal[tuple(self.goal_pos)] = True
        
        self._reward = np.full(self.shape, self.rewards["step"])
        self._reward[self.goal_pos] = self.rewards["goal"]
        self._reward[self._cliff] = self.rewards["fail"]
        
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

    def _to_state(self, pos): return np.ravel_multi_index(pos, self.shape)

    def _calculate_transition_prob(self, current: list[int] | np.ndarray, move: int) -> list[tuple[float, Any, int, bool]]:
        if self.is_slippery:
            deltas = [(POSITION_MAPPING[act], 1.0 / self.nA) for act in [(move - 1) % 4, move, (move + 1) % 4]]
        elif self.is_windy:
            deltas = [(POSITION_MAPPING[act], self.p_random / self.nA) for act in range(self.nA)] + [(POSITION_MAPPING[move], 1.0-self.p_random)]
        else:
            deltas = [(POSITION_MAPPING[move], 1.0)]
        
        outcomes = []
        for delta, p in deltas:
            new_pos = tuple(self._limit_coordinates(np.array(current) + np.array(delta)).astype(int))
            outcomes.append((p, self._to_state(new_pos), self._reward[new_pos], self._is_terminal[new_pos]))
        
        return outcomes

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        state, reward, terminated, truncated, info = super().step(action)
        if self.MoG is not None and reward == self.rewards["goal"]:
            reward = self.MoG.draw_samples(1)[0]
        return state, reward, terminated, truncated, info

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
                resources.files("gymnasium.envs.toy_text.img") / "elf_up.png",
                resources.files("gymnasium.envs.toy_text.img") / "elf_right.png",
                resources.files("gymnasium.envs.toy_text.img") / "elf_down.png",
                resources.files("gymnasium.envs.toy_text.img") / "elf_left.png",
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = resources.files("gymnasium.envs.toy_text.img") / "stool.png"
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = resources.files("gymnasium.envs.toy_text.img") / "cookie.png"
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                resources.files("gymnasium.envs.toy_text.img") / "mountain_bg1.png",
                resources.files("gymnasium.envs.toy_text.img") / "mountain_bg2.png",
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                resources.files("gymnasium.envs.toy_text.img") / "mountain_near-cliff1.png",
                resources.files("gymnasium.envs.toy_text.img") / "mountain_near-cliff2.png",
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = resources.files("gymnasium.envs.toy_text.img") / "mountain_cliff.png"
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


class CliffWalking(GridWorld):
    def __init__(self, shape=(4,12), *args, **kwargs):
        self._cliff = np.zeros(shape, dtype=bool)
        self._cliff[shape[0]-1, 1:-1] = True

        super().__init__(shape=shape, *args, **kwargs)


class Walkway(GridWorld):
    def __init__(self, shape=(1,10), *args, **kwargs):
        self._cliff = np.zeros(shape, dtype=bool)
        super().__init__(shape=shape, *args, **kwargs)


class WindyRooms(GridWorld):
    def __init__(self, shape=(14, 11), *args, **kwargs):
        self._cliff = np.zeros(shape, dtype=bool)
        self._cliff[:, 5] = True
        self._cliff[3, 5] = False

        self._wind = np.zeros(shape, dtype=int)
        self._wind[:, 1] = 1
        self._wind[:, (2, 3, 4)] = 2

        super().__init__(shape=shape, *args, **kwargs)

    def _limit_cliff(self, pos, new_pos): return pos if self._cliff[tuple(new_pos)] else new_pos

    def _calculate_transition_prob(self, pos: list[int] | np.ndarray, action: int) -> list[tuple[float, int, float, bool]]:
        pos = np.array(pos)
        outcomes = []
        
        for a in range(self.nA):
            dir = np.array(POSITION_MAPPING[a])
            new_pos = self._limit_coordinates(pos + dir)
            new_pos = self._limit_cliff(pos, new_pos)
            outcomes.append((self.p_random / 4.0, self._to_state(new_pos), self._reward[tuple(new_pos)], self._is_terminal[tuple(new_pos)]))

        dir = np.array(POSITION_MAPPING[action])
        wind_pos = self._limit_coordinates(pos + np.array(POSITION_MAPPING[UP]) * self._wind[tuple(pos)])
        new_pos = self._limit_coordinates(wind_pos + dir)
        new_pos = self._limit_cliff(wind_pos, new_pos)
        outcomes.append((1.0 - self.p_random, self._to_state(new_pos), self._reward[tuple(new_pos)], self._is_terminal[tuple(new_pos)]))
        return outcomes