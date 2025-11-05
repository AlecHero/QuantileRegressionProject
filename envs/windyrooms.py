from typing import Any
import numpy as np

from gymnasium.error import DependencyNotInstalled

from gymnasium import spaces
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from importlib import resources

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {UP: [-1, 0], RIGHT: [0, 1], DOWN: [1, 0], LEFT: [0, -1]}


class CliffCustomEnv(CliffWalkingEnv):
    def __init__(self, render_mode: str | None = None, is_slippery: bool = False, shape=(4, 12)):
        self.shape = shape
        self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 4

        self.is_slippery = is_slippery

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[self.shape[0]-1, 1:-1] = True

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

    def _calculate_transition_prob(self, current: list[int] | np.ndarray, move: int) -> list[tuple[float, Any, int, bool]]:
        if not self.is_slippery:
            deltas = [POSITION_MAPPING[move]]
        else:
            deltas = [
                POSITION_MAPPING[act] for act in [(move - 1) % 4, move, (move + 1) % 4]
            ]
        outcomes = []
        for delta in deltas:
            new_position = np.array(current) + np.array(delta)
            new_position = self._limit_coordinates(new_position).astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            if self._cliff[tuple(new_position)]:
                outcomes.append((1 / len(deltas), self.start_state_index, -100, False))
            elif tuple(new_position) == (self.shape[0] - 1, self.shape[1] - 1): # Goal state
                outcomes.append((1 / len(deltas), new_state, 10, True))
            else:
                outcomes.append((1 / len(deltas), new_state, -1, False))
        return outcomes

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

from typing import Any
import numpy as np

from gymnasium import spaces

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {UP: [-1, 0], RIGHT: [0, 1], DOWN: [1, 0], LEFT: [0, -1]}


class WindyRoomsEnv(CliffCustomEnv):
    def __init__(self, render_mode: str | None = None, p_random: float = 0.1, shape=(14, 11), rewards={"step":0.0,"goal":1.0}):
        self.p_random = p_random
        self.shape = shape
        self.start_state_index = np.ravel_multi_index((self.shape[0]-1, 0), self.shape)

        self.step_reward = rewards["step"]
        self.goal_reward = rewards["goal"]
    
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

    def _apply_wind_from(self, state: int) -> int:
        """Apply northward wind from given state."""
        pos = np.array(np.unravel_index(state, self.shape))
        wind_strength = int(self._wind[tuple(np.unravel_index(state, self.shape))])
        for _ in range(wind_strength):
            candidate = pos + np.array(POSITION_MAPPING[UP])
            if self._cliff[tuple(candidate)]:
                break
            pos = candidate
        pos = self._limit_coordinates(pos).astype(int)
        return int(np.ravel_multi_index(tuple(pos), self.shape))

    def _move_from_position(self, start_state: int, move: int) -> int:
        """Apply a move unless blocked by a cliff (wall)."""
        start_pos = np.array(np.unravel_index(start_state, self.shape))
        candidate = start_pos + np.array(POSITION_MAPPING[move])
        candidate = self._limit_coordinates(candidate).astype(int)
        if self._cliff[tuple(candidate)]:
            return start_state
        return int(np.ravel_multi_index(tuple(candidate), self.shape))

    def _calculate_transition_prob(
        self, current: list[int] | np.ndarray, move: int
    ) -> list[tuple[float, int, float, bool]]:
        """
        Implements Dabney et al. 2018 windy gridworld:
        10% random-direction move,
        90% intended move,
        then wind applies northward after movement.
        """
        current_state = int(np.ravel_multi_index(tuple(np.array(current)), self.shape))
        outcomes = []

        # All possible directions (intended + 3 random)
        directions = [(move + i) % 4 for i in range(4)]
        for act in directions:
            p = (1.0 - self.p_random) if act == move else self.p_random / 3.0

            # Step 1: try to move
            moved_state = self._move_from_position(current_state, act)

            # Step 2: apply wind
            winded_state = self._apply_wind_from(moved_state)

            # Step 3: reward and terminal
            pos = np.unravel_index(winded_state, self.shape)
            if pos == (self.shape[0] - 1, self.shape[1] - 1):
                outcomes.append((p, winded_state, self.goal_reward, True))
            else:
                outcomes.append((p, winded_state, self.step_reward, False))

        return outcomes