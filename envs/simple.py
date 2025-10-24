import numpy as np

from gymnasium import Env, spaces
from contextlib import closing
from io import StringIO

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {UP: [-1, 0], RIGHT: [0, 1], DOWN: [1, 0], LEFT: [0, -1]}


class SimpleEnv(Env):
    def __init__(self, render_mode: str | None = None):
        self.shape = (3, 3)
        self.start_state_index = np.ravel_multi_index((1, 1), self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 3

        self.P = {
            4: {
                0: (1.0, 1, self._dist_normal, True),
                1: (1.0, 5, self._dist_bimodal, True),
                2: (1.0, 7, self._dist_uniform, True),
                # 3: (1.0, 3, self._dist_skewed, True),
            }
        }
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

    def _dist_normal(self):  return self.np_random.normal(0, 1)
    def _dist_bimodal(self): return self.np_random.normal(-4 if (self.np_random.random() < 0.5) else 0, 1.0)
    
    def _dist_skewed(self):  return self.np_random.exponential(1) - 1.0
    def _dist_uniform(self): return self.np_random.uniform(0, 4)

    def step(self, a):
        p, s, dist_func, t = self.P[self.s][a]
        self.s = s
        self.lastaction = a
        r = dist_func()

        return int(s), r, t, False, {"prob": p}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.s = self.start_state_index
        self.lastaction = None

        return int(self.s), {"prob": 1}
    
    def render(self):
        outfile = StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position != (1, 1):
                output = " T "
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