# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
The Brachistochrone problem. See
https://apmonitor.com/wiki/index.php/Apps/BrachistochroneProblem
and (Bet10)
References:
  [Bet10] John T Betts. Practical methods for optimal control and estimation using nonlinear programming. Volume 19. Siam, 2010.
"""
import sympy as sym
import numpy as np
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
from gymnasium.spaces import Box

class ContiniouBrachistochrone(ControlModel): 
    state_labels= ["$x$", "$y$", "bead speed"]
    action_labels = ['Tangent angle']

    def __init__(self, g=9.82, h=None, x_dist=1): 
        self.g = g
        self.h = h
        self.x_dist = x_dist # or x_B
        super().__init__() 

    def get_cost(self) -> SymbolicQRCost:
        cost = SymbolicQRCost(Q=np.zeros((3,3)), R = np.zeros((1,1)), qc=1.0) #!b #!b Instantiate cost=SymbolicQRCost(...) here corresponding to minimum time.
        return cost

    def x0_bound(self) -> Box:
        return Box(0, 0, shape=(self.state_size,))

    def xF_bound(self) -> Box:
        return Box(np.array([self.x_dist, -np.inf, -np.inf]), np.array([self.x_dist, np.inf, np.inf]))

    def sym_f(self, x, u, t=None): #!f
        v = x[2]
        uu = u[0]
        xp = [v * sym.sin(uu), -v * sym.cos(uu), self.g * sym.cos(uu)]
        return xp

    def sym_h(self, x, u, t):
        r"""
        Add a dynamical constraint of the form

        .. math::

            h(x, u, t) \leq 0
        """
        if self.h is None:
            return []
        else:
            # compute a single dynamical constraint as in (Bet10, Example (4.10)) (Note y-axis is reversed in the example)
            return [ -x[1] - x[0]/2 - self.h ] #!b #!b
