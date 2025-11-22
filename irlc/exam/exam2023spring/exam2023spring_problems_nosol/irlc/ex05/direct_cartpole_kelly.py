# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Kel17] Matthew Kelly. An introduction to trajectory optimization: how to do your own direct collocation. SIAM Review, 59(4):849–904, 2017. (See kelly2017.pdf).
"""
from irlc.ex05.direct import guess
from irlc.ex05.model_cartpole import CartpoleModel
from irlc.ex03.control_cost import SymbolicQRCost
from irlc.ex05.direct import direct_solver, get_opts
import numpy as np
from gymnasium.spaces import Box

class KellyCartpoleModel(CartpoleModel):
    """Completes the Cartpole swingup task in exactly 2 seconds.

    The only changes to the original cartpole model is the inclusion of a new bound on ``tf_bound(self)``,
    to limit the end-time to :math:`t_F = 2`, and an updated cost function so that :math:`Q=0` and :math:`R=I`.
    """
    def get_cost(self) -> SymbolicQRCost:
        Q = np.zeros((4, 4)) #!b
        return SymbolicQRCost(Q=Q, R=np.asarray([[1.0]]) )  #!b Construct and return a new cost-function here.

    def tF_bound(self) -> Box:
        duration = 2 #!b
        return Box(duration, duration, shape=(1,)) #!b Implement the bound on tF here

def make_cartpole_kelly17():
    """
    Creates Cartpole problem. Details about the cost function can be found in (Kel17, Section 6)
    and details about the physical parameters can be found in (Kel17, Appendix E, table 3).
    """
    # this will generate a different carpole environment with an emphasis on applying little force u.
    duration = 2.0
    maxForce = 20
    model = KellyCartpoleModel(max_force=maxForce, mp=0.3, l=0.5, mc=1.0, dist=1)
    guess2 = guess(model)
    guess2['tF'] = duration # Our guess should match the constraints.
    return model, guess2

def compute_solutions():
    model, guess = make_cartpole_kelly17()
    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=40, ftol=1e-6)]
    solutions = direct_solver(model, options)
    return model, solutions

def direct_cartpole():
    model, solutions = compute_solutions()
    from irlc.ex05.direct_plot import plot_solutions
    print("Did we succeed?", solutions[-1]['solver']['success'])
    plot_solutions(model, solutions, animate=True, pdf="direct_cartpole_force")
    model.close()

if __name__ == "__main__":
    direct_cartpole()
