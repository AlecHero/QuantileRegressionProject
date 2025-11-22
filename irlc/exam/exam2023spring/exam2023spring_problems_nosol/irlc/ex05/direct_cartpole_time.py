# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex05.model_cartpole import CartpoleModel
from irlc.ex05.direct import direct_solver, get_opts
from irlc.ex05.direct_plot import plot_solutions
from irlc.ex05.direct import guess

def compute_solutions():
    """
    See: https://github.com/MatthewPeterKelly/OptimTraj/blob/master/demo/cartPole/MAIN_minTime.m
    """
    model = CartpoleModel(max_force=50, mp=0.5, mc=2.0, l=0.5)
    guess2 = guess(model)
    guess2['tF'] = 2
    guess2['u'] = [[0], [0]]

    options = [get_opts(N=8, ftol=1e-3, guess=guess2), # important.
               get_opts(N=16, ftol=1e-6),              # This is a hard problem and we need gradual grid-refinement.
               get_opts(N=32, ftol=1e-6),
               get_opts(N=60, ftol=1e-6)
               ]
    solutions = direct_solver(model, options)
    return model, solutions

if __name__ == "__main__":
    model, solutions = compute_solutions()
    x_sim, u_sim, t_sim = plot_solutions(model, solutions[:], animate=True, pdf="direct_cartpole_mintime")
    model.close()
    print("Did we succeed?", solutions[-1]['solver']['success'])
