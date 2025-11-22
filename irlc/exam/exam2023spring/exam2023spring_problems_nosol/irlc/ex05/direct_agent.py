# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex05.direct import direct_solver, get_opts, guess
from irlc.ex04.model_pendulum import SinCosPendulumModel
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment
from irlc import train
from irlc import Agent
import numpy as np
import matplotlib.pyplot as plt
from irlc import savepdf
from irlc.ex05.direct_plot import plot_solutions

class DirectAgent(Agent):
    def __init__(self, env: ControlEnvironment, options=None):
        cmod = env.discrete_model.continuous_model  # Get the continuous-time model for planning

        if options is None:
            options = [get_opts(N=10, ftol=1e-3, guess=guess(cmod), verbose=False),
                       get_opts(N=60, ftol=1e-6, verbose=False)
                       ]
        solutions = direct_solver(cmod, options)

        # The next 3 lines are for plotting purposes. You can ignore them.
        self.x_grid = np.stack([env.discrete_model.phi_x(x) for x in solutions[-1]['grid']['x']])
        self.u_grid = np.stack([env.discrete_model.phi_u(u) for u in solutions[-1]['grid']['u']])
        self.ts_grid = np.stack(solutions[-1]['grid']['ts'])
        # set self.ufun equal to the solution (policy) function. You can get it by looking at `solutions` computed above
        self.solutions = solutions
        self.ufun = solutions[-1]['fun']['u'] #!b #!b set self.ufun = solutions[....][somethingsomething] (insert a breakpoint, it should be self-explanatory).
        super().__init__(env)

    def pi(self, x, k, info=None): #!f
        t = info['time_seconds']
        """ Return the action given x and t. As a hint, you will only use t, and self.ufun computed a few lines above"""
        if t > self.ts_grid[-1]:
            print("Simulation time is", t, "which exceeds the maximal planning horizon t_F =", self.ts_grid[-1])
            raise Exception("Time exceed agents planning horizon")

        u = self.ufun(t)
        u = np.asarray(self.env.discrete_model.phi_u(u))
        return u

def train_direct_agent(animate=True, plot=False):
    from irlc.ex04.model_pendulum import PendulumModel
    model = PendulumModel()
    """
    Test out implementation on a fairly small grid. Note this will work fairly terribly.
    """
    guess = {'t0': 0,
             'tF': 4,
             'x': [np.asarray([0, 0]), np.asarray([np.pi, 0])],
             'u': [np.asarray([0]), np.asarray([0])]}

    options = [get_opts(N=10, ftol=1e-3, guess=guess),
               get_opts(N=20, ftol=1e-3),
               get_opts(N=80, ftol=1e-6)
               ]

    dmod = DiscreteControlModel(model=model, dt=0.1) # Discretize the pendulum model. Used for creating the environment.
    denv = ControlEnvironment(discrete_model=dmod, Tmax=4, render_mode='human' if animate else None)
    agent = DirectAgent(denv, options=options)
    denv.Tmax = agent.solutions[-1]['fun']['tF'] # Specify max runtime of the environment. Must be based on the Agent's solution.
    stats, traj = train(denv, agent=agent, num_episodes=1, return_trajectory=True)

    if plot:
        from irlc import plot_trajectory
        plot_trajectory(traj[0], env=denv)
        savepdf("direct_agent_pendulum")
        plt.show()

    return stats, traj, agent

if __name__ == "__main__":
    stats, traj, agent = train_direct_agent(animate=True, plot=True)
    print("Obtained cost", -stats[0]['Accumulated Reward'])

    # Let's try to plot the state-vectors for the two models. They are not going to agree that well.
    plt.plot(agent.ts_grid, agent.x_grid, 'r-', label="Direct solver prediction")
    plt.plot(traj[0].time, traj[0].state, 'k-', label='Simulation')
    plt.legend()
    plt.show()
