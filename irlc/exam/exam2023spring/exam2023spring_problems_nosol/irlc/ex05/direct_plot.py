# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import matplotlib.pyplot as plt
import numpy as np
from irlc.ex03.control_model import plot_trajectory
from irlc import savepdf

"""
Helper function for plotting.
"""
def plot_solutions(model, solutions, animate=True, pdf=None, plot_defects=True, Ix=None, animate_repeats=1, animate_all=False, plot=True):

    for k, sol in enumerate(solutions):
        grd = sol['grid']
        x_res = sol['grid']['x']
        u_res = sol['grid']['u']
        ts = sol['grid']['ts']
        u_fun = lambda x, t: sol['fun']['u'](t)
        N = len(ts)
        if pdf is not None:
            pdf_out = f"{pdf}_sol{N}"

        x_sim, u_sim, t_sim = model.simulate(x0=grd['x'][0, :], u_fun=u_fun, t0=grd['ts'][0], tF=grd['ts'][-1], N_steps=1000)
        if animate and (k == len(solutions)-1 or animate_all):
            for _ in range(animate_repeats):
                animate_rollout(model, x0=grd['x'][0, :], u_fun=u_fun, t0=grd['ts'][0], tF=grd['ts'][-1], N_steps=1000, fps=30)

        eqC_val = sol['eqC_val']
        labels = model.state_labels

        if Ix is not None:
            labels = [l for k, l in enumerate(labels) if k in Ix]
            x_res = x_res[:,np.asarray(Ix)]
            x_sim = x_sim[:,np.asarray(Ix)]

        print("Initial State: " + ",".join(labels))
        print(x_res[0])
        print("Final State:")
        print(x_res[-1])
        if plot:
            ax = plot_trajectory(x_res, ts, lt='ko-', labels=labels, legend="Direct state prediction $x(t)$")
            plot_trajectory(x_sim, t_sim, lt='-', ax=ax, labels=labels, legend="RK4 exact simulation")
            # plt.suptitle("State", fontsize=14, y=0.98)
            # make_space_above(ax, topmargin=0.5)

            if pdf is not None:
                savepdf(pdf_out +"_x")
            plt.show(block=True)
            # print("plotting...")
            plot_trajectory(u_res, ts, lt='ko-', labels=model.action_labels, legend="Direct action prediction $u(t)$")
            # print("plotting... B")
            # plt.suptitle("Action", fontsize=14, y=0.98)
            # print("plotting... C")
            # make_space_above(ax, topmargin=0.5)
            # print("plotting... D")
            if pdf is not None:
                savepdf(pdf_out +"_u")
            plt.show(block=True)
            if plot_defects:
                plot_trajectory(eqC_val, ts[:-1], lt='-', labels=labels)
                plt.suptitle("Defects (equality constraint violations)")
                if pdf is not None:
                    savepdf(pdf_out +"_defects")
                plt.show(block=True)
    return x_sim, u_sim, t_sim


def animate_rollout(model, x0, u_fun, t0, tF, N_steps = 1000, fps=10):
    """ Helper function to animate a policy.  """

    import time
    # if sys.gettrace() is not None:
    #     print("Not animating stuff in debugger as it crashes.")
    #     return
    y, _, tt = model.simulate(x0, u_fun, t0, tF, N_steps=N_steps)
    secs = tF-t0
    frames = int( np.ceil( secs * fps ) )
    I = np.round( np.linspace(0, N_steps-1, frames)).astype(int)
    y = y[I,:]

    for i in range(frames):
        model.render(x=y[i], render_mode="human")
        time.sleep(1/fps)
