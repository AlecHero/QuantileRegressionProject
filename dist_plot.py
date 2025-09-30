import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ARROW_DIRECTIONS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

def get_tau(n_quantiles): return ((2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles))

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size)
    qtable_best_action = qtable.argmax(1).reshape(map_size)
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    
    for idx, val in enumerate(qtable_best_action.flatten()):
        if not np.allclose(qtable[idx], 0.0):
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = ARROW_DIRECTIONS[val]
    qtable_directions = qtable_directions.reshape(map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()

def plot_states_actions_distribution(states, actions, map_size):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    plt.show()


### NEWER

def plot_mean_convergence(qtables, ylim=(-0.1,2)):
    mean_Qs = qtables[:,:37].mean(1)
    delta_Q = np.abs(mean_Qs[1:] - mean_Qs[:-1])
    for a in range(4):
        plt.plot(delta_Q[:,a])
    plt.xlabel("Episode")
    plt.ylabel("Mean Q-value change")
    plt.title("Q-value convergence")
    plt.ylim(ylim)
    plt.show()

def plot_state_convergence(qtables, state=36):
    from matplotlib.ticker import StrMethodFormatter
    x = np.arange(qtables.shape[0]) * 100
    for action in range(4):
        plt.plot(x, qtables[:,state,action], label=f"{ARROW_DIRECTIONS[action]}")
    plt.xlabel("Episode")
    plt.ylabel("Max Q-value (averaged over states)")
    plt.title("Q-value evolution")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.show()

from ipywidgets import interact, IntSlider
from IPython.display import clear_output

def _qr_episodes(tables, params, episode, state):
    # clear_output(wait=True)
    plt.figure(figsize=(6,4))
    
    tau = get_tau(params.n_quantiles)
    
    for j, action_quantiles in enumerate(tables[episode, state]):
        dq_dtau = np.gradient(action_quantiles, tau)
        pdf = 1.0 / np.abs(dq_dtau)
        plt.plot(action_quantiles, pdf, label=f"action: {ARROW_DIRECTIONS[j]}")
    
    plt.xlabel("Reward (x)")
    plt.ylabel("PDF f(x)")
    plt.title(f"Episode {episode*params.save_skip}")
    plt.legend()
    plt.show()

def qr_display_episodes(tables, params, state=36):
    slide_func = lambda episode: _qr_episodes(tables, params, episode, state=state)
    interact(slide_func, episode=IntSlider(min=0, max=tables.shape[0]-1, step=1, value=0))


def qr_pdf_kde(action_quantiles, bandwidth=0.5):
    from sklearn.neighbors import KernelDensity
    samples = action_quantiles[:, None]
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples)
    log_pdf = kde.score_samples(samples)
    pdf = np.exp(log_pdf)
    return samples[:,0], pdf

def _qr_episodes_kde(tables, params, episode, state, action, bandwidth=0.1):
    plt.figure(figsize=(6,4))
    
    x, pdf = qr_pdf_kde(tables[episode, state, action], bandwidth=bandwidth)
    plt.plot(x, pdf)
    plt.fill_between(x, pdf, alpha=0.5)
    plt.show()

def qr_display_episodes_kde(tables, params, state=36, action=0):
    slide_func = lambda episode: _qr_episodes_kde(tables, params, episode, state, action)
    interact(slide_func, episode=IntSlider(min=0, max=tables.shape[0]-1, step=1, value=0))


def plot_cdf_states(tables, params, states=[(3, 0), (2, 0), (2, 10), (2, 11)]):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme()
    tau = ((2 * np.arange(params.n_quantiles) + 1) / (2.0 * params.n_quantiles))
    arrow_directions = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, pos in enumerate(states):
        ax = axes[i]
        state = np.ravel_multi_index(pos, params.map_size)

        for j, quantiles in enumerate(tables[-1,state]):
            ax.plot(quantiles, tau, label=f"action: {arrow_directions[j]}")

        ax.set_title(f"Learned Quantiles for Position: {pos}")
        ax.set_xlabel("Space of Returns")
        ax.set_ylabel("Probability Space")
        ax.set_yticks(tau[::5])
        ax.legend()

    plt.tight_layout()
    plt.show()