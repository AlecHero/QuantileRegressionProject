from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

arrow_directions = {0: "↑", 1: "→", 2: "↓", 3: "←"}

def plot_kde(quantiles, params, bandwidth=0.05, is_filled=True, save_path=None):
    from sklearn.neighbors import KernelDensity
    for a in range(params.action_size):
        samples = quantiles[a][:, None]
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples)
        
        x_grid = np.linspace(quantiles[a].min(), quantiles[a].max(), 10_000)[:, None]
        log_pdf = kde.score_samples(x_grid)
        pdf = np.exp(log_pdf)

        plt.plot(x_grid[:,0], pdf, label=f"action: {arrow_directions[a]}")
        if is_filled:
            plt.fill_between(x_grid[:,0], pdf, alpha=0.2)
        # plt.hist(quantiles[a], bins=50)
    
    plt.title("Kernel Density Estimation")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_returns(returns, bandwidth=0.01, is_filled=True, limits=None, save_path=None):
    from sklearn.neighbors import KernelDensity
    samples = returns[:, None]
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples)
    
    if limits is None: limits = (returns.min(), returns.max())
    
    x_grid = np.linspace(*limits, 10_000)[:, None]
    log_pdf = kde.score_samples(x_grid)
    pdf = np.exp(log_pdf)
    
    if is_filled:
        plt.fill_between(x_grid[:,0], pdf, alpha=0.2)
    plt.plot(x_grid[:,0], pdf)
    plt.title("Kernel Density Estimation")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_grad(quantiles, params, is_filled=True, highlight_optimal=False, save_path=None):
    from warnings import catch_warnings, filterwarnings
    with catch_warnings():
        filterwarnings("ignore", category=RuntimeWarning)
        
        tau = ((2 * np.arange(params.n_quantiles) + 1) / (2.0 * params.n_quantiles))
        a_max = quantiles.mean(1).argmax()
        for a in range(quantiles.shape[0]):
            r = quantiles[a]
            grad = np.gradient(tau, r)
            if (np.isnan(grad).any() or np.isinf(grad).any()): continue
            plt.plot(r, grad, label=f"action: {arrow_directions[a]}", alpha=0.55 if highlight_optimal and a != a_max else 1.0)
            if is_filled:
                plt.fill_between(r, np.gradient(tau, r), alpha=0.2)
        plt.title("Density Estimation via Quantile Gradients")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()


def plot_optimal_action(qtable, params, save_path=None):
    map_size = params.map_size
    qtable_val_max = qtable.max(axis=1).reshape(map_size)
    qtable_best_action = qtable.argmax(1).reshape(map_size)
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)

    for idx, val in enumerate(qtable_best_action.flatten()):
        if not np.allclose(qtable[idx], 0.0):
            qtable_directions[idx] = arrow_directions[val]
    qtable_directions = qtable_directions.reshape(map_size)

    fig, ax = plt.subplots(figsize=(map_size[1], map_size[0]))

    heatmap = sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax,
        cmap=sns.color_palette("Blues", as_cmap=True),
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "25", "color": "black"},
        square=True,
    )
    heatmap.set(title="Learned Q-values\nArrows represent best action")

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_mean_convergence(qtables, params, save_path=None):
    eps_n = qtables.shape[0]
    skip = params.save_skip

    mean_Qs = qtables.mean(1)
    delta_Q = np.abs(mean_Qs[1:] - mean_Qs[:-1])

    for a in range(params.action_size):
        plt.plot(np.arange(1, eps_n) * skip, delta_Q[:, a])

    plt.xlabel("Episode")
    plt.ylabel("Mean Q-value change")
    plt.title("Q-value convergence")

    xticks = np.linspace(0, eps_n * skip, 6, dtype=int)
    plt.xticks(xticks, [f"{x:,}" for x in xticks])  # format with commas
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_value_approx(y, params):
    fig, ax = plt.subplots(figsize=(10,2))
    ax.plot(y)
    ax.set_yscale("log")

    ticks = ax.get_xticks()[1:-1].astype(int)
    ax.set_xticks(ticks, [f"{x:,}" for x in ticks*params.save_skip])
    ax.set_xlim(ticks[0], ticks[-1])
    
    ax.set_xlabel("Episodes", fontweight="bold")
    ax.set_ylabel(" $ W_1(Z_{MC}^\pi(x_s), Z(x_s)) $");
    ax.set_title("Value Distribution Approximation Error", fontweight="bold");

    return fig, ax