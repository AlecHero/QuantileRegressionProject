from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#FF752B", "#17becf", "#c9f600", "#ff2e97", "#00ffb3"]
MC_NAME = "$Z_{MC}^\pi$"
# MC_NAME = "True Dist."

def W1(truth, estimate): return abs(truth - estimate).mean(-1)
def VSE(z1, z2): return (z1.mean(-1) - z2.mean(-1))**2

def get_kde(q, bandwidth):
    samples = q[:, None]
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples)
    return_kde = lambda x: np.exp(kde.score_samples(x[:, None]))
    return return_kde


def plot_pdf(plot_objs, returns=None, bandwidth=0.08, figsize=(5,4), show_hist=False, x_spacing=0.03, N=32, save_to=None, hist_alpha=0.2, include_legend=False, linewidth=1.0, ylabel=None, returns_name=MC_NAME):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    if ylabel is None:
        fig.supylabel("Estimated Value Distribution", fontweight="bold", fontsize="large")
    else:
        fig.supylabel(ylabel, fontweight="bold", fontsize="large")        

    all_data = np.concatenate([obj[0] for obj in plot_objs] + ([returns] if returns is not None else []))
    bins = np.histogram_bin_edges(all_data, bins=N)
    xmin, xmax = all_data.min(), all_data.max()
    x_add = max(np.abs(xmin), np.abs(xmax))*x_spacing
    Xs = np.linspace(xmin - x_add, xmax + x_add, 500)
    
    if returns is not None:
        ax.plot(Xs, get_kde(returns, bandwidth=bandwidth)(Xs), label=returns_name, linestyle="--", linewidth=2, color=COLORS[0])
        if show_hist: ax.hist(returns, bins=bins, density=True, alpha=hist_alpha, color=COLORS[0])
    
    for idx, (quantile, name) in enumerate(plot_objs):
        ax.plot(Xs, get_kde(quantile, bandwidth=bandwidth)(Xs), label=name, color=COLORS[idx+1], linewidth=linewidth)
        if show_hist: ax.hist(quantile, bins=bins, density=True, alpha=hist_alpha, color=COLORS[idx+1])

    ax.set_xlim((Xs.min(), Xs.max()))
    ax.set_xlabel("Returns", fontweight="bold", fontsize="large")
    ax.grid(True)
    if include_legend: ax.legend(prop={"weight":"bold", "size":"large"})
    plt.tight_layout()
    
    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_to}/pdf_compare.pdf", format="pdf", bbox_inches="tight")


def plot_cdf(plot_objs, returns, tau, figsize=(5,4), save_to=None, linewidths=(2,2), returns_name=MC_NAME):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    fig.supylabel("Cumulative Value Distribution", fontweight="bold")

    ax.plot(returns, tau, label=returns_name, linestyle="--", linewidth=linewidths[0], color=COLORS[0])
    for idx, (quantile, name) in enumerate(plot_objs):
        ax.plot(quantile, tau, label=name, color=COLORS[idx+1], linewidth=linewidths[1])

    ax.set_ylim(0.0,1.0)
    ax.set_xlabel("Returns", fontweight="bold")
    ax.grid(True)
    ax.legend(prop={"weight":"bold", "size":"large"})
    plt.tight_layout()
    
    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_to}/cdf_compare.pdf", format="pdf", bbox_inches="tight")


def get_fixed(stat):
    log_val = np.log(np.clip(stat, 1e-12, None))
    mu_log = log_val.mean(0)
    sd_log = log_val.std(0)
    return np.exp(mu_log), np.exp(mu_log - sd_log), np.exp(mu_log + sd_log)


def plot_metrics(plot_qs, returns, figsize=(10,6), save_skip=1.0, save_to=None, ylim0=None, ylim1=None, alphas=[0.3,0.3]):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'hspace': 0.33})

    for idx, (quantile, name) in enumerate(plot_qs):
        vse = VSE(returns, quantile)
        mu,sd1,sd2 = get_fixed(vse)
        axes[0].plot(mu, label=name, color=COLORS[idx+1])
        axes[0].fill_between(range(vse.shape[1]), sd1, sd2, alpha=alphas[0], color=COLORS[idx+1])
        
        if quantile.shape[-1] > 1:
            w1 = W1(returns, quantile)
            mu,sd1,sd2 = get_fixed(w1)
            axes[1].plot(mu, label=name, color=COLORS[idx+1])
            axes[1].fill_between(range(w1.shape[1]), sd1, sd2, alpha=alphas[1], color=COLORS[idx+1])
    
    axes[1].set_title("Value Distribution Approximation Error", fontweight="bold")
    axes[1].set_ylabel("$ W_1(Z_{True}, Z) $", fontsize="large")
    axes[1].set_yscale("log")
    ticks = axes[1].get_xticks().astype(int)
    axes[1].set_xticks(ticks, [f"{x:,}" for x in ticks*save_skip])
    axes[1].set_xlim(ticks[1], ticks[-2])
    if ylim1 is not None: axes[1].set_ylim(*ylim1)
    axes[1].legend(loc=1, prop={"weight":"bold", "size":"large"})
    axes[1].set_xlabel("Episodes", fontweight="bold", fontsize="large")

    if ylim0 is not None: axes[0].set_ylim(*ylim0)
    axes[0].set_title("Value Function Approximation Error", fontweight="bold")
    axes[0].set_ylabel("$ (V_{True} - V)^2 $", fontsize="large")
    axes[0].set_yscale("log")
    axes[0].set_xticks(ticks, [f"{x:,}" for x in ticks*save_skip])
    axes[0].set_xlim(ticks[1], ticks[-2])
    axes[0].legend(loc=1, prop={"weight":"bold", "size":"large"})
    
    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_to}/true_compare.pdf", format="pdf", bbox_inches="tight")


def plot_qq(plot_objs, returns, figsize=(4,4), save_to=None, alpha=0.3, markersize=10):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(returns, returns, "--", label=MC_NAME, color=COLORS[0])
    for idx, (quantile, name) in enumerate(plot_objs):
        ax.plot(returns, quantile, "h", label=name, alpha=alpha, markersize=markersize, color=COLORS[idx+1])
    ax.set_box_aspect(1)

    ax.set_xlabel(MC_NAME, fontweight="bold")
    ax.set_ylabel("Estimated Quantiles", fontweight="bold")
    ax.grid(True)
    ax.legend(prop={"weight":"bold", "size":"large"})

    for lh in ax.get_legend().legend_handles: lh.set_alpha(1)
    plt.tight_layout()
    
    if save_to is not None:
        Path(save_to).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_to}/qq_compare.pdf", format="pdf", bbox_inches="tight")


## OLD FUNCTIONS:


# def plot_kde(quantiles, params, bandwidth=0.05, is_filled=True, save_path=None):
#     from sklearn.neighbors import KernelDensity
#     for a in range(params.action_size):
#         samples = quantiles[a][:, None]
#         kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples)
        
#         x_grid = np.linspace(quantiles[a].min(), quantiles[a].max(), 10_000)[:, None]
#         log_pdf = kde.score_samples(x_grid)
#         pdf = np.exp(log_pdf)

#         plt.plot(x_grid[:,0], pdf, label=f"action: {arrow_directions[a]}")
#         if is_filled:
#             plt.fill_between(x_grid[:,0], pdf, alpha=0.2)
#         # plt.hist(quantiles[a], bins=50)
    
#     plt.title("Kernel Density Estimation")
#     plt.legend()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()


# def plot_returns(returns, bandwidth=0.01, is_filled=True, limits=None, save_path=None):
#     from sklearn.neighbors import KernelDensity
#     samples = returns[:, None]
#     kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples)
    
#     if limits is None: limits = (returns.min(), returns.max())
    
#     x_grid = np.linspace(*limits, 10_000)[:, None]
#     log_pdf = kde.score_samples(x_grid)
#     pdf = np.exp(log_pdf)
    
#     if is_filled:
#         plt.fill_between(x_grid[:,0], pdf, alpha=0.2)
#     plt.plot(x_grid[:,0], pdf)
#     plt.title("Kernel Density Estimation")
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         plt.close()


# def plot_grad(quantiles, params, is_filled=True, highlight_optimal=False, save_path=None):
#     from warnings import catch_warnings, filterwarnings
#     with catch_warnings():
#         filterwarnings("ignore", category=RuntimeWarning)
        
#         tau = ((2 * np.arange(params.n_quantiles) + 1) / (2.0 * params.n_quantiles))
#         a_max = quantiles.mean(1).argmax()
#         for a in range(quantiles.shape[0]):
#             r = quantiles[a]
#             grad = np.gradient(tau, r)
#             if (np.isnan(grad).any() or np.isinf(grad).any()): continue
#             plt.plot(r, grad, label=f"action: {arrow_directions[a]}", alpha=0.55 if highlight_optimal and a != a_max else 1.0)
#             if is_filled:
#                 plt.fill_between(r, np.gradient(tau, r), alpha=0.2)
#         plt.title("Density Estimation via Quantile Gradients")
#         plt.legend()
        
#         if save_path:
#             plt.savefig(save_path, bbox_inches="tight")
#             plt.close()


def plot_map(qtable, params, is_policy=False, V=None):
    from seaborn import heatmap, color_palette
    arrow_directions = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    
    # qtable_val_max = qtable.max(axis=1).reshape(shape)
    # qtable_best_action = qtable.argmax(1).reshape(shape)
    # qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)

    # for idx, val in enumerate(qtable_best_action.flatten()):
    #     if not np.allclose(qtable[idx], 0.0):
    #         qtable_directions[idx] = arrow_directions[val]
    # qtable_directions = qtable_directions.reshape(shape)

    fig, ax = plt.subplots(figsize=(params.shape[1], params.shape[0]))

    if is_policy:
        annot = []
        for actions in qtable:
            if max(actions) == 1 / params.nA:
                annot.append(" ")
            else:
                annot.append(arrow_directions[np.argmax(actions)])
        annot = np.asarray(annot).reshape(params.shape)
        q = np.zeros(params.shape)
        fs = "20"
    else:
        q = qtable.reshape(params.shape)
        annot=q.round(2)
        fs = "15"
    
    heatmap = heatmap(
        V.reshape(params.shape) if V is not None else q,
        annot=annot,
        fmt="",
        ax=ax,
        cmap=color_palette("Blues", as_cmap=True),
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": fs, "color": "black"},
        square=True,
    )
    heatmap.set(title="Learned Q-values\nArrows represent best action")

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")


# def plot_mean_convergence(qtables, params, save_path=None):
#     eps_n = qtables.shape[0]
#     skip = params.save_skip

#     mean_Qs = qtables.mean(1)
#     delta_Q = np.abs(mean_Qs[1:] - mean_Qs[:-1])

#     for a in range(params.action_size):
#         plt.plot(np.arange(1, eps_n) * skip, delta_Q[:, a])

#     plt.xlabel("Episode")
#     plt.ylabel("Mean Q-value change")
#     plt.title("Q-value convergence")

#     xticks = np.linspace(0, eps_n * skip, 6, dtype=int)
#     plt.xticks(xticks, [f"{x:,}" for x in xticks])  # format with commas
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()


# def plot_value_approx(y, params):
#     fig, ax = plt.subplots(figsize=(10,2))
#     ax.plot(y)
#     ax.set_yscale("log")

#     ticks = ax.get_xticks()[1:-1].astype(int)
#     ax.set_xticks(ticks, [f"{x:,}" for x in ticks*params.save_skip])
#     ax.set_xlim(ticks[0], ticks[-1])
    
#     ax.set_xlabel("Episodes", fontweight="bold")
#     ax.set_ylabel(" $ W_1(Z_{MC}^\pi(x_s), Z(x_s)) $")
#     ax.set_title("Value Distribution Approximation Error", fontweight="bold")

#     return fig, ax