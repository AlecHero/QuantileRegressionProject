import numpy as np
from tqdm import tqdm
from typing import NamedTuple
from IPython.display import clear_output
from analysis import plot_optimal_action, plot_mean_convergence

from pathlib import Path
import datetime
import json

from learners import Qlearning, QuantileRegression

class Params(NamedTuple):
    model_name: str
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    seed: int  # Define a seed so that we get reproducible results
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    map_size: tuple[int, int] # Size of the map (for gridworlds)
    n_quantiles: int  # Number of quantiles
    lr_decay: int # Use learning rate decay 1/2 every 2_000 episodes
    save_skip: int # How many episodes to skip between each save
    huber_k: float # K parameter in huber loss
    learner_name: str


def run_env(env, learner, explorer, params, show_progress=False, s_init=None):
    episodes = np.arange(params.total_episodes)
    tables_run = np.zeros((params.n_runs, params.total_episodes//params.save_skip, params.state_size, params.action_size, params.n_quantiles))

    with tqdm(total=params.n_runs * params.total_episodes) as pbar:
        for run in range(params.n_runs):
            lr = params.learning_rate
            learner.reset_table()
            learner.set_learning_rate(lr)
            
            for ep in episodes:
                # if params.lr_decay is not None and ep % params.lr_decay == 0 and ep != 0:
                #     lr *= 0.5
                #     learner.set_learning_rate(lr)
                learner.set_learning_rate(params.learning_rate * 1 / np.sqrt(ep+1))
                
                state, _ = env.reset()
                if s_init is not None:
                    env.unwrapped.s = s_init
                done = False
                while not done:
                    action = explorer.choose_action(action_space=env.action_space, state=state, qtable=learner.get_qtable())
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    learner.update(state, action, reward, new_state)
                    state = new_state
                
                if ep % params.save_skip == 0:
                    tables_run[run, ep//params.save_skip] = learner.get_table()
                
                pbar.set_postfix(run=run+1, ep=ep+1)
                pbar.update(1)
    return tables_run


def run_sweep(env, learner, params):
    tables_run = np.zeros((params.n_runs, params.total_episodes//params.save_skip, params.state_size, params.action_size, params.n_quantiles))
    valid_states = [np.ravel_multi_index(coord, params.map_size) for coord in np.argwhere(~env.unwrapped._cliff)][:-1]

    lr = params.learning_rate
    for run in range(params.n_runs):
        for episode in tqdm(range(params.total_episodes)):
            if params.lr_decay is not None and episode % params.lr_decay == 0 and episode != 0:
                lr *= 0.5
                learner.set_learning_rate(lr)
            for state in valid_states:
                for action in range(params.action_size):
                    env.reset()
                    env.unwrapped.s = state
                    next_state, reward, _, _, _ = env.step(action)
                    learner.update(state, action, reward, next_state)
            if episode % params.save_skip == 0:
                tables_run[run][episode//params.save_skip] = learner.get_table().copy()
    return tables_run


def save_experiment(tables_run, params, save_to="experiments", name=None):
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M")
    exp_dir = Path(f"{save_to}/{params.model_name}_{params.learner_name}_{timestamp if name is None else name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    np.savez(exp_dir / "tables.npz", tables=tables_run)
    with open(exp_dir / "params.json", "w") as f:
        json.dump(params._asdict(), f, indent=4)


def save_plots(tables_run, params, plot_info, exp_dir):
    from analysis import plot_grad, plot_returns, plot_mean_convergence, plot_kde
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    tables = tables_run.mean(0)
    qtables = tables.mean(-1) if params.n_quantiles > 1 else tables[-1]
    
    try: plot_mean_convergence(qtables[:,plot_info["state"]:plot_info["state"]+1], params, save_path=plots_dir / f"convergence_{plot_info['state']}.png")
    except: pass
    plot_mean_convergence(qtables, params, save_path=plots_dir / f"convergence_mean.png")

    try: plot_grad(tables[-1, plot_info["state"]], params, is_filled=plot_info["is_filled"], save_path=plots_dir / f"grad_{plot_info['state']}.png")
    # except: plot_kde(tables[-1, plot_info["state"]], params, is_filled=plot_info["is_filled"], bandwidth=plot_info["bandwidth"], save_path=plots_dir / f"kde_{plot_info['state']}.png")
    except: pass

    all_actions = tables[-1, plot_info["state"]].reshape(-1).copy()
    all_actions.sort()
    plot_returns(all_actions, bandwidth=plot_info["bandwidth"], is_filled=plot_info["is_filled"], save_path=plots_dir / f"all_actions_{plot_info['state']}.png")
    
    plot_optimal_action(qtables[-1], params, save_path=plots_dir / f"optimal_action.png")


def load_experiment(experiment_name=None):
    if experiment_name is None:
        import os
        experiments_dir = "experiments/"
        experiment_name = max(
            (d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))),
            key=lambda d: os.path.getctime(os.path.join(experiments_dir, d))
        )
    
    exp_dir = Path(f"experiments/{experiment_name}")
    params = Params(**json.load(open(exp_dir / "params.json", "r")))
    
    tables_file = exp_dir / "tables.npz"
    if tables_file.exists():
        tables = np.load(tables_file)["tables"]

    if len(tables.shape) == 5:
        tables = tables.mean(0)
    qtables = tables.mean(-1)
    return tables, qtables, params