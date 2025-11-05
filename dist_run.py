import numpy as np
from tqdm import tqdm
from typing import NamedTuple
from IPython.display import clear_output
from analysis import plot_optimal_action

from pathlib import Path
import datetime
import json

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
    use_lr_decay: bool # Use learning rate decay 1/2 every 2_000 episodes
    decay_rate: float # Decay rate for exponential learning rate decay
    save_skip: int # How many episodes to skip between each save
    huber_k: float # K parameter in huber loss


def run_env(env, learner, explorer, params, show_progress=False):
    episodes = np.arange(params.total_episodes)
    tables_run = np.zeros((params.n_runs, params.total_episodes//params.save_skip, params.state_size, params.action_size, params.n_quantiles))

    lr = params.learning_rate
    seed_seqs = np.random.SeedSequence(params.seed).spawn(params.n_runs)
    get_seed = lambda seed_seq: int(seed_seq.generate_state(1)[0])
    
    for run in range(params.n_runs):
        learner.reset_table()
        explorer.set_rng(seed_seqs[run])
        env.reset(seed=get_seed(seed_seqs[run]))
        
        for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes"):
            if show_progress and episode != 0 and episode % 100 == 0:
                qtable = learner.get_qtable()
                clear_output(wait=True)
                plot_optimal_action(qtable, params)
                from analysis import plot_mean_convergence
                plot_mean_convergence(tables_run[run,:(episode//params.save_skip)].mean(-1), params)
                
            if params.use_lr_decay and episode % 2_000 == 0 and episode != 0:
                lr *= 0.5
                learner.set_learning_rate(lr)
            elif params.decay_rate is not None:
                lr = params.learning_rate * np.exp(-params.decay_rate * episode)
                learner.set_learning_rate(lr)
            
            state, _ = env.reset()
            done = False

            while not done:
                qtable = learner.get_qtable() # Get the mean over the quantiles to get a Q-table for action selection
                action = explorer.choose_action(action_space=env.action_space, state=state, qtable=qtable)

                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                learner.update(state, action, reward, new_state)
                state = new_state
            
            if episode % params.save_skip == 0:
                tables_run[run][episode//params.save_skip] = learner.get_table().copy()
    return tables_run


# def run_policy(env, learner, policy, params):
#     tables_run = np.zeros(( params.n_runs,
#                             params.total_episodes//params.save_skip,
#                             params.state_size,
#                             params.action_size,
#                             params.n_quantiles))

#     for run in range(params.n_runs):
#         lr = params.learning_rate
#         learner.reset_table()
#         for episode in tqdm(range(params.total_episodes), desc=f"Run {run}/{params.n_runs} - Episodes"):
#             if params.use_lr_decay and episode % 2_000 == 0 and episode != 0:
#                 lr *= 0.5
#                 learner.set_learning_rate(lr)
            
#             state, _ = env.reset()
#             done = False
#             while not done:
#                 action = policy[state]
#                 new_state, reward, terminated, truncated, _ = env.step(action)
#                 done = terminated or truncated
#                 learner.update(state, action, reward, new_state)
#                 state = new_state
            
#             if episode % params.save_skip == 0:
#                 tables_run[run][episode//params.save_skip] = learner.get_table().copy()

#     return tables_run


def run_sweep(env, learner, params):
    tables_run = np.zeros((params.n_runs, params.total_episodes//params.save_skip, params.state_size, params.action_size, params.n_quantiles))
    valid_states = [np.ravel_multi_index(coord, params.map_size) for coord in np.argwhere(~env.unwrapped._cliff)][:-1]

    lr = params.learning_rate
    for run in range(params.n_runs):
        for episode in tqdm(range(params.total_episodes)):
            if params.use_lr_decay and episode % 2_000 == 0 and episode != 0:
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


def save_experiment(tables, params):
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M")
    exp_dir = Path(f"experiments/{params.model_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    np.savez(exp_dir / "tables.npz", tables=tables)
    with open(exp_dir / "params.json", "w") as f:
        json.dump(params._asdict(), f, indent=4)


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

    return params, tables