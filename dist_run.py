import numpy as np
from tqdm import tqdm
from typing import NamedTuple

from pathlib import Path
import datetime
import json

class Params(NamedTuple):
    model_name: str
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    n_quantiles: int  # Number of quantiles
    use_lr_decay: bool # Use learning rate decay 1/2 every 2_000 episodes
    save_skip: int # How many episodes to skip between each save
    huber_k: float # K parameter in huber loss


def run_env(env, learner, explorer, params):
    episodes = np.arange(params.total_episodes)
    all_tables = np.zeros((params.n_runs, params.total_episodes//params.save_skip, params.state_size, params.action_size, params.n_quantiles))

    lr = params.learning_rate
    seeds = np.random.SeedSequence(params.seed).spawn(params.n_runs)
    for run in range(params.n_runs):
        learner.reset_table()
        for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes"):
            if params.use_lr_decay and episode % 2_000 == 0:
                lr *= 0.5
                learner.set_learning_rate(lr)
            
            state = env.reset(seed=seeds[run])[0]
            done = False

            while not done:
                qtable = learner.get_qtable() # Get the mean over the quantiles to get a Q-table for action selection
                action = explorer.choose_action(action_space=env.action_space, state=state, qtable=qtable)

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                learner.update(state, action, reward, new_state)
                state = new_state
            
            if episode % params.save_skip == 0:
                all_tables[run][episode//params.save_skip] = learner.get_table()
    return all_tables

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