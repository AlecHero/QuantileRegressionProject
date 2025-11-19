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
    # decay_rate: float # Decay rate for exponential learning rate decay
    save_skip: int # How many episodes to skip between each save
    huber_k: float # K parameter in huber loss
    learner_name: str


def run_env(env, learner, explorer, params, show_progress=False, s_init=None):
    episodes = np.arange(params.total_episodes)
    tables_run = np.zeros((params.n_runs, params.total_episodes//params.save_skip, params.state_size, params.action_size, params.n_quantiles))

    for run in range(params.n_runs):
        lr = params.learning_rate
        learner.reset_table()
        learner.set_learning_rate(lr)
        
        for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes"):
            if show_progress and episode != 0 and episode % 100 == 0:
                qtable = learner.get_qtable()
                clear_output(wait=True)
                plot_optimal_action(qtable, params)
                plot_mean_convergence(tables_run[run,:(episode//params.save_skip)].mean(-1), params)
            
            if params.lr_decay is not None and episode % params.lr_decay == 0 and episode != 0:
                lr *= 0.5
                learner.set_learning_rate(lr)
            
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
            
            if episode % params.save_skip == 0:
                tables_run[run, episode//params.save_skip] = learner.get_table()
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


def save_experiment(tables_run, params, plot_info=None, name=None):
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M")
    if name is None:
        exp_dir = Path(f"new_experiments/{params.model_name}_{params.learner_name}_{timestamp}")
    else:
        exp_dir = Path(f"new_experiments/{params.model_name}_{params.learner_name}_{name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    np.savez(exp_dir / "tables.npz", tables=tables_run)
    with open(exp_dir / "params.json", "w") as f:
        json.dump(params._asdict(), f, indent=4)
    
    if plot_info is not None:
        save_plots(tables_run, params, plot_info, exp_dir)


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


def PolicyIteration(env, gamma=0.99, theta=1e-8):
    ## CHAT-GPT:
    nS = env.observation_space.n
    nA = env.action_space.n
    
    policy = np.ones((nS, nA)) / nA
    V = np.zeros(nS)

    P = env.unwrapped.P
    try:
        desc = env.unwrapped.desc
    except: pass

    # Policy Iteration
    is_policy_stable = False
    while not is_policy_stable:
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for s in range(nS):
                try:
                    if s in env.unwrapped._cliff.flatten().nonzero()[0] or s == nS-1:
                        continue
                except: pass
                try:
                    _pos = np.unravel_index(s, desc.shape)
                    if desc[_pos] in b"GH": continue
                except: pass
                    
                v = V[s]
                V[s] = sum(policy[s,a] * sum(prob * (reward + gamma * V[next_s])
                        for prob, next_s, reward, done in P[s][a]) for a in range(nA))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        is_policy_stable = True
        for s in range(nS):
            try:
                if s in env.unwrapped._cliff.flatten().nonzero()[0] or s == nS-1:
                    continue
            except: pass
            try:
                _pos = np.unravel_index(s, desc.shape)
                if desc[_pos] in b"GH": continue
            except: pass
            old_action = np.argmax(policy[s])
            # Compute action-values
            Q_s = np.array([sum(prob * (reward + gamma * V[next_s]) for prob, next_s, reward, done in P[s][a])
                            for a in range(nA)])
            best_action = np.argmax(Q_s)
            if old_action != best_action:
                is_policy_stable = False
            # Update policy to be greedy
            policy[s] = np.eye(nA)[best_action]
    return policy, V


def MonteCarlo(env, policy, params, s_init, rng:np.random.Generator, epsilon=0.0, total_episodes=1_000, policy_eps=None):
    from tqdm import tqdm
    returns = []
    nA = params.action_size
    for _ in tqdm(range(total_episodes)):
        env.reset()
        env.unwrapped.s = s_init
        s = s_init
        G = 0.0
        discount = params.gamma
        done = False
        
        while not done:
            if rng.random() < epsilon:
                if policy_eps is not None:
                    a = rng.choice(np.flatnonzero(policy_eps[s] >= policy_eps[s].max()))
                else:
                    a = rng.integers(nA)
            else:
                a = rng.choice(np.flatnonzero(policy[s] == policy[s].max()))
            
            next_s, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G += discount * reward
            discount *= params.gamma
            s = next_s
        
        returns.append(G)
    return np.array(returns)