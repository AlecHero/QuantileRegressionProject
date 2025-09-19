import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import NamedTuple
from dist_q import Qlearning, EpsilonGreedy, QuantileRegression


class Params(NamedTuple):
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
    # proba_frozen: float  # Probability that a tile is frozen
    n_quantiles: int  # Number of quantiles


def run_env_qlearning(env, explorer, params):
    learner = Qlearning(params.learning_rate, params.gamma, params.state_size, params.action_size)
    
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(action_space=env.action_space, state=state, qtable=learner.qtable)

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(state, action, reward, new_state)

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions


def run_env_quantile(env, explorer, params):
    learner = QuantileRegression(params.learning_rate, params.gamma, params.state_size, params.action_size, params.n_quantiles)
    
    rewards = np.zeros((params.total_episodes, 
                        params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    thetas = np.zeros((params.n_runs, params.state_size, params.action_size, params.n_quantiles))
    all_states = []
    all_actions = []

    init_lr = params.learning_rate
    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_theta()  # Reset the Q-table between runs
        params = params._replace(learning_rate = init_lr)
        
        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            if episode % (params.total_episodes//5) == 0:
                params = params._replace(learning_rate = params.learning_rate * 0.5)
                learner.set_learning_rate(params.learning_rate)
            
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                qtable = learner.theta.mean(2) # Get the mean over the quantiles to get a Q-table for action selection
                action = explorer.choose_action(action_space=env.action_space, state=state, qtable=qtable)

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                learner.update(state, action, reward, new_state)

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        thetas[run, :, :] = learner.theta

    return rewards, steps, episodes, thetas, all_states, all_actions


def postprocess(episodes, params, rewards, steps):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    return res, st