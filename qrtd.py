from tqdm import tqdm
import numpy as np
from typing import NamedTuple


class Params(NamedTuple):
    model_name: str
    n_runs: int  # Number of runs
    n_episodes: int  # Total episodes
    save_skip: int # How many episodes to skip between each save
    lr_decay: int # Half lr every lr_decay episodes
    seed: int # Define a seed so that we get reproducible results

    N: int # Number of quantiles
    alpha: float # Learning rate
    gamma: float # Discounting rate
    epsilon: float # Exploration probability
    
    nA: int # Number of possible actions
    nS: int # Number of possible states
    shape: tuple[int, int] # Size of the map (for gridworlds)


def rho(tau, u, kappa):
    if kappa == 0:
        return tau - (u < 0)
    else:
        return np.abs(tau - (u < 0)) * np.where(np.abs(u) <= kappa, u, kappa * np.sign(u))


def train_qrtd(env, policy, params, kappa=0, is_td=False):
    tau = (np.arange(params.N) + 0.5) / params.N
    nS = int(env.observation_space.n)
    table = np.zeros((params.n_runs, params.n_episodes//params.save_skip, nS, params.N))
    
    with tqdm(total=params.n_runs * params.n_episodes) as pbar:
        for run in range(params.n_runs):
            theta = np.zeros((nS, params.N))
            
            for ep in range(params.n_episodes):
                lr = params.alpha**((1 + ep) // 2000)
                s, _ = env.reset()
                done = False
                while not done:
                    sp, r, terminated, truncated, _ = env.step(policy[s].argmax())
                    done = terminated or truncated
                    Ttheta = (r + params.gamma * theta[sp])
                    
                    if is_td:
                        theta[s] += lr * (Ttheta - theta[s])
                    else:
                        u = Ttheta[:, None] - theta[s][None, :]
                        theta[s] += lr * rho(tau, u, kappa).mean(0)
                    s = sp
                
                if ep % params.save_skip == 0:
                    table[run, ep//params.save_skip] = theta.copy()
                    pbar.set_postfix(run=run+1, ep=ep+1)
                    pbar.update(params.save_skip)
    return table