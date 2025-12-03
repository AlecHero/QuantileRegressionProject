import numpy as np
from tqdm import tqdm
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
    kappa: float # Huber Kappa value
    
    nA: int # Number of possible actions
    nS: int # Number of possible states
    shape: tuple[int, int] # Size of the map (for gridworlds)


class EpsilonGreedy:
    def __init__(self, rng):
        self.rng = rng

    def choose_action(self, epsilon, action_space, state, policy):
        if self.rng.uniform(0, 1) < epsilon:
            return action_space.sample()
        else:
            return self.rng.choice(np.flatnonzero(policy[state] == policy[state].max()))


class Learner():
    def __init__(self, params, policy=None, init_theta=None):
        self.n_episodes = params.n_episodes
        self.n_runs = params.n_runs
        self.save_skip = params.save_skip
        self.seed = params.seed
        
        self.nS = params.nS
        self.nA = params.nA
        self.N = params.N
        
        self.kappa = params.kappa
        
        self.gamma = params.gamma
        self.epsilon = params.epsilon
        self.policy = policy
        self.init_theta = init_theta
        self.reset_theta()

    def update(self, s, a, r, sp):
        raise NotImplementedError

    def train(self, env, explorer, scheduler):
        self.explorer = explorer
        self.A = env.action_space
        np.random.seed(self.seed)
        table = np.zeros((self.n_runs, self.n_episodes//self.save_skip, self.nS, self.nA, self.N))
        
        with tqdm(total=self.n_runs * self.n_episodes) as pbar:
            for run in range(self.n_runs):
                self.reset_theta()
                
                for t in range(self.n_episodes):
                    self.lr = scheduler(t)
                    s, _ = env.reset()
                    a = self.explorer.choose_action(self.epsilon, self.A, s, self.get_policy())
                    
                    done = False
                    while not done:
                        sp, r, terminated, truncated, _ = env.step(a)
                        done = terminated or truncated
                        self.update(s, a, r, sp)
                        
                        s = sp
                        a = self.explorer.choose_action(self.epsilon, self.A, s, self.get_policy())
                    
                    if (t != 0 and t % self.save_skip == 0) or self.save_skip == 1:
                        table[run, t//self.save_skip] = self.theta.copy()
                    
                    pbar.set_postfix(run=run+1, t=t+1)
                    pbar.update(1)
        return table

    def reset_theta(self):
        if self.init_theta is None:
            self.theta = np.zeros((self.nS, self.nA, self.N))
        else:
            self.theta = self.init_theta.copy()
    
    def get_Q(self, s=None):
        return self.theta.mean(-1) if s is None else self.theta[s].mean(-1)

    def get_policy(self):
        return self.policy if self.policy is not None else self.get_Q()

    def get_ap(self, sp):
        raise NotImplementedError


class Qlearner(Learner):
    def __init__(self, params, policy=None, init_theta=None):
        super().__init__(params, policy=policy, init_theta=init_theta)
        self.N = 1
        self.reset_theta()
    
    def update(self, s, a, r, sp):
        if (self.epsilon == 0.0 or self.epsilon is None) and self.policy is not None:
            Ttheta = r + self.gamma *  np.dot(self.policy[sp], self.theta[sp])
        else:
            Ttheta = r + self.gamma * self.theta[sp, self.get_Q(sp).argmax()]
        self.theta[s, a] += self.lr * (Ttheta - self.theta[s, a])


class SARSA(Qlearner):
    def get_ap(self, sp):
        return self.explorer.choose_action(self.epsilon, self.A, sp, self.get_policy())

    def update(self, s, a, r, sp):
        Ttheta = r + self.gamma * self.theta[sp, self.get_ap(sp)]
        self.theta[s, a] += self.lr * (Ttheta - self.theta[s, a])


class TD(Qlearner):
    def update(self, s, a, r, sp):
        Ttheta = r + self.gamma * self.theta[sp]
        self.theta[s] += self.lr * (Ttheta - self.theta[s])


class QR(Learner):
    def __init__(self, params, policy=None, init_theta=None):
        super().__init__(params, policy=policy, init_theta=init_theta)
        self.tau = (np.arange(self.N) + 0.5) / self.N
    
    @staticmethod
    def HL_grad(u, kappa):
        return np.where(np.abs(u) <= kappa, u, kappa * np.sign(u))
    
    def _rho(self, u):
        if self.kappa == 0.0:
            return (self.tau - (u < 0).astype(float)).mean(0)
        else:
            return (np.abs(self.tau - (u < 0).astype(float)) * QR.HL_grad(u, self.kappa)).mean(0)
    
    def update(self, s, a, r, sp):
        if (self.epsilon == 0.0 or self.epsilon is None) and self.policy is not None:
            Ttheta = r + self.gamma * np.dot(self.policy[sp], self.theta[sp])
        else:
            Ttheta = r + self.gamma * self.theta[sp, self.get_Q(sp).argmax()]
        u = Ttheta[:, None] - self.theta[s, a][None, :]
        self.theta[s, a] += self.lr * self._rho(u)


class QRTD(QR):
    def update(self, s, a, r, sp):
        Ttheta = r + self.gamma * self.theta[sp]
        u = Ttheta[:, None] - self.theta[s][None, :]
        self.theta[s] += self.lr * self._rho(u)


class QR_SARSA(QR):
    def get_ap(self, sp):
        return self.explorer.choose_action(self.epsilon, self.A, sp, self.get_policy())

    def update(self, s, a, r, sp):
        Ttheta = r + self.gamma * self.theta[sp, self.get_ap(sp)]
        u = Ttheta[:, None] - self.theta[s, a][None, :]
        self.theta[s, a] += self.lr * self._rho(u)


def PolicyIteration(env, gamma=0.99, theta=1e-8):
    ## CHAT-GPT:
    nS = env.observation_space.n
    nA = env.action_space.n
    
    policy = np.ones((nS, nA)) / nA
    V = np.zeros(nS)
    P = env.unwrapped.P

    # Policy Iteration
    is_policy_stable = False
    while not is_policy_stable:
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for s in range(nS):
                if env.unwrapped._is_terminal.flatten()[s]: continue                
                v = V[s]
                V[s] = sum(policy[s,a] * sum(prob * (reward + gamma * V[next_s])
                        for prob, next_s, reward, done in P[s][a]) for a in range(nA))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        is_policy_stable = True
        for s in range(nS):
            if env.unwrapped._is_terminal.flatten()[s]: continue
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


def MonteCarlo(env, policy, gamma, s_init=None, a_init=None, total_episodes=5_000, return_steps=False, use_tqdm=True):
    from tqdm import tqdm
    returns = []
    steps = []
    for _ in tqdm(range(total_episodes), disable=not use_tqdm):
        _s, _ = env.reset()
        s = _s if s_init is None else s_init
        env.unwrapped.s = s
        G = 0.0
        discount = 1.0
        done = False
        step = 0
        _a = a_init
        
        while not done:
            if _a is not None:
                a = _a
                _a = None
            else: a = policy[s].argmax()
            sp, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G += discount * r
            discount *= gamma
            s = sp
            step += 1
        
        returns.append(G)
        steps.append(step)
    return np.asarray(returns) if not return_steps else np.asarray(steps)


class MixtureOfGaussians:
    def __init__(self, pis, mus, sigmas, rng:np.random.Generator):
        self.pis = np.array(pis)
        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.rng = rng

    def draw_samples(self, n):
        samples = np.empty(n)
        for i in range(n):
            idx = self.rng.multinomial(1, self.pis).argmax()
            samples[i] = self.rng.normal(self.mus[idx], self.sigmas[idx])
        return samples

    def pdf(self, x):
        return np.sum([pi * np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                       for pi, mu, s in zip(self.pis, self.mus, self.sigmas)], axis=0)

    def ppf(self, q, x_min=None, x_max=None, num_points=100_000):
        from scipy.interpolate import interp1d
        x_min = self.mus.min() - 5 * self.sigmas.max()
        x_max = self.mus.max() + 5 * self.sigmas.max()
        
        _cdf_x = np.linspace(x_min, x_max, num_points)
        pdf_vals = self.pdf(_cdf_x)
        
        _cdf_vals = np.cumsum(pdf_vals)
        _cdf_vals /= _cdf_vals[-1]
        
        inv_cdf = interp1d(_cdf_vals, _cdf_x, bounds_error=False, fill_value=(_cdf_x[0], _cdf_x[-1]))
        return inv_cdf(q)