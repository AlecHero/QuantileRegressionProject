import numpy as np

def huber(u, k=1.0):
    return np.where(np.abs(u) < k, 0.5 * np.power(u, 2), k * (np.abs(u) - 0.5 * k))

def du_huber(u, k=1.0):
    return np.where(np.abs(u) <= k, u / k, k * np.sign(u))


class Qlearning:
    def __init__(self, params):
        self.state_size = params.state_size
        self.action_size = params.action_size
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.reset_table()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        self.qtable[state, action] += self.learning_rate * delta

    def reset_table(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size, 1))

    def get_qtable(self):
        return self.qtable.copy()

    def get_table(self):
        return self.get_qtable()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


class QuantileRegression():
    def __init__(self, params, init_theta=None):
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.state_size = params.state_size
        self.action_size = params.action_size
        self.n_quantiles = params.n_quantiles
        self.k = params.huber_k
        self.init_theta = init_theta.copy()
        self.reset_table()
        self.tau = (np.arange(self.n_quantiles) + 0.5) / self.n_quantiles

    def _rho(self, u):
        if self.k==0.0:
            return -(self.tau - (u < 0)).mean(0)
        else:
            return -np.abs((self.tau - (u < 0).astype(float)).mean(0)) * du_huber(u, self.k).mean(0)

    def update(self, state, action, reward, new_state):
        pred_quantiles = self.theta[state, action]
        greedy_action = self.theta[new_state].mean(1).argmax()
        target_quantiles = reward + self.gamma * self.theta[new_state, greedy_action]
        
        u = target_quantiles[:, None] - pred_quantiles[None, :]
        self.theta[state, action] -= self.learning_rate * self._rho(u)

    def reset_table(self):
        """Reset the theta values."""
        if self.init_theta is not None:
            self.theta = self.init_theta.copy()
        else:
            self.theta = np.zeros((self.state_size, self.action_size, self.n_quantiles))

    def get_qtable(self):
        return self.theta.mean(2)
    
    def get_table(self):
        return self.theta.copy()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


class EpsilonGreedy:
    def __init__(self, epsilon, rng, policy=None):
        self.epsilon = epsilon
        self.rng = rng
        self.policy = policy

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # Find the indices where the Q-value equals the maximum value
            # Choose a random action from the indices where the Q-value is maximum
            if self.policy is None:
                max_ids = np.flatnonzero(qtable[state] == qtable[state].max())
            else:
                max_ids = np.flatnonzero(self.policy[state] == self.policy[state].max())
            action = self.rng.choice(max_ids)
        return action


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