import numpy as np

def huber(u, k=1.0):
    return np.where(np.abs(u) < k, 0.5 * np.power(u, 2), k * (np.abs(u) - 0.5 * k))

def du_huber(u, k=1.0):
    return np.where(np.abs(u) <= k, u, k * np.sign(u))


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
    def __init__(self, params):
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.state_size = params.state_size
        self.action_size = params.action_size
        self.n_quantiles = params.n_quantiles
        self.k = params.huber_k
        self.reset_table()
        self.tau = (np.arange(self.n_quantiles) + 0.5) / self.n_quantiles

    def _rho(self, u):
        if self.k==0.0:
            return -(self.tau - (u < 0))
        else:
            return -np.abs(self.tau - (u < 0)) * du_huber(u, k=self.k)

    def update(self, state, action, reward, new_state):
        pred_quantiles = self.theta[state, action]
        greedy_action = self.theta[new_state].mean(1).argmax()
        target_quantiles = reward + self.gamma * self.theta[new_state, greedy_action]
        
        u = (target_quantiles[:, None] - pred_quantiles[None, :]).mean(0)
        self.theta[state, action] -= self.learning_rate * self._rho(u)
        
        # _q = self._rho(u)
        # for i in range(self.n_quantiles):
        #     self.theta[state, action][i] -= self.learning_rate * _q[i].mean()

    def reset_table(self):
        """Reset the theta values."""
        self.theta = np.full((self.state_size, self.action_size, self.n_quantiles), 1.0/self.n_quantiles, dtype=float)

    def get_qtable(self):
        return self.theta.mean(2)
    
    def get_table(self):
        return self.theta.copy()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate