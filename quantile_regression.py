import numpy as np

def huber(u, k=1.0):
    if k == 0.0: return 1.0
    return np.where(np.abs(u) < k, 0.5 * np.power(u, 2), k * (np.abs(u) - 0.5 * k))

def du_huber(u, k=1.0):
    if k == 0.0: return 1.0
    return np.where(np.abs(u) <= k, u, k * np.sign(u))

class QuantileRegression:
    def __init__(self, params):
        self.learning_rate = params.learning_rate
        self.gamma = params.gamma
        self.state_size = params.state_size
        self.action_size = params.action_size
        self.n_quantiles = params.n_quantiles
        self.k = params.huber_k
        self.reset_table()
        self.tau = ((2 * np.arange(self.n_quantiles) + 1) / (2.0 * self.n_quantiles))

    def update(self, state, action, reward, new_state):
        pred_quantiles = self.theta[state, action]
        greedy_action = self.theta[new_state].mean(1).argmax()
        target_quantiles = reward + self.gamma * self.theta[new_state, greedy_action]
        
        u = target_quantiles[None, :] - pred_quantiles[:, None]
        grad_loss = -np.abs(self.tau[:, None] - (u < 0)) * du_huber(u, self.k)
        self.theta[state, action] -= self.learning_rate * grad_loss.mean(1)

    def reset_table(self):
        """Reset the theta values."""
        self.theta = np.zeros((self.state_size, self.action_size, self.n_quantiles))

    def get_qtable(self):
        return self.theta.mean(2)
    
    def get_table(self):
        return self.theta

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate