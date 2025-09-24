import numpy as np

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
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
        return self.qtable

    def get_table(self):
        return self.get_qtable()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

def huber(x, k=1.0):
    return np.where(np.abs(x) < k, 0.5 * np.power(x, 2), k * (np.abs(x) - 0.5 * k))


class QuantileRegression:
    def __init__(self, learning_rate, gamma, state_size, action_size, n_quantiles):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.n_quantiles = n_quantiles
        self.reset_table()
        
        self.tau = ((2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles))

    def update(self, state, action, reward, new_state):
        delta = (
            reward
            + self.gamma * self.theta[new_state, self.theta[new_state].mean(1).argmax()] # Ttheta
            - self.theta[state, action]
        )
        self.theta[state, action] += self.learning_rate * (self.tau - (delta < 0)) * huber(delta)

    def reset_table(self):
        """Reset the theta values."""
        self.theta = np.zeros((self.state_size, self.action_size, self.n_quantiles))

    def get_qtable(self):
        return self.theta.mean(2)
    
    def get_table(self):
        return self.theta

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


class EpsilonGreedy:
    def __init__(self, epsilon, rng):
        self.epsilon = epsilon
        self.rng = rng

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
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = self.rng.choice(max_ids)
        return action