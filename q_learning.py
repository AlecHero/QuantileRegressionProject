import numpy as np


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
        return self.qtable

    def get_table(self):
        return self.get_qtable()

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
    
    def set_rng(self, seed_seq):
        self.rng = np.random.default_rng(seed_seq)