import numpy as np

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