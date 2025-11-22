# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (Freely available online).
"""
from irlc import savepdf
import matplotlib.pyplot as plt
from irlc.ex09.value_iteration import value_iteration
from irlc.ex09.mdp import MDP

class GamblerEnv(MDP):
    """
    The gamler problem (see description given in (SB18, Example 4.3))

    See the MDP class for more information about the methods. In summary:
    > the state is the amount of money you have. if state = goal or state = 0 the game ends (use this for is_terminal)
    > A are the available actions (a list). Note that these depends on the state; see below or example for details.
    > Psr are the transitions (see MDP class for documentation)
    """
    def __init__(self, goal=100, p_heads=0.4):
        super().__init__(initial_state=goal//2)
        self.goal = goal
        self.p_heads = p_heads

    def is_terminal(self, state): 
        """ Implement if the state is terminal (0 or self.goal) """
        return state == 0 or state == self.goal

    def A(self, s):  
        """ Action is the amount you choose to gamle.
        You can gamble from 0 and up to the amount of money you have (state),
        but not so much you will exceed the goal amount (see (SB18) for details).
        In other words, return this as a list, and the number of elements should depend on the state s. """
        return range(0, min(s, self.goal - s) + 1)

    def Psr(self, s, a):  
        """ Implement transition probabilities here. 
        the reward is 1 if you win (obtain goal amount) and otherwise 0. Remember the format should
         return a dictionary with entries:
        > { (sp, r) : probability }
        
        You can see the small-gridworld example (see exercise description) for an example of how to use this function, 
        but now you should keep in mind that since you can win (or not) the dictionary you return should have two entries:
        one with a probability of self.p_heads (winning) and one with a probability of 1-self.p_heads (loosing). 
        """
        outcome_dict = {(s + a, (s + a) == self.goal): self.p_heads,
                        (s - a, 0): 1-self.p_heads}
        return outcome_dict

def gambler():
    """
    Gambler's problem from (SB18, Example 4.3)
    """
    mdp = GamblerEnv(p_heads=0.4)
    pi, V = value_iteration(mdp, gamma=1., theta=1e-11)

    V = [V[s] for s in mdp.states]
    plt.bar(mdp.states, V)
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.title('Final value function (expected return) vs State (Capital)')
    plt.grid()
    savepdf("gambler_valuefunction")
    plt.show()

    y = [pi[s] for s in mdp.nonterminal_states]
    plt.bar(mdp.nonterminal_states, y, align='center', alpha=0.5)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.title('Capital vs Final Policy')
    plt.grid()
    savepdf("gambler_policy")
    plt.show()


if __name__ == "__main__":

    gambler()
