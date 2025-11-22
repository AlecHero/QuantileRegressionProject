from irlc.exam.exam2024spring.mdp import MDP
from irlc.exam.exam2024spring.policy_evaluation import policy_evaluation
from irlc.exam.exam2024spring.value_iteration import value_iteration

class GamblerEnv(MDP):
    """
    The gamler problem (see description given in (SB18, Example 4.3))

    See the MDP class for more information about the methods. In summary:
    > the state is the amount of money you have. if state = goal or state = 0 the game ends (use this for is_terminal)
    > A are the available actions (a list). Note that these depends on the state; see below or example for details.
    > Psr are the transitions (see MDP class for documentation)
    """
    def __init__(self, r_airbnb, always_airbnb=False):
        super().__init__(initial_state=0)
        self.pwin = 0.45
        self.r_airbnb = r_airbnb
        self.always_airbnb = always_airbnb
        

    # def initial_state_distribution(self):
    #     return {0:1, 1:0}

    def is_terminal(self, s):
        return False

    def A(self, s):
        """ Action is the amount you choose to gamle.
        You can gamble from 0 and up to the amount of money you have (state),
        but not so much you will exceed the goal amount (see (SB18) for details).
        In other words, return this as a list, and the number of elements should depend on the state s. """
        if s == 1:
            return {2}
        elif s == 0:
            if self.always_airbnb:
                return {0}
            else:
                return {0,1}
    
    # def nonterminal_states(self):
    #     self._nonterminal_states = {0,1}
    #     return self._nonterminal_states

    def Psr(self, s, a):  
        """ Implement transition probabilities here. 
        the reward is 1 if you win (obtain goal amount) and otherwise 0. Remember the format should
         return a dictionary with entries:
        > { (sp, r) : probability }
        
        You can see the small-gridworld example (see exercise description) for an example of how to use this function, 
        but now you should keep in mind that since you can win (or not) the dictionary you return should have two entries:
        one with a probability of self.p_heads (winning) and one with a probability of 1-self.p_heads (loosing). 
        """
        # print(s, a)
        if s == 0:        
            if a == 0: # if airbnb
                return {(0, self.r_airbnb): 1}
            
            if a == 1: # if gamble
                outcome_dict = {(0, 2): self.pwin,
                                (1, 0): 1-self.pwin}
                return outcome_dict
        else:
            if a == 2: # if lost house
                return {(1,0): 1}

def a_always_airbnb(r_airbnb : float, gamma : float) -> float:
    mdp = GamblerEnv(r_airbnb, always_airbnb=False)
    pi0 = {0: {0:1, 1:0}, 1:{2:1}}
    V = policy_evaluation(pi0, mdp, gamma)
    return V[0]

def b_random_decisions(r_airbnb : float, gamma : float) -> float:
    mdp = GamblerEnv(r_airbnb, always_airbnb=False)
    pi0 = {s: {a: 1/len(mdp.A(s)) for a in mdp.A(s)} for s in range(0,2)}
    V = policy_evaluation(pi0, mdp, gamma)
    return V[0]

def c_is_it_better_to_gamble(r_airbnb : float, gamma : float) -> bool:
    # mdp = GamblerEnv(r_airbnb, always_airbnb=False)
    # pi0 = {s: {a: 1/len(mdp.A(s)) for a in mdp.A(s) } for s in mdp.nonterminal_states }
    
    # V = policy_evaluation(pi0, mdp, gamma)
    # print(V)
    return 0#better_to_gamble

if __name__ == "__main__":
    print("a) The expected return is approximately 1, your result:", a_always_airbnb(r_airbnb=0.01, gamma=0.99))
    print("b) The expected return is approximately 1.612, your result:", b_random_decisions(r_airbnb=0.01, gamma=0.99))
    print("c1) In this case, you should return False as it is better to AirBnB, your result:", c_is_it_better_to_gamble(r_airbnb=0.02, gamma=0.99))
    print("c2) In this case, you should return True as it is better to gamble, your result:", c_is_it_better_to_gamble(r_airbnb=0.01, gamma=0.99))
