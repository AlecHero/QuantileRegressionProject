# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex12.sarsa_lambda_agent import SarsaLambdaAgent
from irlc.gridworld.gridworld_environments import OpenGridEnvironment, BookGridEnvironment, SuttonCornerGridEnvironment, SuttonMazeEnvironment
from irlc import train, savepdf
from irlc.ex12.sarsa_lambda_open import keyboard_play
import matplotlib.pyplot as plt
from irlc.ex11.q_agent import QAgent
from irlc.ex11.sarsa_agent import SarsaAgent
from irlc.ex11.nstep_sarsa_agent import SarsaNAgent
from irlc.ex10.mc_agent import MCAgent
from irlc.ex10.mc_evaluate import MCEvaluationAgent
from irlc.exam_tabular_examples.sarsa_nstep_delay import SarsaDelayNAgent
from irlc.exam_tabular_examples.sarsa_lambda_delay import SarsaLambdaDelayAgent
from irlc import interactive

def open_snapshop(Agent, method_label="Unknown method", file_name=None, alpha=0.5, autoplay=False, **kwargs):
    env = OpenGridEnvironment(render_mode='human')
    agent = Agent(env, gamma=0.99, epsilon=0.1, alpha=alpha, **kwargs)
    agent.label =method_label
    print("Running", agent)
    env, agent = interactive(env, agent, autoplay=autoplay)
    train(env, agent, num_episodes=3)
    if file_name is not None:
        env.plot()
        plt.title(method_label)
        savepdf("exam_tabular_"+file_name)
    env.close()

if __name__ == "__main__":
    """ All simulations run using gamma=0.99, epsilon=0.1 and alpha=0.5 (when applicable). """

    """ The following lines will show all the agents using automatic play. It is used to generate screenshots. 
    Uncomment to go to interactive play. """
    # import numpy as np
    # np.random.seed(42)
    # env = SuttonMazeEnvironment(living_reward=-2, render_mode='human')
    # agent = MCAgent(env, alpha=0.8, epsilon=0, gamma=0.4)
    # env, agent = interactive(env, agent)
    # train(env, agent, num_episodes=2)

    open_snapshop(MCAgent, "Monte-Carlo control (first visit)", file_name="mc_first", alpha=None)
    open_snapshop(MCAgent, "Monte-Carlo control (every visit)", file_name="mc_every", alpha=None, first_visit=False)
    open_snapshop(SarsaAgent, "Sarsa", file_name="sarsa")
    open_snapshop(SarsaNAgent, "n-step Sarsa (n=8)", file_name="sarsa_n8", n=8)
    open_snapshop(QAgent, "Q-learning", file_name="q_learning")
    open_snapshop(SarsaLambdaAgent, "Sarsa(Lambda)", file_name="sarsa_lambda")
    open_snapshop(MCEvaluationAgent, "Monte-Carlo value-estimation (first visit)", file_name="mc_evaluation_first", alpha=None)
    open_snapshop(MCEvaluationAgent, "Monte-Carlo value-estimation (every visit)", file_name="mc_evaluation_every", first_visit=False)

    """ MC-methods for value estimation. This is the upgraded demo which also shows the number of times 
    a state has been visited in the value-estimation algorithm. """
    keyboard_play(MCEvaluationAgent, "Monte-Carlo value-estimation (first visit)", alpha=None)
    keyboard_play(MCEvaluationAgent, "Monte-Carlo value-estimation (every visit)", alpha=None, first_visit=False)

    """ Control methods: 
    Play with the agents (using keyboard input) """
    keyboard_play(MCAgent, "Monte-Carlo control (first visit)", alpha=None)
    keyboard_play(MCAgent, "Monte-Carlo control (every visit)", alpha=None, first_visit=False)
    keyboard_play(QAgent, "Q-learning")

    """ These agents also accept keyboard input, but they are not guaranteed to update the Q-values correctly because the next state A' (in Suttons notation) 
    is generated in the train() method; i.e. we cannot easily overwrite it using the keyboard. I have included them for completeness, but 
    be a little careful with them. """
    keyboard_play(SarsaAgent, "Sarsa")
    keyboard_play(SarsaNAgent, "n-step Sarsa (n=8)", n=8)
    keyboard_play(SarsaLambdaAgent, "Sarsa(Lambda)")

    """ Bonus keyboard input agents: These agents implement the same methods as their counterparts above, however they 'wait' with updating
    Q(S_t, A_t) until time t+1 when the (actual) next action A_{t+1} is available. This means that when they are used in conjunction with keyboard inputs,
    the Q-values will be updated correctly since we can actually set A_{t+1} equal to the keyboard input.
    This also mean the updates to the Q-values appear to lag one step behind the methods above. 
    I have included them in the case some find it useful to test the Q-values using the keyboard, 
    however, the implementations/delay-idea is not part of the exam pensum: only use them if you find them useful for studying, and otherwise just rely on the 
    description of the methods in the lecture material. 
    """
    keyboard_play(SarsaDelayNAgent, "Sarsa (delayed)", n=1) # We use that Sarsa is equal to n-step sarsa with n=1.
    keyboard_play(SarsaDelayNAgent, "n-step Sarsa (n=8, delayed)", n=8)
    keyboard_play(SarsaLambdaDelayAgent, "Sarsa(Lambda) (delayed)")
