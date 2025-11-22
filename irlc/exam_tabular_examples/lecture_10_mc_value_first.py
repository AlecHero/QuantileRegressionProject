# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.ex10.mc_evaluate import MCEvaluationAgent
from irlc import interactive, train

if __name__ == "__main__":
    env = BookGridEnvironment(view_mode=1, render_mode='human')
    agent = MCEvaluationAgent(env, gamma=.9, alpha=0.4)
    agent.label = 'MC (first visit)'
    env, agent = interactive(env, agent)
    env.view_mode = 1 # Automatically set value-function view-mode.
    train(env, agent, num_episodes=100)
    env.close()
