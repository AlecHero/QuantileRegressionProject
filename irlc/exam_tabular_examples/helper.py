# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import interactive, train


def keyboard_play_value(env, agent, method_label='MC',  q=False):
    env, agent = interactive(env, agent)
    agent.label = method_label
    agent.label = 'MC (first visit)'
    env.view_mode = 1 # Set value-function view-mode.
    train(env, agent, num_episodes=100)
    env.close()
