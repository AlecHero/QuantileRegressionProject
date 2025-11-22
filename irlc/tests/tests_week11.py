# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report, cache
import numpy as np
from irlc import train
import irlc.ex10.envs
import gymnasium as gym
from irlc.tests.tests_week08 import train_recording
from irlc.tests.tests_week10 import TD0Question, MCAgentQuestion


# This problem no longer exists.
# class NStepSarseEvaluationQuestion(TD0Question):
#     """ Test of TD-n evaluation agent """
#     # class EvaluateTabular(VExperienceItem):
#     #     title = "Value-function test"
#     gamma = 0.8
#     def get_env_agent(self):
#         envn = "SmallGridworld-v0"
#         from irlc.ex11.nstep_td_evaluate import TDnValueAgent
#         env = gym.make(envn)
#         agent = TDnValueAgent(env, gamma=self.gamma, n=5)
#         return env, agent



class QAgentQuestion(MCAgentQuestion):
    """ Test of Q Agent """
    # class EvaluateTabular(QExperienceItem):
    #     title = "Q-value test"

    def get_env_agent(self):
        from irlc.ex11.q_agent import QAgent
        env = gym.make("SmallGridworld-v0")
        agent = QAgent(env, gamma=.8)
        return env, agent


# class LinearWeightVectorTest(UTestCase):



# class LinearValueFunctionTest(LinearWeightVectorTest):
#     title = "Linear value-function test"
#     def compute_answer_print(self):
#         trajectories, Q = self.precomputed_payload()
#         env, agent = self.get_env_agent()
#         train_recording(env, agent, trajectories)
#         self.Q = Q
#         self.question.agent = agent
#         vfun = [agent.Q[s,a] for s, a in zip(trajectories[0].state, trajectories[0].action)]
#         return vfun

# class TabularAgentStub(UTestCase):
#
#     pass

class TabularAgentStub(UTestCase):
    """ Average return over many simulated episodes """
    gamma = 0.95
    epsilon = 0.2
    tol = 0.1
    tol_qs = 0.3

    def get_env(self):
        return gym.make("SmallGridworld-v0")

    def get_env_agent(self):
        raise NotImplementedError()
        # from irlc.ex11.sarsa_agent import SarsaAgent
        # agent = SarsaAgent(self.get_env(), gamma=self.gamma)
        # return agent.env, agent

    def get_trained_agent(self):
        env, agent = self.get_env_agent()
        stats, _ = train(env, agent, num_episodes=9000)
        return agent, stats

    def chk_accumulated_reward(self):
        agent, stats = self.get_trained_agent()
        s0, _ = agent.env.reset()
        actions, qs = agent.Q.get_Qs(s0)
        print("Tolerance is", self.tol_qs)
        self.assertL2(qs, tol=self.tol_qs)
        self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=self.tol)

    # def test_accumulated_reward(self):
    #     env, agent = self.get_env_agent()
    #     stats, _ = train(env, agent, num_episodes=5000)
    #     s = env.reset()
    #     actions, qs = agent.Q.get_Qs(s)
    #     self.assertL2(qs, tol=0.3)
    #     self.assertL2(np.mean([s['Accumulated Reward'] for s in stats]), tol=self.tol)

class SarsaQuestion(TabularAgentStub):


    def get_env_agent(self):
        from irlc.ex11.sarsa_agent import SarsaAgent
        agent = SarsaAgent(self.get_env(), gamma=self.gamma)
        return agent.env, agent

    def test_accumulated_reward(self):
        self.tol_qs = 2.7 # Got 2.65 in one run.
        self.chk_accumulated_reward()


class NStepSarsaQuestion(TabularAgentStub):
    title = "N-step Sarsa"
    # class SarsaReturnItem(SarsaQuestion):
    def get_env_agent(self):
        from irlc.ex11.nstep_sarsa_agent import SarsaNAgent
        agent = SarsaNAgent(self.get_env(), gamma=self.gamma, n=5)
        return agent.env, agent

    def test_accumulated_reward(self):
        self.tol_qs = 2.7
        self.chk_accumulated_reward()


class LinearAgentStub(UTestCase):
    # class LinearExperienceItem(LinearWeightVectorTest):
    tol = 1e-6
    # title = "Linear sarsa agent"
    alpha = 0.08
    num_episodes = 300
    # title = "Weight-vector test"
    # testfun = QPrintItem.assertL2
    gamma = 0.8
    tol_w = 1e-5


    def get_env_agent(self):
        raise NotImplementedError()

    def get_env(self):
        return gym.make("MountainCar500-v0")

    # def get_env_agent(self):
    #     return None, None

    @cache
    def compute_trajectories(self):
        env, agent = self.get_env_agent()
        _, trajectories = train(env, agent, return_trajectory=True, num_episodes=1, max_steps=100)
        return trajectories, agent.Q.w

    def chk_Q_weight_vector_w(self):
        trajectories, w = self.compute_trajectories()
        env, agent = self.get_env_agent()
        train_recording(env, agent, trajectories)
        print(w)
        print(agent.Q.w)
        self.assertL2(agent.Q.w, w, tol=self.tol_w)

    pass
class LinearSarsaAgentQuestion(LinearAgentStub):
    """ Sarsa Agent with linear function approximators """

    def get_env_agent(self):
        env = self.get_env()
        from irlc.ex11.semi_grad_sarsa import LinearSemiGradSarsa
        agent = LinearSemiGradSarsa(env, gamma=1, alpha=self.alpha, epsilon=0)
        return env, agent

    def test_Q_weight_vector_w(self):
        self.tol_w = 1.4
        self.chk_Q_weight_vector_w()

class LinearQAgentQuestion(LinearAgentStub):
    """ Test of Linear Q Agent """

    def get_env_agent(self):
        env = self.get_env()
        alpha = 0.1
        from irlc.ex11.semi_grad_q import LinearSemiGradQAgent
        agent = LinearSemiGradQAgent(env, gamma=1, alpha=alpha, epsilon=0)
        return env, agent

    def test_Q_weight_vector_w(self):
        # self.tol_qs = 1.9
        self.tol_w = 7
        self.chk_Q_weight_vector_w()


class Week11Tests(Report):
    title = "Tests for week 11"
    pack_imports = [irlc]
    individual_imports = []
    questions =[
        # (NStepSarseEvaluationQuestion, 10),
        (QAgentQuestion, 10),
        (LinearQAgentQuestion, 10),
        (LinearSarsaAgentQuestion, 10),
        (SarsaQuestion, 10),
        (NStepSarsaQuestion, 5),
        ]
if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week11Tests())
