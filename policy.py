import numpy as np


def PolicyIteration(env, gamma=0.99, theta=1e-8):
    ## CHAT-GPT:
    nS = env.observation_space.n
    nA = env.action_space.n
    
    policy = np.ones((nS, nA)) / nA
    V = np.zeros(nS)

    P = env.unwrapped.P

    # Policy Iteration
    is_policy_stable = False
    while not is_policy_stable:
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for s in range(nS):
                if s in env.unwrapped._cliff.flatten().nonzero()[0] or s == nS-1:
                    continue
                v = V[s]
                V[s] = sum(policy[s,a] * sum(prob * (reward + gamma * V[next_s])
                        for prob, next_s, reward, done in P[s][a]) for a in range(nA))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        is_policy_stable = True
        for s in range(nS):
            if s in env.unwrapped._cliff.flatten().nonzero()[0] or s == nS-1:
                continue
            old_action = np.argmax(policy[s])
            # Compute action-values
            Q_s = np.array([sum(prob * (reward + gamma * V[next_s]) for prob, next_s, reward, done in P[s][a])
                            for a in range(nA)])
            best_action = np.argmax(Q_s)
            if old_action != best_action:
                is_policy_stable = False
            # Update policy to be greedy
            policy[s] = np.eye(nA)[best_action]
    return policy, V


def MonteCarlo(env, policy, s_init=143, gamma=0.99, total_episodes=1000, epsilon=0.1):
    returns = []
    for _ in range(total_episodes):
        env.reset()
        env.unwrapped.s = s_init
        s = s_init
        G = 0.0
        discount = gamma
        done = False
        
        while not done:
            if np.random.rand() < epsilon:
                a = np.random.randint(4)
            else:
                a = policy[s].argmax()
            next_s, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G += discount * reward
            discount *= gamma
            s = next_s
        
        returns.append(G)
    returns.sort()
    return np.array(returns)