import numpy as np


def PolicyIteration(env, gamma=0.99, theta=1e-8):
    ## CHAT-GPT:
    nS = env.observation_space.n
    nA = env.action_space.n
    
    policy = np.ones((nS, nA)) / nA
    V = np.zeros(nS)

    P = env.unwrapped.P
    desc = env.unwrapped.desc

    # Policy Iteration
    is_policy_stable = False
    while not is_policy_stable:
        # --- Policy Evaluation ---
        while True:
            delta = 0
            for s in range(nS):
                try:
                    if s in env.unwrapped._cliff.flatten().nonzero()[0] or s == nS-1:
                        continue
                except: pass
                try:
                    _pos = np.unravel_index(s, desc.shape)
                    if desc[_pos] in b"GH": continue
                except: pass
                    
                v = V[s]
                V[s] = sum(policy[s,a] * sum(prob * (reward + gamma * V[next_s])
                        for prob, next_s, reward, done in P[s][a]) for a in range(nA))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # --- Policy Improvement ---
        is_policy_stable = True
        for s in range(nS):
            try:
                if s in env.unwrapped._cliff.flatten().nonzero()[0] or s == nS-1:
                    continue
            except: pass
            try:
                _pos = np.unravel_index(s, desc.shape)
                if desc[_pos] in b"GH": continue
            except: pass
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


def MonteCarlo(env, policy, params, s_init, rng:np.random.Generator, epsilon=0.0, total_episodes=1_000):
    from tqdm import tqdm
    returns = []
    nA = params.action_size
    for _ in tqdm(range(total_episodes)):
        env.reset()
        env.unwrapped.s = s_init
        s = s_init
        G = 0.0
        discount = params.gamma
        done = False
        
        while not done:
            if rng.random() < epsilon:
                a = rng.integers(nA)
            else:
                a = rng.choice(np.flatnonzero(policy[s] == policy[s].max()))
            
            next_s, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G += discount * reward
            discount *= params.gamma
            s = next_s
        
        returns.append(G)
    return np.array(returns)