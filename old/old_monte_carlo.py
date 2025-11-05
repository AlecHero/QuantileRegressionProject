def monte_carlo_returns(env, policy, start_state, gamma=0.99, episodes=10_000):
    returns = []

    for _ in range(episodes):
        episode = []
        s, _ = env.reset()
        done = False

        while not done:
            a = policy[s]
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            episode.append((s, r))
            s = s_next

        # Compute return from start_state
        G = 0
        seen_start = False
        for t in reversed(range(len(episode))):
            s_t, r_t = episode[t]
            G = gamma * G + r_t
            if s_t == start_state:
                seen_start = True
        if seen_start:
            returns.append(G)

    return np.array(returns)

returns = monte_carlo_returns(env, policy.argmax(1), 143)
returns.sort()