from gymnasium.envs.registration import register


register(
    id="CliffSlippery",
    kwargs={"is_slippery": True, "rewards":{"step":0,"goal":1.0,"fail":-1.0}},
    entry_point="envs.gridworld:CliffWalking",
)

register(
    id="CliffSimple",
    kwargs={"is_windy": True, "p_random":0.25, "rewards":{"step":0,"goal":1.0,"fail":-1.0}},
    entry_point="envs.gridworld:CliffWalking",
)

register(
    id="Walkway",
    kwargs={"rewards":{"step":-1.0,"goal":1.0,"fail":0.0}, "shape":(1,10)},
    entry_point="envs.gridworld:Walkway",
)

register(
    id="WindyRooms",
    kwargs={"rewards":{"step":0.0,"goal":1.0,"fail":0.0}},
    entry_point="envs.gridworld:WindyRooms",
)

register(
    id="Simple",
    entry_point="envs.simple:SimpleEnv",
)