from gymnasium.envs.registration import register

register(
    id="CliffSlippery",
    kwargs={"is_slippery": True, "rewards":{"step":-1,"goal":10.0,"fail":-100.0}},
    entry_point="envs.gridworld:CliffWalking",
)

register(
    id="CliffSlipperyLow",
    kwargs={"is_slippery": True, "rewards":{"step":0,"goal":1.0,"fail":-1.0}},
    entry_point="envs.gridworld:CliffWalking",
)

register(
    id="CliffWindy",
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
    kwargs={"p_random":0.1, "rewards":{"step":0.0,"goal":1.0,"fail":0.0}},
    entry_point="envs.gridworld:WindyRooms",
)

register(
    id="WindyRoomsNegative",
    kwargs={"p_random":0.1, "rewards":{"step":0.0,"goal":1.0,"fail":-1.0}},
    entry_point="envs.gridworld:WindyRoomsNegative",
)