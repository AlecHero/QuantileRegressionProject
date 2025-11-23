from gymnasium.envs.registration import register


register(
    id="CliffCustomSlippery-v1",
    kwargs={"is_slippery": True},
    entry_point="envs.cliffcustom:CliffCustomEnv",
)

register(
    id="CliffMini",
    kwargs={"is_windy": True, "p_random":0.25, "rewards":{"step":0,"goal":1.0,"fail":-1.0}},
    entry_point="envs.cliffcustom:CliffCustomEnv",
)

register(
    id="CliffSimple-v1",
    kwargs={"is_windy": True, "p_random":0.25, "rewards":{"step":0,"goal":1.0,"fail":-1.0}},
    entry_point="envs.cliffcustom:CliffCustomEnv",
)

register(
    id="CliffCustomSlipperyLow-v1",
    kwargs={"is_slippery": True, "rewards":{"step":0,"goal":1.0,"fail":0.0}},
    entry_point="envs.cliffcustom:CliffCustomEnv",
)

register(
    id="WindyRooms-v1",
    entry_point="envs.windyrooms:WindyRoomsEnv",
)

register(
    id="WindyRoomsHard-v1",
    kwargs={"p_random":0.2},
    entry_point="envs.windyrooms:WindyRoomsEnv",
)

register(
    id="WindyRoomsEasy-v1",
    kwargs={"rewards":{"step":-1.0,"goal":100.0}},
    entry_point="envs.windyrooms:WindyRoomsEnv",
)

register(
    id="Windy-v1",
    kwargs={"is_two_room": False},
    entry_point="envs.windyrooms:WindyRoomsEnv",
)

register(
    id="WindyHard-v1",
    kwargs={"is_two_room": False, "p_random":0.3},
    entry_point="envs.windyrooms:WindyRoomsEnv",
)

register(
    id="WindyVeryHard-v1",
    kwargs={"p_random":0.5},
    entry_point="envs.windyrooms:WindyRoomsEnv",
)

register(
    id="Simple-v1",
    entry_point="envs.simple:SimpleEnv",
)