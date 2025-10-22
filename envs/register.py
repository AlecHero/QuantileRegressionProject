from gymnasium.envs.registration import register
from gymnasium.envs import registry

def register_envs():
    if not registry.get("CliffCustomSlippery-v1"):
        register(
            id="CliffCustomSlippery-v1",
            kwargs={"is_slippery": True},
            entry_point="envs.cliffcustom:CliffCustomEnv",
        )
    if not registry.get("CliffCustomSlipperyLow-v1"):
        register(
            id="CliffCustomSlipperyLow-v1",
            kwargs={"is_slippery": True, "rewards":{"step":0,"goal":1.0,"fail":-10.0}},
            entry_point="envs.cliffcustom:CliffCustomEnv",
        )
    if not registry.get("WindyRooms-v1"):
        register(
            id="WindyRooms-v1",
            entry_point="envs.windyrooms:WindyRoomsEnv",
        )
    if not registry.get("Simple-v1"):
        register(
            id="Simple-v1",
            entry_point="envs.simple:SimpleEnv",
        )