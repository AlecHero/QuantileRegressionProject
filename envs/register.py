from gymnasium.envs.registration import register
from gymnasium.envs import registry

def register_envs():
    if not registry.get("CliffCustom-v1"):
        register(
            id="CliffCustom-v1",
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