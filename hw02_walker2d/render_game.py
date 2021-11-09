import pybullet_envs
import time
import torch
from gym import make
from train import PPO


ENV_NAME = "Walker2DBulletEnv-v0"


def main() -> None:
    env = make(ENV_NAME)
    ppo = PPO(env.observation_space.shape[0], env.action_space.shape[0])
    with open("actor.pth", "rb") as fp:
        actor_dict = torch.load(fp, map_location="cpu")
    with open("critic.pth", "rb") as fp:
        critic_dict = torch.load(fp, map_location="cpu")
    # game_manager_dqn.dqn.model.load_state_dict(state_dict)
    ppo.actor.model.load_state_dict(actor_dict)
    ppo.critic.model.load_state_dict(critic_dict)
    env.render()
    while True:
        state = env.reset()
        for _ in range(1000):
            state, reward, done, _ = env.step(ppo.act(state)[0])
            time.sleep(0.01)


if __name__ == "__main__":
    main()
