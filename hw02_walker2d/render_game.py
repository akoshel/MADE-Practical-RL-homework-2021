import pybullet_envs
import time
from gym import make

ENV_NAME = "Walker2DBulletEnv-v0"


def main() -> None:
    env = make(ENV_NAME)
    env.render()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            time.sleep(0.01)


if __name__ == "__main__":
    main()
