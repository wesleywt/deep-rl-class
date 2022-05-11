from pyvirtualdisplay import Display
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from huggingface_sb3 import package_to_hub

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from huggingface_sb3 import package_to_hub

env = make_vec_env('LunarLander-v2', n_envs=16)

# You have to use the full path for the load to work.
model = PPO.load("/home/wesley/PycharmProjects/deep-rl-class/unit1/ppo-LunarLander-v2")
env_id = "LunarLander-v2"
model_name = "ppo-LunarLander-v2"
model_architecture = "PPO"
repo_id = "wesleywt/ppo-LunarLander-v2"
commit_message = "Trained model"
eval_env = DummyVecEnv([lambda : gym.make(env_id)])
package_to_hub(model=model,
               model_name=model_name,
               model_architecture=model_architecture,
               env_id=env_id,
               eval_env=eval_env,
               repo_id=repo_id,
               commit_message=commit_message)