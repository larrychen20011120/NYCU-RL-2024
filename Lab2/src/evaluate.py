from dqn_agent_atari import AtariDQNAgent, AtariDDQNAgent, AtariDuelDQNAgent
import argparse
from datetime import datetime

import gym

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack
from gym.wrappers import RecordVideo

import warnings

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str,
                        help="Path of the learned model weight")
    parser.add_argument("--model", type=str, choices=["DQN", "DDQN", "DuelDQN"],
                        help="Select three model types: DQN, DDQN, DuelDQN")
    parser.add_argument("--game", type=str, choices=["MsPacman", "Enduro"],
                        help="Record as video or just render it on window")
    parser.add_argument("--mode", type=str, choices=["video", "print"], default="video",
                        help="Record as video or just print it on terminal")

    args = parser.parse_args()

    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 1e7,
		"gamma": 0.99,
		"batch_size": 64,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		"logdir": f'log/{args.game}-v5/DQN',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 5,
		"env_id": f'ALE/{args.game}-v5',
        "max_n_models": 50,
	}

    if args.model == "DQN":
        agent = AtariDQNAgent(config)
    elif args.model == "DDQN":
        agent = AtariDDQNAgent(config)
    elif args.model == "DuelDQN":
        agent = AtariDuelDQNAgent(config)
    else:
        print("no such model!!")
        exit(-1)


    if args.mode == "video":

        # load the weight
        agent.load(args.weight)
        env = gym.make(config["env_id"], render_mode="rgb_array")
        env = atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
        env = FrameStack(env, 4)

        # record only one testing result
        env = RecordVideo(
            env=env, 
            video_folder=f"video/{args.game}/{args.model}/", 
            name_prefix=datetime.now().strftime("%m-%d-%H:%M:%S"), 
            episode_trigger=lambda x: x % 2 == 0
        )
        observation, info = env.reset()
        total_reward = 0

        # Start the recorder
        env.start_video_recorder()

        while True:
            action = agent.decide_agent_actions(observation, agent.eval_epsilon, env.action_space)
            next_observation, reward, terminate, truncate, info = env.step(action)
            total_reward += reward

            if terminate or truncate:
                break
                
            observation = next_observation
        
        print("Total Reward:", total_reward)

        # close the video recorder before the env!
        env.close_video_recorder()

        # Close the environment
        env.close()
    else:
        agent.load_and_evaluate(args.weight)

    