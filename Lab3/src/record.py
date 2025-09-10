import gym

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack
from gym.wrappers import RecordVideo

from ppo_agent_atari import AtariPPOAgent

if __name__ == '__main__':

    config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": 'log/Enduro_release2/',
		"update_ppo_epoch": 3,
		"learning_rate": 1e-6,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"env_id": 'ALE/Enduro-v5',
		"eval_interval": 100,
		"eval_episode": 3,
        "max_n_models": 20,
	}

    agent = AtariPPOAgent(config)
    # load the weight
    agent.load("log/Enduro_release2/model_99896002_2223.pth")


    env = gym.make(config["env_id"], render_mode="rgb_array")
    env = atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
    env = FrameStack(env, 4)

    # record only one testing result
    env = RecordVideo(
        env=env, 
        video_folder=f"video/", 
        name_prefix="result", 
        episode_trigger=lambda x: x % 2 == 0
    )
    observation, info = env.reset()
    total_reward = 0

    # Start the recorder
    env.start_video_recorder()

    while True:
        action, _, _ = agent.decide_agent_actions(observation, eval=True)
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