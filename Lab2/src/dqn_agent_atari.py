import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent, DQNBaseAgentParallel
from models.atari_model import AtariNetDQN, AtariNetDuelDQN
import gym
import random

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = atari_preprocessing.AtariPreprocessing(self.env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.env = FrameStack(self.env, 4)

		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4) 
	
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		if random.random() < epsilon:
			# exploration by randomly pick the action
			action = action_space.sample()
		else:
			observation = torch.FloatTensor(np.array(observation)).unsqueeze(0).to(self.device)
			with torch.no_grad():
				q_value = self.behavior_net(observation)
				action = q_value.argmax(dim=1).item()

		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		
		q_value = self.behavior_net(state).gather(1, action.to(dtype=torch.int64))
		with torch.no_grad():
			q_next = self.target_net(next_state)
			q_next = q_next.max(1)[0].unsqueeze(1)

			# if episode terminates at next_state, then q_target = reward
			q_target = reward + self.gamma * q_next * (1 - done)
        
		
		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DDQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	

class AtariDDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = atari_preprocessing.AtariPreprocessing(self.env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.env = FrameStack(self.env, 4)

		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4) 
	
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		if random.random() < epsilon:
			# exploration by randomly pick the action
			action = action_space.sample()
		else:
			observation = torch.FloatTensor(np.array(observation)).unsqueeze(0).to(self.device)
			with torch.no_grad():
				q_value = self.behavior_net(observation)
				action = q_value.argmax(dim=1).item()
		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		
		q_value = self.behavior_net(state).gather(1, action.to(dtype=torch.int64))

		with torch.no_grad():
			# find the best action from behavior network
			q_next = self.behavior_net(next_state)
			next_action = q_next.argmax(dim=1, keepdim=True)
			# use target network to do such action
			q_next = self.target_net(next_state).gather(1, next_action)

			# if episode terminates at next_state, then q_target = reward
			q_target = reward + self.gamma * q_next * (1 - done)
        
		
		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DDQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()


class AtariDuelDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDuelDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = atari_preprocessing.AtariPreprocessing(self.env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.env = FrameStack(self.env, 4)

		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDuelDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDuelDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4) 
	
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		if random.random() < epsilon:
			# exploration by randomly pick the action
			action = action_space.sample()
		else:
			observation = torch.FloatTensor(np.array(observation)).unsqueeze(0).to(self.device)
			with torch.no_grad():
				q_value = self.behavior_net(observation)
				action = q_value.argmax(dim=1).item()
		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		
		q_value = self.behavior_net(state).gather(1, action.to(dtype=torch.int64))

		with torch.no_grad():
			# find the best action from behavior network
			q_next = self.behavior_net(next_state)
			next_action = q_next.argmax(dim=1, keepdim=True)
			# use target network to do such action
			q_next = self.target_net(next_state).gather(1, next_action)

			# if episode terminates at next_state, then q_target = reward
			q_target = reward + self.gamma * q_next * (1 - done)
        
		
		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DDQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()


class AtariDQNAgentParallel(DQNBaseAgentParallel):
	def __init__(self, config):
		super(AtariDQNAgentParallel, self).__init__(config)
		### TODO ###
		# initialize env
		self.envs = gym.vector.SyncVectorEnv([
			lambda: self.__make_wrap(config["env_id"])
			for _ in range(self.num_envs)	
		])

		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.envs.single_action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.envs.single_action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4) 

	def __make_wrap(self, env_id):
		env = gym.make(env_id)
		env = atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
		env = FrameStack(env, 4)
		return env
	
		
	def decide_agent_actions(self, observations, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		if random.random() < epsilon:
			# exploration by randomly pick the action
			actions = np.array([ action_space.sample() for _ in range(len(observations))])
		else:
			observation = torch.FloatTensor(np.array(observations)).to(self.device)
			with torch.no_grad():
				q_value = self.behavior_net(observation)
				actions = q_value.argmax(dim=1).cpu()

		return actions
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		
		q_value = self.behavior_net(state).gather(1, action.to(dtype=torch.int64))
		with torch.no_grad():
			q_next = self.target_net(next_state)
			q_next = q_next.max(1)[0].unsqueeze(1)

			# if episode terminates at next_state, then q_target = reward
			q_target = reward + self.gamma * q_next * (1 - done)
        
		
		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DDQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()