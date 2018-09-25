import gym
#from gym import wrappers
import mujoco_py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

env = gym.make('Hopper-v2')
#env = wrappers.Monitor(env, './log', video_callable=lambda episode_id: episode_id%5==0)
#env = wrappers.Monitor(env, 'Hopper')
#print(env.action_space.high[0])
#print(env.action_space)		#3
#print(env.observation_space)	#11
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
toTensor = torch.Tensor
FloatTensor = torch.FloatTensor

Na = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
Ns = env.observation_space.shape[0]
EPISODE = 500
BUFFER_SIZE = 1e5
BATCH_SIZE = 256
GAMMA = 0.99
LR_C = 1e-3
LR_A = 1e-4
TAU = 1e-3

class RelayBuffer:
	def __init__(self, BUFFER_SIZE):
		self.buffer_size = BUFFER_SIZE
		self.memory = []

	def push(self, data):
		self.memory.append(data)
		if len(self.memory) > self.buffer_size:
			del self.memory[0]

	def sample(self, BATCH_SIZE):
		return random.sample(self.memory, BATCH_SIZE)

	def __len__(self):
		return len(self.memory)

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


class ActorNet(nn.Module):
	def __init__(self, ipt = Ns, opt = Na, dis = 0.003):
		super(ActorNet, self).__init__()
		self.fc1 = nn.Linear(ipt, 400)
		self.fc2 = nn.Linear(400, 300)
		self.fc3 = nn.Linear(300, opt)
		self.tanh = nn.Tanh()
		self.init_weight(dis)
	
	def init_weight(self, dis):
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		self.fc3.weight.data.uniform_(-dis, dis)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.tanh(self.fc3(x))
		return x


class CriticNet(nn.Module):
	def __init__(self, ipt = Ns, opt = 1, dis = 0.0003):
		super(CriticNet, self).__init__()
		self.fc1 = nn.Linear(ipt, 400)
		self.fc2 = nn.Linear(400+Na, 300)
		self.fc3 = nn.Linear(300, opt)
		self.init_weight(dis)
	
	def init_weight(self, dis):
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		self.fc3.weight.data.uniform_(-dis, dis)
	
	def forward(self, xs):
		x, a = xs
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(torch.cat([x, a], 1)))
		x = self.fc3(x)
		return x

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)



actor_target_net = ActorNet().cuda()
actor_policy_net = ActorNet().cuda()

critic_target_net = CriticNet().cuda()
critic_policy_net = CriticNet().cuda()

if os.path.isfile('actor_target.pth'):
	actor_target_net.load_state_dict(torch.load('actor_target.pth'))
if os.path.isfile('actor_policy.pth'):
	actor_policy_net.load_state_dict(torch.load('actor_policy.pth'))
if os.path.isfile('critic_target.pth'):
	critic_target_net.load_state_dict(torch.load('critic_target.pth'))
if os.path.isfile('critic_policy.pth'):
	critic_policy_net.load_state_dict(torch.load('critic_policy.pth'))

relay_buffer = RelayBuffer(BUFFER_SIZE)
if os.path.isfile('BUFFER.pth'):
	relay_buffer = torch.load('BUFFER.pth')

optimizer_critic = optim.Adam(critic_policy_net.parameters(), lr = LR_C, weight_decay = 0.01)
optimizer_actor = optim.Adam(actor_policy_net.parameters(), lr = LR_A)

noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(Na))
def select_action(obs, n_step):
	state = Variable(toTensor([obs])).cuda()
	action = actor_policy_net(state).data.cpu() + toTensor(noise())
	return action


def critic_loss_func(predicted, target):
	return torch.sum((target - predicted)**2) / BATCH_SIZE

def soft_update(target, policy, tau):
	for target_param, policy_param in zip(target.parameters(), policy.parameters()):
		target_param.data = tau * policy_param + (1-tau) * target_param


def train():
	#sampling
	if len(relay_buffer) < BATCH_SIZE:
		return
	else:
		sample_batch = relay_buffer.sample(BATCH_SIZE)
	s, a, r, _s, D = zip(*sample_batch)
	state_batch = Variable(torch.cat(s, 0)).cuda()
	action_batch = Variable(torch.cat(a, 0)).cuda()
	reward_batch = Variable(torch.cat(r, 0)).cuda()
	_state_batch = Variable(torch.cat(_s, 0)).cuda()

	#compute y
	optimizer_critic.zero_grad()
	ya = actor_target_net(_state_batch)
	ys = critic_target_net([_state_batch, ya])
	y = reward_batch + GAMMA * ys

	for i in range(len(D)):
		if D[i]:
			y[i] = 0

	#compute predicted value of critic policy net
	predicted = critic_policy_net([state_batch, action_batch])

	#compute loss
	critic_loss = critic_loss_func(predicted, y)
	critic_loss.backward()
	optimizer_critic.step()

	#Actor part
	optimizer_actor.zero_grad()
	act = actor_policy_net(state_batch)
	predicted = -critic_policy_net([state_batch, act])
	actor_loss = predicted.mean()
	actor_loss.backward()
	optimizer_actor.step()

	#update target network
	soft_update(critic_target_net, critic_policy_net, TAU)
	soft_update(actor_target_net, actor_policy_net, TAU)



timer = 0
R = 0
n_step = 0
Return = []
return_writer = csv.writer(open("./Return.csv", 'w'))
for episode in range(EPISODE):
	obs = env.reset()
	done = False
	timer = 0
	R = 0
	while not done:
		action = select_action(obs, n_step)
		action = torch.clamp(action, min = -1, max = 1)
		step_action = action.max(1)[0].numpy()
		obs_, reward, done, _ = env.step(step_action[0])

		transition = [
			FloatTensor([obs]),
			action,
			FloatTensor([reward]),
			FloatTensor([obs_]),
			done
		]
		relay_buffer.push(transition)

		train()
		R += reward
		timer += 1
		n_step += 1
		obs = obs_
	return_writer.writerow([R])
	Return.append(R)
	print('Episode: %3d,\tStep: %5d,\tReturn: %f' %(episode, timer, R))
	torch.save(actor_target_net.state_dict(), 'actor_target.pth')
	torch.save(actor_policy_net.state_dict(), 'actor_policy.pth')
	torch.save(critic_target_net.state_dict(), 'critic_target.pth')
	torch.save(critic_policy_net.state_dict(), 'critic_policy.pth')
	torch.save(relay_buffer, 'BUFFER.pth')

